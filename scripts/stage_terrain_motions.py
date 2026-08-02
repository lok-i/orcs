"""Stage a (terrain, motion) dataset into the orcs-native layout.

    python scripts/stage_terrain_motions.py --source omni

The rule this script exists to enforce: **the runtime loads only orcs-native
files.** Every dataset quirk — joint order, frame rate, quaternion convention,
zip packaging, terrain file format — dies here, so `orcs.core.data` never grows
a per-dataset branch.

Output layout, which IS the tile<->clip pairing (no manifest to desync):

    <out>/<family>/level_<L>/tile.json            terrain geometry, tile-local
    <out>/<family>/level_<L>/sample<N>/motion.npz orcs-native, IL order, 50 Hz
    <out>/<family>/level_<L>/sample<N>/metadata.json

`<family>` becomes a sub-terrain grid COLUMN and `<L>` a ROW, which is exactly
the `<dataset>/<motion>/<sampleN>` shape `orcs.core.data.scan` already walks.

Four transforms, in order:

  1. resample to 50 Hz          lerp position/joints, slerp orientation
  2. joint permute -> IL        BY NAME. IsaacLab BFS order is what motion.npz
                                means, and the loader's `IL2MJ` assumes it.
  3. velocities                 central difference (joints, root lin) and an
                                SO3 derivative (root ang) — mjlab's convention
  4. FK                         write the full state into the sim, read back
                                body poses/twists

**Body-array convention, read this before touching the output.** `motion.npz`
carries `body_*_w` as (T, 37, ...) in ISAACLAB body order, and only the 14
tracked rows are filled — the rest are zeros. No canonical 37-name IsaacLab
body list exists in any dependency, and reconstructing it by BFS is off-by-one
against the flat-hand robot (IsaacLab keeps fixed links MuJoCo merges away), so
inventing one would be a silent misalignment waiting to happen. The loader
reads exactly `mocke.mdp.joint_maps.G1_TRACKED_BODIES` and nothing else touches
the array, so the unfilled rows are unreachable. `body_names` is written into
the npz as a legend so the file explains itself.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from mjlab.scene import Scene, SceneCfg
from mjlab.sim.sim import Simulation, SimulationCfg
from mjlab.terrains import TerrainEntityCfg
from mjlab.utils.lab_api.math import (
    axis_angle_from_quat,
    quat_conjugate,
    quat_mul,
    quat_slerp,
)
from mocke.mdp.joint_maps import G1_TRACKED_BODIES, IL2MJ
from mocke.sonic import profile

from orcs.assets import get_g1_flat_hand_cfg
from orcs.core.paths import DATA_ROOT
from orcs.tasks.perloco.sources import SOURCES
from orcs.tasks.perloco.terrain_spec import ClipSpec, TileSpec

_N_IL_BODIES = 37
"""Row count of `body_*_w`, matching the existing retargeted dataset. Only the
`G1_TRACKED_BODIES` indices are filled — see the module docstring."""


# ---------------------------------------------------------------------------
# joint order
# ---------------------------------------------------------------------------

def il_joint_names(mj_joint_names: list[str]) -> list[str]:
    """IsaacLab BFS joint order, derived — not hardcoded.

    `IL2MJ[k]` is the IL slot of MJ slot k (that is what makes
    `data[:, IL2MJ]` an IL->MJ conversion), so scattering the MJ names through
    it reconstructs the IL order exactly. Verified equal to the retargeting
    repo's own `ISAAC_JOINT_NAMES`.
    """
    out: list[str | None] = [None] * len(IL2MJ)
    for mj_slot, il_slot in enumerate(IL2MJ):
        out[il_slot] = mj_joint_names[mj_slot]
    assert all(n is not None for n in out), "IL2MJ is not a permutation"
    return out  # type: ignore[return-value]


# ---------------------------------------------------------------------------
# resampling + velocities (mjlab csv_to_npz conventions)
# ---------------------------------------------------------------------------

def _resample(clip: ClipSpec, out_fps: float, device: str) -> dict:
    """Lerp position/joints, slerp orientation, onto an `out_fps` grid."""
    pos = torch.as_tensor(clip.root_pos, dtype=torch.float32, device=device)
    quat = torch.as_tensor(clip.root_quat, dtype=torch.float32, device=device)
    jnt = torch.as_tensor(clip.joint_pos, dtype=torch.float32, device=device)

    n_in = pos.shape[0]
    duration = (n_in - 1) / clip.fps
    times = torch.arange(0, duration, 1.0 / out_fps, device=device,
                         dtype=torch.float32)
    phase = times / duration
    i0 = (phase * (n_in - 1)).floor().long()
    i1 = torch.minimum(i0 + 1, torch.tensor(n_in - 1, device=device))
    blend = (phase * (n_in - 1) - i0).unsqueeze(1)

    quat_out = torch.zeros(len(times), 4, device=device)
    for k in range(len(times)):
        quat_out[k] = quat_slerp(quat[i0[k]], quat[i1[k]], float(blend[k]))
    return {
        "root_pos": pos[i0] * (1 - blend) + pos[i1] * blend,
        "root_quat": quat_out,
        "joint_pos": jnt[i0] * (1 - blend) + jnt[i1] * blend,
    }


def _velocities(state: dict, dt: float) -> dict:
    """Central differences for the linear channels, SO3 derivative for spin."""
    q = state["root_quat"]
    q_rel = quat_mul(q[2:], quat_conjugate(q[:-2]))
    omega = axis_angle_from_quat(q_rel) / (2.0 * dt)
    return {
        "root_lin_vel": torch.gradient(state["root_pos"], spacing=dt, dim=0)[0],
        "joint_vel": torch.gradient(state["joint_pos"], spacing=dt, dim=0)[0],
        "root_ang_vel": torch.cat([omega[:1], omega, omega[-1:]], dim=0),
    }


# ---------------------------------------------------------------------------
# forward kinematics
# ---------------------------------------------------------------------------

class _Fk:
    """Batched FK: B frames per sim.forward(), not one.

    One env per FRAME (not per clip — clips have different lengths), so a
    500-frame clip is 2 forward calls at B=256 instead of 500.
    """

    def __init__(self, batch: int, fps: float, device: str) -> None:
        self.batch, self.device = batch, device
        scene_cfg = SceneCfg(
            num_envs=batch,
            env_spacing=0.0,  # every env at the origin: FK is pose-only
            terrain=TerrainEntityCfg(terrain_type="plane"),
            entities={"robot": profile.robot_cfg(base=get_g1_flat_hand_cfg())},
        )
        self.scene = Scene(scene_cfg, device=device)
        model = self.scene.compile()
        sim_cfg = SimulationCfg()
        sim_cfg.mujoco.timestep = 1.0 / fps
        self.sim = Simulation(num_envs=batch, cfg=sim_cfg, model=model,
                              device=device)
        self.scene.initialize(self.sim.mj_model, self.sim.model, self.sim.data)
        self.robot = self.scene["robot"]

        # Entity-local names, NOT the compiled model's — Scene prefixes those
        # with the entity ("robot/left_hip_pitch_joint").
        self.mj_joint_names = list(self.robot.joint_names)
        # tracked body -> (row in the IL-ordered output, index in sim body order)
        self.tracked: list[tuple[int, int]] = []
        for name, il_idx in G1_TRACKED_BODIES:
            if name not in self.robot.body_names:
                raise ValueError(
                    f"tracked body '{name}' absent from the robot "
                    f"(have: {list(self.robot.body_names)})")
            self.tracked.append((il_idx, self.robot.body_names.index(name)))
        self.il_names = il_joint_names(self.mj_joint_names)

        # THE guard against a joint-order scramble, and it has to be by NAME.
        # A "rigid span" check cannot do this job: a span that is rigid is
        # invariant to ANY joint values, wrong ones included. What must hold is
        # the permutation identity the runtime loader relies on — applying
        # IL2MJ to an IL-ordered array yields MuJoCo order.
        permuted = [self.il_names[i] for i in IL2MJ]
        if permuted != self.mj_joint_names:
            bad = [(k, a, b) for k, (a, b) in
                   enumerate(zip(permuted, self.mj_joint_names, strict=True)) if a != b]
            raise AssertionError(
                "IL2MJ does not map this robot's IL order onto its MuJoCo "
                f"order; first mismatches (slot, got, want): {bad[:3]}")

    def __call__(self, state: dict, vel: dict) -> dict:
        """IL-ordered state -> IL-ordered body arrays.

        `joint_pos`/`joint_vel` arrive in **IsaacLab** order (what motion.npz
        means) and are permuted to **MuJoCo** order on the way into the sim —
        `IL2MJ` is exactly the permutation the runtime loader applies. Getting
        this backwards runs FK on scrambled joints: legs and torso still look
        plausible because several IL and MJ slots coincide, and the only loud
        symptom is a limb whose length is not constant.
        """
        n = state["root_pos"].shape[0]
        joint_pos_mj = state["joint_pos"][:, IL2MJ]
        joint_vel_mj = vel["joint_vel"][:, IL2MJ]
        out = {k: np.zeros((n, _N_IL_BODIES, d), dtype=np.float32)
               for k, d in (("body_pos_w", 3), ("body_quat_w", 4),
                            ("body_lin_vel_w", 3), ("body_ang_vel_w", 3))}
        origins = self.scene.env_origins

        for lo in range(0, n, self.batch):
            hi = min(lo + self.batch, n)
            k = hi - lo
            root = self.robot.data.default_root_state.clone()
            root[:k, 0:3] = state["root_pos"][lo:hi] + origins[:k]
            root[:k, 3:7] = state["root_quat"][lo:hi]
            root[:k, 7:10] = vel["root_lin_vel"][lo:hi]
            root[:k, 10:13] = vel["root_ang_vel"][lo:hi]
            self.robot.write_root_state_to_sim(root)

            jp = self.robot.data.default_joint_pos.clone()
            jv = self.robot.data.default_joint_vel.clone()
            jp[:k] = joint_pos_mj[lo:hi]
            jv[:k] = joint_vel_mj[lo:hi]
            self.robot.write_joint_state_to_sim(jp, jv)

            self.sim.forward()
            self.scene.update(self.sim.mj_model.opt.timestep)

            d = self.robot.data
            src = {
                "body_pos_w": d.body_link_pos_w - origins[:, None, :],
                "body_quat_w": d.body_link_quat_w,
                "body_lin_vel_w": d.body_link_lin_vel_w,
                "body_ang_vel_w": d.body_link_ang_vel_w,
            }
            for key, arr in src.items():
                a = arr[:k].detach().cpu().numpy()
                for il_row, sim_col in self.tracked:
                    out[key][lo:hi, il_row] = a[:, sim_col]
        # Root round-trip: the pelvis body must land exactly where we asked.
        # Catches a bad root write / frame convention, which FK alone hides.
        err = float(np.abs(out["body_pos_w"][:, dict(G1_TRACKED_BODIES)["pelvis"]]
                           - state["root_pos"].cpu().numpy()).max())
        if err > 1e-4:
            raise AssertionError(
                f"FK sanity: pelvis body position differs from the commanded "
                f"root position by {err:.2e} m")
        return out


# ---------------------------------------------------------------------------
# writing
# ---------------------------------------------------------------------------

def _write_tile(out_root: Path, tile: TileSpec) -> None:
    d = out_root / tile.key
    d.mkdir(parents=True, exist_ok=True)
    payload = {
        "family": tile.family,
        "level": tile.level,
        "boxes": [{"pos": list(b.pos), "quat": list(b.quat), "half": list(b.half)}
                  for b in tile.boxes],
    }
    if tile.hfield is not None:
        np.save(d / "hfield.npy", tile.hfield.heights.astype(np.float32))
        payload["hfield"] = {"file": "hfield.npy",
                             "size": list(tile.hfield.size),
                             "base": tile.hfield.base}
    (d / "tile.json").write_text(json.dumps(payload, indent=1) + "\n")


def _write_clip(sample_dir: Path, clip: ClipSpec, state: dict, vel: dict,
                bodies: dict, fps: float, il_names: list[str]) -> int:
    sample_dir.mkdir(parents=True, exist_ok=True)
    body_names = [""] * _N_IL_BODIES
    for name, il_idx in G1_TRACKED_BODIES:
        body_names[il_idx] = name
    np.savez(
        sample_dir / "motion.npz",
        fps=np.array([int(fps)]),
        joint_pos=state["joint_pos"].cpu().numpy().astype(np.float32),
        joint_vel=vel["joint_vel"].cpu().numpy().astype(np.float32),
        joint_names=np.array(il_names),
        body_names=np.array(body_names),
        **bodies,
    )
    (sample_dir / "metadata.json").write_text(json.dumps({
        "clip": clip.name,
        "family": clip.family,
        "level": clip.level,
        "frames": int(state["joint_pos"].shape[0]),
        **clip.meta,
    }, indent=1) + "\n")
    return int(state["joint_pos"].shape[0])


# ---------------------------------------------------------------------------

def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--source", default="omni", choices=sorted(SOURCES))
    ap.add_argument("--root", type=Path, default=None,
                    help="dataset root (default: per-source under ORCS_DATA_ROOT)")
    ap.add_argument("--out", type=Path,
                    default=DATA_ROOT / "terrain_motions",
                    help="staging root; <source> is appended")
    ap.add_argument("--families", nargs="*", default=None)
    ap.add_argument("--levels", nargs="*", type=float, default=None)
    ap.add_argument("--fps", type=float, default=50.0)
    ap.add_argument("--batch", type=int, default=256, help="FK frames per forward")
    ap.add_argument("--device", default="cuda:0")
    args = ap.parse_args()

    root = args.root or DATA_ROOT / SOURCES[args.source].default_root
    src = SOURCES[args.source](
        root,
        families=tuple(args.families) if args.families else None,
        levels=tuple(args.levels) if args.levels else None,
    )
    out_root = args.out / src.name
    device = args.device if torch.cuda.is_available() else "cpu"
    if device != args.device:
        print("[stage] CUDA unavailable, falling back to CPU")

    tiles = list(src.tiles())
    for tile in tiles:
        _write_tile(out_root, tile)
    print(f"[stage] {len(tiles)} tiles -> {out_root}")

    fk = _Fk(args.batch, args.fps, device)
    # Source joint order -> IL, BY NAME. A dataset that renames or reorders a
    # joint fails here instead of silently transposing the robot.
    per_tile: dict[str, int] = {}
    dt = 1.0 / args.fps
    total = 0
    for clip in src.clips():
        try:
            perm = [list(clip.joint_names).index(n) for n in fk.il_names]
        except ValueError as e:
            raise ValueError(
                f"{clip.name}: source joints do not cover the IsaacLab set "
                f"({e}). Source order: {clip.joint_names}") from None

        state = _resample(clip, args.fps, device)
        state["joint_pos"] = state["joint_pos"][:, perm]
        vel = _velocities(state, dt)
        bodies = fk(state, vel)

        n = per_tile.get(clip.tile_key, 0)
        per_tile[clip.tile_key] = n + 1
        frames = _write_clip(out_root / clip.tile_key / f"sample{n}", clip,
                             state, vel, bodies, args.fps, fk.il_names)
        total += frames
        print(f"[stage] {clip.tile_key}/sample{n}  {clip.name}  {frames} frames")

    print(f"[stage] {sum(per_tile.values())} clips over {len(per_tile)} tiles, "
          f"{total} frames @ {args.fps:g} Hz -> {out_root}")
    missing = [t.key for t in tiles if t.key not in per_tile]
    if missing:
        print(f"[stage] WARNING {len(missing)} tiles have no clip: {missing[:5]}")


if __name__ == "__main__":
    main()
