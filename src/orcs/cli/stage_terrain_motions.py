"""Stage a (terrain, motion) dataset into the orcs-native layout.

    python scripts/stage_terrain_motions.py --source omni

The rule this script exists to enforce: **the runtime loads only orcs-native
files.** Every dataset quirk — joint order, frame rate, quaternion convention,
zip packaging, terrain file format — dies here, so `orcs.core.data` never grows
a per-dataset branch.

Output layout, which IS the tile<->clip pairing (no manifest to desync):

    <out>/<family>/level_<L>/tile.json            terrain geometry, tile-local
    <out>/<family>/level_<L>/sample<N>/motion.npz orcs-native, IL order, 50 Hz
    <out>/<family>/level_<L>/sample<N>/smpl_motion.npz  --smpl only
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

Steps 2 and 4, and the body-array convention that goes with them, live in
`orcs.cli._motion_npz` — shared with every other producer of a `motion.npz`.
Read that module before touching the output.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from mjlab.utils.lab_api.math import (
    axis_angle_from_quat,
    quat_conjugate,
    quat_mul,
    quat_slerp,
)

from orcs.cli._motion_npz import Fk, il_body_names
from orcs.core.paths import DATA_ROOT
from orcs.tasks.perloco.sources import SOURCES
from orcs.tasks.perloco.terrain_spec import ClipSpec, SmplSpec, TileSpec

# ---------------------------------------------------------------------------
# resampling + velocities (mjlab csv_to_npz conventions)
# ---------------------------------------------------------------------------

def _lerp(a: torch.Tensor, i0: torch.Tensor, i1: torch.Tensor,
          blend: torch.Tensor) -> torch.Tensor:
    b = blend.reshape(-1, *([1] * (a.dim() - 1)))
    return a[i0] * (1 - b) + a[i1] * b


def _slerp(q: torch.Tensor, i0: torch.Tensor, i1: torch.Tensor,
           blend: torch.Tensor) -> torch.Tensor:
    out = torch.zeros(len(i0), 4, device=q.device)
    for k in range(len(i0)):
        out[k] = quat_slerp(q[i0[k]], q[i1[k]], float(blend[k]))
    return out


def _grid(n_in: int, phase: torch.Tensor) -> tuple[torch.Tensor, ...]:
    """Fractional phase in [0,1] -> (i0, i1, blend) into an n_in-frame array."""
    f = phase * (n_in - 1)
    i0 = f.floor().long()
    return i0, torch.minimum(i0 + 1, torch.tensor(n_in - 1, device=f.device)), f - i0


def _resample(clip: ClipSpec, out_fps: float, device: str) -> tuple[dict, torch.Tensor]:
    """Lerp position/joints, slerp orientation, onto an `out_fps` grid.

    Returns the phase vector too: every other channel of the clip resamples
    onto the SAME phase, which is the only alignment that survives a source
    whose reference and retarget have different durations (GRAIL's do).
    """
    pos = torch.as_tensor(clip.root_pos, dtype=torch.float32, device=device)
    quat = torch.as_tensor(clip.root_quat, dtype=torch.float32, device=device)
    jnt = torch.as_tensor(clip.joint_pos, dtype=torch.float32, device=device)

    n_in = pos.shape[0]
    duration = (n_in - 1) / clip.fps
    times = torch.arange(0, duration, 1.0 / out_fps, device=device,
                         dtype=torch.float32)
    phase = times / duration
    i0, i1, blend = _grid(n_in, phase)
    return {
        "root_pos": _lerp(pos, i0, i1, blend),
        "root_quat": _slerp(quat, i0, i1, blend),
        "joint_pos": _lerp(jnt, i0, i1, blend),
    }, phase


def _resample_smpl(smpl: SmplSpec, phase: torch.Tensor, device: str) -> dict:
    """The human reference onto the robot clip's phase grid. See `SmplSpec`."""
    j, v, q = (torch.as_tensor(a, dtype=torch.float32, device=device)
               for a in (smpl.joints, smpl.joints_viz, smpl.root_quat))
    i0, i1, blend = _grid(j.shape[0], phase)
    return {
        "smpl_joints": _lerp(j, i0, i1, blend),
        "smpl_root_quat_w": _slerp(q, i0, i1, blend),
        "smpl_joints_viz_w": _lerp(v, i0, i1, blend),
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
                bodies: dict, fps: float, il_names: list[str],
                smpl: dict | None = None) -> int:
    sample_dir.mkdir(parents=True, exist_ok=True)
    if smpl is not None:  # the -Smpl command space; contract in core.data.smpl
        np.savez(sample_dir / "smpl_motion.npz",
                 **{k: v.cpu().numpy().astype(np.float32) for k, v in smpl.items()})
    np.savez(
        sample_dir / "motion.npz",
        fps=np.array([int(fps)]),
        joint_pos=state["joint_pos"].cpu().numpy().astype(np.float32),
        joint_vel=vel["joint_vel"].cpu().numpy().astype(np.float32),
        joint_names=np.array(il_names),
        body_names=np.array(il_body_names()),
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
    ap.add_argument("--smpl", action="store_true",
                    help="also stage the SMPL-X human reference (grail only)")
    args = ap.parse_args()

    root = args.root or DATA_ROOT / SOURCES[args.source].default_root
    src = SOURCES[args.source](
        root,
        families=tuple(args.families) if args.families else None,
        levels=tuple(args.levels) if args.levels else None,
        **({"smpl": True} if args.smpl else {}),
    )
    out_root = args.out / src.name
    device = args.device if torch.cuda.is_available() else "cpu"
    if device != args.device:
        print("[stage] CUDA unavailable, falling back to CPU")

    tiles = list(src.tiles())
    for tile in tiles:
        _write_tile(out_root, tile)
    print(f"[stage] {len(tiles)} tiles -> {out_root}")

    fk = Fk(args.batch, args.fps, device)
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

        state, phase = _resample(clip, args.fps, device)
        state["joint_pos"] = state["joint_pos"][:, perm]
        vel = _velocities(state, dt)
        bodies = fk(state, vel)
        smpl = _resample_smpl(clip.smpl, phase, device) if clip.smpl else None

        n = per_tile.get(clip.tile_key, 0)
        per_tile[clip.tile_key] = n + 1
        frames = _write_clip(out_root / clip.tile_key / f"sample{n}", clip,
                             state, vel, bodies, args.fps, fk.il_names, smpl)
        total += frames
        print(f"[stage] {clip.tile_key}/sample{n}  {clip.name}  {frames} frames")

    print(f"[stage] {sum(per_tile.values())} clips over {len(per_tile)} tiles, "
          f"{total} frames @ {args.fps:g} Hz -> {out_root}")
    missing = [t.key for t in tiles if t.key not in per_tile]
    if missing:
        print(f"[stage] WARNING {len(missing)} tiles have no clip: {missing[:5]}")


if __name__ == "__main__":
    main()
