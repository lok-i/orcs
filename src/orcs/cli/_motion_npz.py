"""Writing an orcs-native `motion.npz`: IL joint order, IL body rows, FK.

Dataset-blind — the half of staging that is the same whether the clip came from
a terrain dataset, a retargeting run, or a generator. Every producer of
`motion.npz` goes through here, because the two conventions below are the ones a
second copy would silently get wrong.

Under `cli/`, not `core/`: it builds a robot, and `core` may not import `assets`
(docs/ethos.md §4). Staging-time only — the runtime never imports this.

**Body-array convention.** `body_*_w` is (T, 37, ...) in ISAACLAB body order and
only the `G1_TRACKED_BODIES` rows are filled; the rest are zeros. No canonical
37-name IsaacLab body list exists in any dependency and reconstructing it by BFS
is off-by-one against the flat-hand robot (IsaacLab keeps fixed links MuJoCo
merges away), so inventing one would be a silent misalignment. The loader reads
exactly those 14 rows and nothing else touches the array; `il_body_names()`
writes the legend so the file explains itself.

**Joint order.** `motion.npz` means IsaacLab order and the runtime loader applies
`IL2MJ` to get MuJoCo order. :class:`Fk` asserts that identity by NAME at
construction — a "rigid span" check cannot do this job, since a span that is
rigid is invariant to any joint values, wrong ones included.
"""

from __future__ import annotations

import numpy as np
import torch
from mjlab.scene import Scene, SceneCfg
from mjlab.sim.sim import Simulation, SimulationCfg
from mjlab.terrains import TerrainEntityCfg
from mocke.mdp.joint_maps import G1_TRACKED_BODIES, IL2MJ
from mocke.sonic import profile

from orcs.assets import get_g1_flat_hand_cfg

__all__ = ["N_IL_BODIES", "Fk", "il_body_names", "il_joint_names"]

N_IL_BODIES = 37
"""Row count of `body_*_w`, matching the existing retargeted dataset."""


def il_joint_names(mj_joint_names: list[str]) -> list[str]:
    """IsaacLab BFS joint order, derived — not hardcoded.

    `IL2MJ[k]` is the IL slot of MJ slot k (that is what makes `data[:, IL2MJ]`
    an IL->MJ conversion), so scattering the MJ names through it reconstructs
    the IL order exactly. Verified equal to the retargeting repo's own
    `ISAAC_JOINT_NAMES`.
    """
    out: list[str | None] = [None] * len(IL2MJ)
    for mj_slot, il_slot in enumerate(IL2MJ):
        out[il_slot] = mj_joint_names[mj_slot]
    assert all(n is not None for n in out), "IL2MJ is not a permutation"
    return out  # type: ignore[return-value]


def il_body_names() -> list[str]:
    """The `body_names` legend — tracked rows named, unfilled rows blank."""
    names = [""] * N_IL_BODIES
    for name, il_idx in G1_TRACKED_BODIES:
        names[il_idx] = name
    return names


class Fk:
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
        out = {k: np.zeros((n, N_IL_BODIES, d), dtype=np.float32)
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


def default_stand(device: str) -> tuple[torch.Tensor, torch.Tensor, float]:
    """The G1's nominal stand: (joint_pos IL-ordered, root_quat, root_z).

    Read off the robot entity's own `init_state` — the SAME pose the action term
    offsets from (`use_default_offset=True`) and `reset_scene_to_default` writes.
    That identity is what makes a zero adapter action mean "hold the reference".
    """
    scene = Scene(
        SceneCfg(num_envs=1, env_spacing=0.0,
                 terrain=TerrainEntityCfg(terrain_type="plane"),
                 entities={"robot": profile.robot_cfg(base=get_g1_flat_hand_cfg())}),
        device=device,
    )
    model = scene.compile()
    sim = Simulation(num_envs=1, cfg=SimulationCfg(), model=model, device=device)
    scene.initialize(sim.mj_model, sim.model, sim.data)
    robot = scene["robot"]
    jp_mj = robot.data.default_joint_pos[0].detach().cpu()  # MuJoCo order
    jp_il = torch.zeros_like(jp_mj)
    jp_il[IL2MJ] = jp_mj  # scatter MJ -> IL (inverse of the loader's gather)
    root = robot.data.default_root_state[0].detach().cpu()
    return jp_il, root[3:7].clone(), float(root[2])
