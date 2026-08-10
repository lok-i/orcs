"""ObjectMotionCommand — the OBJECT half of multi-clip motion tracking.

The clip library, robot RSI, phase annealing, last-frame freeze and the N-step
robot reference accessors are task-blind and live in
:class:`orcs.core.mdp.commands.MultiClipMotionCommand`. What this file adds is
everything that mentions an object:

  1. object tracking on the same timeline (_ConcatMotionLoader's extra channels)
  2. object RSI, conditional on init phase and reference contact
  3. object goal from the final clip frame + the `at_goal` metric, and the
     GOAL-DOMAIN anneal that walks a fraction of it off the demo manifold
  4. N-step future OBJECT reference accessors
  5. omni mode: env->object identity, so an env samples only ITS object's clips
  6. the ghost/goal/SMPL debug viz

Functionally 1:1 with fcrl's ObjectMotionCommand, built on mjlab.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Literal

import mujoco
import numpy as np
import torch
from mjlab.utils.lab_api.math import (
    matrix_from_quat,
    quat_error_magnitude,
    quat_from_euler_xyz,
    quat_mul,
)

from orcs.core.data.loader import ConcatMotionLoader
from orcs.core.data.scan import (
    load_field_or_make_zeros,
    motion_dirs,
    scan_flat,
)
from orcs.core.data.smpl import draw_smpl_ghost, load_smpl_channels
from orcs.core.mdp.commands import (
    MultiClipMotionCommand,
    MultiClipMotionCommandCfg,
    sample_se3,
)
from orcs.core.schedules import anneal_alpha
from orcs.tasks.uolm.mdp.contact_schedule import ContactSchedule
from orcs.tasks.uolm.mdp.demo_loader import get_motion_files_for_objects

if TYPE_CHECKING:
    from mjlab.envs import ManagerBasedRlEnv

__all__ = ["ObjectMotionCommandCfg", "ObjectMotionCommand", "motion_dirs"]

# Body-name roster for the tracked-body cfg default — canonical copy lives in
# mocke.mdp.joint_maps. The IL->MJ joint permutation and the tracked-body slice
# are applied by core's ConcatMotionLoader, not here.

_VIZ_FRAME_SCALE = 0.45  # goal/ref frame axis length (m)


# ---------------------------------------------------------------------------
# Concatenated multi-clip motion loader
# ---------------------------------------------------------------------------

_scan_flat_dataset = scan_flat
"""Deprecated alias for :func:`orcs.core.data.scan.scan_flat` — kept because a
downstream consumer (vibe's ONNX export) imports this name from here."""


class _ConcatMotionLoader(ConcatMotionLoader):
    """The robot timeline (core) + uolm's three extra channels.

    object tracking   obj_{pos,quat,lin_vel,ang_vel} from object_motion.npz
    SMPL reference    smpl_{joints,root_quat,joints_viz} from smpl_motion.npz
    contact schedule  ContactSchedule — the single source of contact truth;
                      the richer (T,N,N) narrowphase matrix subsumes the
                      deprecated scalar object_motion.npz["contact"] flag
                      (== object_robot_contact_any)

    All three ride the same concatenated timeline as the robot arrays, so a
    global frame index addresses every channel.
    """

    tag = "uolm"

    def _init_extra(self, contact_graph_body_names=None) -> None:
        self._contact_graph_body_names = contact_graph_body_names
        self._all_sj: list[torch.Tensor] = []
        self._all_sq: list[torch.Tensor] = []
        self._all_sv: list[torch.Tensor] = []
        self._all_op: list[torch.Tensor] = []
        self._all_oq: list[torch.Tensor] = []
        self._all_olv: list[torch.Tensor] = []
        self._all_oav: list[torch.Tensor] = []
        self._clip_lengths: list[int] = []

    def _load_extra(self, sample_dir, npz, n_frames: int) -> None:
        device = str(self.device)
        T = n_frames

        sj, sq, sv = load_smpl_channels(sample_dir, T, device)
        self._all_sj.append(sj)
        self._all_sq.append(sq)
        self._all_sv.append(sv)

        of = sample_dir / "object_motion.npz"
        od = np.load(of) if of.exists() else None
        obp, _ = load_field_or_make_zeros(od, "obj_pos_w", (T, 3), device)
        obq, exist = load_field_or_make_zeros(od, "obj_quat_w", (T, 4), device)
        if not exist:
            obq[:, 0] = 1.0
        oblv, _ = load_field_or_make_zeros(od, "obj_lin_vel_w", (T, 3), device)
        obav, _ = load_field_or_make_zeros(od, "obj_ang_vel_w", (T, 3), device)

        self._all_op.append(obp)
        self._all_oq.append(obq)
        self._all_olv.append(oblv)
        self._all_oav.append(obav)
        self._clip_lengths.append(T)

    def _finalize_extra(self) -> None:
        self.contact = ContactSchedule(
            self.motion_files, self._clip_lengths, self.device)

        # per-body contact-graph node vector, concatenated onto the same
        # timeline as obj_pos (only if a reward asks for it).
        names = self._contact_graph_body_names
        self.obj_bodywise_contact: torch.Tensor | None = (
            torch.cat([
                self.contact.body_object_contacts(i, names)
                for i in range(len(self.motion_files))
            ])  # (T_tot, K)
            if names else None
        )

        # (T_tot,) ANY-robot-body<->object contact — gates the conditional
        # object RSI (twist rand only where the robot has control-authority).
        self.obj_contact_any = torch.cat([
            self.contact.object_robot_contact_any(i)
            for i in range(len(self.motion_files))
        ])

        # ── SMPL human reference (smpl command space) ──
        self.smpl_joints = torch.cat(self._all_sj)     # (T_tot, 24, 3) RAW (encoder)
        self.smpl_root_quat = torch.cat(self._all_sq)  # (T_tot, 4) z-up, wxyz
        self.smpl_joints_viz = torch.cat(self._all_sv)  # (T_tot, 24, 3) z-up world

        # ── object tracking ──
        self.obj_pos = torch.cat(self._all_op)         # (T_tot, 3)
        self.obj_quat = torch.cat(self._all_oq)        # (T_tot, 4)
        self.obj_lin_vel = torch.cat(self._all_olv)    # (T_tot, 3)
        self.obj_ang_vel = torch.cat(self._all_oav)    # (T_tot, 3)


# ---------------------------------------------------------------------------
# ObjectMotionCommand
# ---------------------------------------------------------------------------

class ObjectMotionCommand(MultiClipMotionCommand):
    """MotionCommand + object tracking, multi-clip loader, RSI, phase annealing."""

    cfg: ObjectMotionCommandCfg

    def _build_loader(self) -> _ConcatMotionLoader:
        """The clip library, object-keyed in omni mode.

        Omni (ordered_object_names set): dataset_dir is the multi-dataset root,
        clips are object-keyed (each sample's metadata.json) and loaded grouped
        in object order, so clip->object is a repeat_interleave. Flat otherwise.
        """
        cfg = self.cfg
        if cfg.ordered_object_names:
            files_by_obj, files_ordered = get_motion_files_for_objects(
                list(cfg.ordered_object_names), cfg.dataset_dir,
                exclude_motions=list(cfg.exclude_motions or ()),
            )
            counts = [len(files_by_obj[n]) for n in cfg.ordered_object_names]
            self._clip_object_ids = torch.repeat_interleave(
                torch.arange(len(counts), device=self.device),
                torch.tensor(counts, device=self.device),
            )
        else:
            # flat scan honors exclude_motions (folder- or sample-level)
            files_ordered = (
                scan_flat(cfg.dataset_dir, cfg.exclude_motions)
                if cfg.exclude_motions else None
            )
            self._clip_object_ids = None
        return _ConcatMotionLoader(
            cfg.dataset_dir, self.device,
            contact_graph_body_names=cfg.contact_graph_body_names,
            motion_files=files_ordered,
        )

    def _init_task(self) -> None:
        cfg, env = self.cfg, self._env

        # Object entity in the scene
        self.object = env.scene[cfg.object_entity_name]

        # Omni mode: env->object identity from the sim's per-world variant
        # table (single source of truth — VariantEntityCfg assignment, fixed
        # at sim init). Variant order == ordered_object_names order by
        # construction (orcs.assets), so the table IS the object-id map.
        # Caching is sound here precisely BECAUSE it is fixed at sim init — a
        # task whose identity map moves at runtime must not copy this.
        if cfg.ordered_object_names:
            w2v = env.sim.world_to_variant.get(cfg.object_entity_name)
            assert w2v is not None, (
                f"ordered_object_names set but scene entity "
                f"'{cfg.object_entity_name}' has no variant table — spawn it "
                "via orcs.assets.omni_object_entity_cfg"
            )
            self._env_object_ids = w2v.to(device=self.device, dtype=torch.long)
            assert int(self._env_object_ids.max()) < len(cfg.ordered_object_names)
            self._clip_allowed = (
                self._clip_object_ids[None, :]
                == torch.arange(len(cfg.ordered_object_names),
                                device=self.device)[:, None]
            )  # (n_objects, n_clips)
        else:
            self._env_object_ids = None
            self._clip_allowed = None

        # Object goal: final-frame orientation per env (set at reset)
        self._object_goal_pos = torch.zeros(
            self.num_envs, 3, device=self.device
        )
        self._object_goal_quat = torch.zeros(
            self.num_envs, 4, device=self.device
        )
        self._object_goal_quat[:, 0] = 1.0

        # Goal-domain annealing (orcs.core.schedules): alpha_goal ramps
        # `alpha_goal_init` -> 1 = the fraction of envs whose goal is drawn from
        # the TASK domain instead of the demo's terminal pose.
        self.alpha_goal: float = cfg.alpha_goal_init
        self._goal_is_task = torch.zeros(
            self.num_envs, dtype=torch.bool, device=self.device
        )
        assert not (self.goal_anneals and self._sample_task_goal_quat(1) is None), (
            f"{type(self).__name__} has goal annealing configured but defines no "
            "task-domain goal set — override _sample_task_goal_quat."
        )
        self.metrics["alpha_goal"] = torch.zeros(self.num_envs, device=self.device)

        # Object tracking metrics
        self.metrics["error_object_pos"] = torch.zeros(self.num_envs, device=self.device)
        self.metrics["error_object_ori"] = torch.zeros(self.num_envs, device=self.device)
        # Object goal metrics
        self.metrics["error_object_pos_goal"] = torch.zeros(self.num_envs, device=self.device)
        self.metrics["error_object_ori_goal"] = torch.zeros(self.num_envs, device=self.device)
        self.metrics["at_goal"] = torch.zeros(self.num_envs, device=self.device)

    # ── goal-domain annealing ──

    @property
    def goal_anneals(self) -> bool:
        """Can a task-domain goal occur AT ALL this run? Run-level on purpose:
        a term incompatible with an off-manifold goal must switch off for EVERY
        env, or the reward function varies across envs in a way the critic
        cannot observe (it sees the goal, never the domain)."""
        return (self.cfg.goal_anneal_end > self.cfg.goal_anneal_start
                or self.cfg.alpha_goal_init > 0.0)

    @property
    def goal_is_task(self) -> torch.Tensor:
        """(B,) bool — this episode's goal is TASK-domain, not the demo's end
        pose. TELEMETRY ONLY; never gate a reward or termination on it — see
        `goal_anneals`."""
        return self._goal_is_task

    def _sample_task_goal_quat(self, n: int) -> torch.Tensor | None:
        """Task-domain goal orientations -> (n, 4), or None if this task defines
        none (then annealing is a config error, asserted at init)."""
        del n
        return None

    def _assign_goal_domain(self, env_ids: torch.Tensor) -> None:
        """Exact-count controller: keep the STANDING task-domain count at
        round(alpha_goal * N) by picking k of the RESETTING envs, so the realized
        fraction is variance-free where per-reset Bernoulli would drift.

        Overwrites `_object_goal_quat` in place; the caller re-derives whatever
        it caches off the goal. alpha_goal=0 => k=0 => exact no-op.
        """
        m = len(env_ids)
        if m == 0 or not self.goal_anneals:
            return
        standing = torch.ones(self.num_envs, dtype=torch.bool, device=self.device)
        standing[env_ids] = False
        n_target = round(self.alpha_goal * self.num_envs)
        k = min(max(n_target - int((self._goal_is_task & standing).sum()), 0), m)
        sel = torch.zeros(m, dtype=torch.bool, device=self.device)
        if k > 0:
            sel[torch.randperm(m, device=self.device)[:k]] = True
            quat = self._sample_task_goal_quat(k)
            assert quat is not None
            self._object_goal_quat[env_ids[sel]] = quat
        self._goal_is_task[env_ids] = sel

    def _log_goal_domain_split(self, env_ids: torch.Tensor, metric: str) -> None:
        """Log `metric`'s mean over the resetting envs, split by goal domain.

        Call BEFORE `CommandTerm.reset` — it zeroes metrics and then reassigns
        the domain flag. Empty side -> key omitted, never a fake 0. Split a
        metric the TASK actually scores: `at_goal` is yaw-inclusive, so for an
        up-face goal it sits near its floor and says nothing.
        """
        log = self._env.extras.setdefault("log", {})
        is_task = self._goal_is_task[env_ids]
        for name, m in (("task", is_task), ("demo", ~is_task)):
            if bool(m.any()):
                log[f"GoalAnnealing/{metric}_{name}"] = float(
                    self.metrics[metric][env_ids][m].mean()
                )

    def _clip_allowance(self, env_ids: torch.Tensor) -> torch.Tensor | None:
        """Omni mode: each env samples only clips of ITS object."""
        if self._clip_allowed is None:
            return None
        return self._clip_allowed[self._env_object_ids[env_ids]].float()

    def _reset_task(
        self,
        env_ids: torch.Tensor,
        clip_ids: torch.Tensor,
        t: torch.Tensor,
        origins: torch.Tensor,
    ) -> None:
        """Object goal + object RSI. Runs after the robot's RSI, so the RNG
        draw order (robot pose/twist/joint, then object) is unchanged."""
        # Goal = object pose at actual clip end
        clip_end_frames = self.motion.clip_ends[clip_ids] - 1
        self._object_goal_pos[env_ids] = self.motion.obj_pos[clip_end_frames]
        self._object_goal_quat[env_ids] = self.motion.obj_quat[clip_end_frames]

        # RSI: object, randomization conditional on init phase (fcrl parity).
        #   clip start           -> pose rand (placement uncertainty; frame 0
        #                           is non-contact by demo construction).
        #   mid-clip, ref-contact-> twist rand (robot has control-authority,
        #                           recoverable; pose rand would break the
        #                           holding invariant).
        # Ref contact flag, not sensors (stale at reset). Global-concat
        # timeline: clip start == clip_offsets, not 0.
        obj_pos = self.motion.obj_pos[t] + origins
        obj_quat = self.motion.obj_quat[t].clone()
        obj_lin_vel = self.motion.obj_lin_vel[t].clone()
        obj_ang_vel = self.motion.obj_ang_vel[t].clone()

        at_clip_start = t == self.motion.clip_offsets[clip_ids]
        if self.cfg.object_init_pose_range:
            m = at_clip_start
            if m.any():
                s = sample_se3(
                    self.cfg.object_init_pose_range, int(m.sum()), self.device)
                obj_pos[m] = obj_pos[m] + s[:, 0:3]
                obj_quat[m] = quat_mul(
                    quat_from_euler_xyz(s[:, 3], s[:, 4], s[:, 5]), obj_quat[m])
        if self.cfg.object_in_contact_velocity_range:
            m = ~at_clip_start & (self.motion.obj_contact_any[t] > 0.5)
            if m.any():
                s = sample_se3(
                    self.cfg.object_in_contact_velocity_range, int(m.sum()),
                    self.device)
                obj_lin_vel[m] = obj_lin_vel[m] + s[:, 0:3]
                obj_ang_vel[m] = obj_ang_vel[m] + s[:, 3:6]

        obj_state = torch.cat(
            [obj_pos, obj_quat, obj_lin_vel, obj_ang_vel], dim=-1)
        self.object.write_root_state_to_sim(obj_state, env_ids=env_ids)

        # LAST: the goal-domain draw sits after the object RSI draws, so turning
        # annealing on cannot shift the RNG stream the rest of this reset uses.
        self._assign_goal_domain(env_ids)

    def _update_task(self) -> None:
        """Goal-domain alpha + the RobotObjectContactGraph disparity log
        (|ref - live| per body, averaged over envs)."""
        log = self._env.extras.setdefault("log", {})
        self.alpha_goal = anneal_alpha(
            getattr(self._env, "policy_update_count", 0),
            self.cfg.goal_anneal_start, self.cfg.goal_anneal_end,
            self.cfg.alpha_goal_init, 1.0,
        )
        self.metrics["alpha_goal"][:] = self.alpha_goal
        log["GoalAnnealing/alpha"] = self.alpha_goal
        log["GoalAnnealing/frac_task"] = (
            int(self._goal_is_task.sum()) / self.num_envs)

        if not (self.cfg.contact_graph_body_names
                and self.cfg.contact_graph_sensor_name):
            return
        sensor = self._env.scene.sensors[self.cfg.contact_graph_sensor_name]
        force = torch.norm(sensor.data.force, dim=-1)  # (N, K) sensor order
        cols = [sensor.primary_names.index(b)
                for b in self.cfg.contact_graph_body_names]
        live = (force[:, cols] > self.cfg.contact_force_threshold).float()
        disparity = (self.object_bodywise_contact - live).abs().mean(dim=0)
        for k, body in enumerate(self.cfg.contact_graph_body_names):
            log[f"RobotObjectContactGraph/{body}"] = disparity[k].item()
        log["RobotObjectContactGraph/mean"] = disparity.mean().item()

    def _update_metrics(self) -> None:
        super()._update_metrics()
        # Object tracking metrics
        self.metrics["error_object_pos"] = torch.norm(
            self.object_pos_w - self.object.data.root_link_pos_w, dim=-1
        )
        self.metrics["error_object_ori"] = quat_error_magnitude(
            self.object_quat_w, self.object.data.root_link_quat_w
        )
        # Object goal metrics
        goal_pos_w = self._object_goal_pos + self._env.scene.env_origins
        self.metrics["error_object_pos_goal"] = torch.norm(
            self.object.data.root_link_pos_w - goal_pos_w, dim=-1
        )
        self.metrics["error_object_ori_goal"] = quat_error_magnitude(
            self.object.data.root_link_quat_w, self._object_goal_quat
        )
        # Success = pos AND ori at goal (fcrl parity; tables move).
        self.metrics["at_goal"] = (
            (self.metrics["error_object_pos_goal"] < self.cfg.success_pos_threshold)
            & (self.metrics["error_object_ori_goal"] < self.cfg.success_ori_threshold)
        ).float()

    # ── reference accessors (for tracking rewards) ──

    @property
    def object_pos_w(self) -> torch.Tensor:
        """Reference object position, world frame. (B, 3)."""
        return self.motion.obj_pos[self.time_steps] + self._env.scene.env_origins

    @property
    def object_quat_w(self) -> torch.Tensor:
        """Reference object quaternion. (B, 4)."""
        return self.motion.obj_quat[self.time_steps]

    @property
    def object_lin_vel_w(self) -> torch.Tensor:
        """Reference object linear velocity. (B, 3)."""
        return self.motion.obj_lin_vel[self.time_steps]

    @property
    def object_ang_vel_w(self) -> torch.Tensor:
        """Reference object angular velocity. (B, 3)."""
        return self.motion.obj_ang_vel[self.time_steps]

    @property
    def object_bodywise_contact(self) -> torch.Tensor:
        """(B, K) reference per-body robot<->object contact at the current frame —
        the contact-graph node vector, ordered by cfg.contact_graph_body_names."""
        assert self.motion.obj_bodywise_contact is not None, (
            "object_bodywise_contact requires cfg.contact_graph_body_names")
        return self.motion.obj_bodywise_contact[self.time_steps]

    @property
    def object_goal_pos_env(self) -> torch.Tensor:
        """Object goal position in env frame (no world offset). (B, 3)."""
        return self._object_goal_pos

    @property
    def object_goal_quat(self) -> torch.Tensor:
        """Object goal quaternion. (B, 4)."""
        return self._object_goal_quat

    # ── N-step future object reference (robot ones are inherited) ──

    @property
    def motion_object_pos_w_future(self) -> torch.Tensor:
        """Future N-step reference object position, world frame. (B, N, 3)."""
        idx = self._future_time_indices()
        return self.motion.obj_pos[idx] + self._env.scene.env_origins[:, None, :]

    @property
    def motion_object_quat_w_future(self) -> torch.Tensor:
        """Future N-step reference object quaternion. (B, N, 4)."""
        return self.motion.obj_quat[self._future_time_indices()]

    # ── debug viz: ghost robot + object ──

    def _debug_vis_smpl(self, visualizer, batch: int) -> None:
        """SMPL human ghost — the reference stick figure, z-up world."""
        t = self.time_steps[batch]
        draw_smpl_ghost(
            visualizer,
            self.motion.smpl_joints_viz[t].cpu().numpy()
            + self._env.scene.env_origins[batch].cpu().numpy(),
            matrix_from_quat(self.motion.smpl_root_quat[t]).cpu().numpy(),
            label=f"smpl_{batch}",
        )

    def _debug_vis_impl(self, visualizer) -> None:
        import copy

        env_indices = visualizer.get_env_indices(self.num_envs)
        if not env_indices:
            return

        # SMPL command space: human stick-figure + object ref/goal (the robot
        # ghost would render the placeholder motion.npz — misleading, skip it).
        if self.cfg.command_space == "smpl":
            for batch in env_indices:
                self._debug_vis_smpl(visualizer, batch)
                pos = self.object_pos_w[batch].cpu().numpy()
                rot = matrix_from_quat(self.object_quat_w[batch]).cpu().numpy()
                visualizer.add_frame(
                    position=pos, rotation_matrix=rot,
                    scale=_VIZ_FRAME_SCALE, label=f"ref_obj_frame_{batch}",
                    axis_radius=0.005,
                )
                self._debug_vis_goal(visualizer, batch)
            return

        if self._ghost_model is None:
            self._ghost_model = copy.deepcopy(self._env.sim.mj_model)
            for gi in range(self._ghost_model.ngeom):
                if (
                    self._ghost_model.geom_contype[gi] != 0
                    or self._ghost_model.geom_conaffinity[gi] != 0
                ):
                    self._ghost_model.geom_rgba[gi, 3] = 0
                else:
                    self._ghost_model.geom_rgba[gi] = self._ghost_color
            # Omni: snapshot the warp model's PER-WORLD mesh tables so each
            # env's ghost renders ITS object variant (not variant-0's mesh).
            self._ghost_tables = None
            if self._env_object_ids is not None:
                did = self._env.sim.model.geom_dataid.cpu().numpy()
                if did.ndim == 2:
                    self._ghost_tables = (
                        did,
                        self._env.sim.model.geom_matid.cpu().numpy(),
                        self._ghost_model.geom_rgba.copy(),
                    )

        robot = self._env.scene[self.cfg.entity_name]
        r_fj = robot.indexing.free_joint_q_adr.cpu().numpy()
        r_jq = robot.indexing.joint_q_adr.cpu().numpy()
        o_fj = self.object.indexing.free_joint_q_adr.cpu().numpy()

        for batch in env_indices:
            if getattr(self, "_ghost_tables", None) is not None:
                did, mid, base_rgba = self._ghost_tables
                gm = self._ghost_model
                gm.geom_dataid[:] = did[batch]
                gm.geom_matid[:] = mid[batch]
                gm.geom_rgba[:] = base_rgba
                # unfilled variant slots carry dataid -1: hide + point at a
                # valid mesh so the renderer never dereferences -1
                unfilled = (gm.geom_type == mujoco.mjtGeom.mjGEOM_MESH) & (
                    gm.geom_dataid < 0)
                gm.geom_rgba[unfilled, 3] = 0.0
                gm.geom_dataid[unfilled] = 0
            qpos = np.zeros(self._env.sim.mj_model.nq)
            qpos[r_fj[0:3]] = self.body_pos_w[batch, 0].cpu().numpy()
            qpos[r_fj[3:7]] = self.body_quat_w[batch, 0].cpu().numpy()
            qpos[r_jq] = self.joint_pos[batch].cpu().numpy()
            qpos[o_fj[0:3]] = self.object_pos_w[batch].cpu().numpy()
            qpos[o_fj[3:7]] = self.object_quat_w[batch].cpu().numpy()
            visualizer.add_ghost_mesh(
                qpos, model=self._ghost_model, label=f"ghost_{batch}",
            )

            # Orientation frame on reference object
            pos = self.object_pos_w[batch].cpu().numpy()
            rot = matrix_from_quat(self.object_quat_w[batch]).cpu().numpy()
            visualizer.add_frame(
                position=pos, rotation_matrix=rot,
                scale=_VIZ_FRAME_SCALE, label=f"ref_obj_frame_{batch}",
                axis_radius=0.005,
            )

            self._debug_vis_goal(visualizer, batch)

    def _debug_vis_goal(self, visualizer, batch: int) -> None:
        """Goal display — generic: frame + success halo at the goal POSE
        (pos+ori) in the world. Task children may override (e.g. repose's
        floating colored-cube orientation marker)."""
        origin = self._env.scene.env_origins[batch].cpu().numpy()
        goal_center = origin + self._object_goal_pos[batch].cpu().numpy()
        goal_rot = matrix_from_quat(self._object_goal_quat[batch]).cpu().numpy()
        visualizer.add_frame(
            position=goal_center, rotation_matrix=goal_rot,
            scale=_VIZ_FRAME_SCALE, label=f"goal_frame_{batch}",
            axis_radius=0.003,
        )
        at_goal = self.metrics["at_goal"][batch] > 0.5
        visualizer.add_sphere(
            center=goal_center,
            radius=0.25 if at_goal else 0.0,
            color=(0.15, 0.9, 0.25, 0.2) if at_goal else (0.0, 0.0, 0.0, 0.0),
            label=f"goal_success_{batch}",
        )


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

@dataclass(kw_only=True)
class ObjectMotionCommandCfg(MultiClipMotionCommandCfg):
    """MultiClipMotionCommandCfg + object tracking.

    The clip-library fields (`dataset_dir`, `exclude_motions`, `future_steps`,
    the tracked-body/anchor defaults, phase annealing) are inherited from core.
    Everything below mentions an object.
    """

    object_entity_name: str = "object"
    """The scene entity to track and RSI ("object", unified)."""

    # Robot-motion space of the reference: "robot" = retargeted G1 clips drive
    # tracking (rewards, ghost); "smpl" = human SMPL clips drive the SONIC smpl
    # tokenizer (loader expects smpl_motion.npz per sample; motion.npz then
    # only seeds RSI/wrists) — object plumbing identical in both.
    command_space: Literal["robot", "smpl"] = "robot"

    # Omni mode: spawn-order object names (must match the scene's
    # VariantEntityCfg variant order — orcs.assets preserves it).
    # When set, dataset_dir is the multi-dataset root (clips object-keyed via
    # each sample's metadata.json) and every env samples only clips of its
    # assigned object (env->object read from sim.world_to_variant).
    # None -> single-object flat scan (e.g. the repose cube task).
    ordered_object_names: tuple[str, ...] | None = None

    # Conditional object RSI randomization ({x..yaw} -> (lo, hi) dicts):
    #   object_init_pose_range        — applied only at clip-start inits
    #                                   (frame 0 is non-contact by demo
    #                                   construction -> pose rand is safe).
    #   object_in_contact_velocity_range — applied only mid-clip AND when the
    #                                   demo reference has robot<->object
    #                                   contact (control-authority -> a twist
    #                                   kick is recoverable, not a fling).
    # None/{} -> disabled (ideal domain).
    object_init_pose_range: dict[str, tuple[float, float]] | None = None
    object_in_contact_velocity_range: dict[str, tuple[float, float]] | None = None

    # contact-graph reward: robot bodies (contact-legend link names) whose
    # per-body object-contact reference is loaded as `command.object_bodywise_contact`.
    # Single source of column order for ref + live. None -> not loaded.
    contact_graph_body_names: tuple[str, ...] | None = None
    # object-filtered multi-primary ContactSensor read for the
    # RobotObjectContactGraph logging (ref-vs-live disparity). None -> not logged.
    contact_graph_sensor_name: str | None = None
    contact_force_threshold: float = 0.1

    # Goal-domain annealing (orcs.core.schedules; needs `_sample_task_goal_quat`).
    # alpha_goal = the fraction of envs whose goal is TASK-domain rather than the
    # demo's terminal pose, ramping `alpha_goal_init` -> 1 over [start, end].
    # end <= start = OFF, so alpha_goal == alpha_goal_init all run: the default
    # is pure demo goals, and `alpha_goal_init=1.0` is pure task domain (what
    # play/eval wants — it needs no policy-update counter).
    goal_anneal_start: int = 0
    goal_anneal_end: int = 0
    alpha_goal_init: float = 0.0

    # Success thresholds for the `at_goal` metric (pos AND ori, fcrl parity)
    success_pos_threshold: float = 0.15
    """Position error (m) below which the object goal counts as reached."""
    success_ori_threshold: float = 0.35
    """Angular error (rad) below which the object goal counts as reached."""

    def build(self, env: ManagerBasedRlEnv) -> ObjectMotionCommand:
        return ObjectMotionCommand(self, env)
