"""ObjectMotionCommand — the OBJECT half of multi-clip motion tracking.

The clip library, robot RSI, phase annealing, last-frame freeze and the N-step
robot reference accessors are task-blind and live in
:class:`orcs.core.mdp.commands.MultiClipMotionCommand`. What this file adds is
everything that mentions an object:

  1. object tracking on the same timeline (_ConcatMotionLoader's extra channels)
  2. object RSI, conditional on init phase and reference contact
  3. object goal from the final clip frame + the `at_goal` metric
  4. N-step future OBJECT reference accessors
  5. omni mode: env->object identity, so an env samples only ITS object's clips
  6. the ghost/goal/SMPL debug viz

Functionally 1:1 with fcrl's ObjectMotionCommand, built on mjlab.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Literal

import mujoco
import numpy as np
import torch
from mjlab.utils.lab_api.math import (
    matrix_from_quat,
    quat_error_magnitude,
    quat_from_euler_xyz,
    quat_mul,
    sample_uniform,
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
from orcs.tasks.uolm.mdp.contact_schedule import ContactSchedule
from orcs.tasks.uolm.mdp.demo_loader import get_motion_files_for_objects

if TYPE_CHECKING:
    from mjlab.envs import ManagerBasedRlEnv

__all__ = [
    "ObjectMotionCommandCfg",
    "ObjectMotionCommand",
    "ProgressivePoolMotionCommandCfg",
    "ProgressivePoolMotionCommand",
    "motion_dirs",
]

# Per-frame SMPL keypoint *direction* channel (palm/foot/pelvis/head). Ported
# from geometry-aware-policy's src.rewards.reference: staging writes a
# `smpl_dirs.npz` per sample with one (T, 3) unit-vector array per keypoint,
# and the loader concatenates them onto the same timeline as smpl_joints_viz so
# `time_steps` indexes them directly. Read live by the keypoint direction reward
# via `command.motion.smpl_dirs`. Keys must match src.debug.directions.KEYPOINTS.
_KEYPOINT_DIR_NAMES = (
    "left_palm", "right_palm", "left_foot", "right_foot", "pelvis", "head",
)

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
        # per-keypoint reference-direction lists (smpl_dirs.npz), same timeline.
        self._all_dirs: dict[str, list[torch.Tensor]] = {
            n: [] for n in _KEYPOINT_DIR_NAMES}
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

        # SMPL keypoint reference directions (staged smpl_dirs.npz). Absent is
        # legal (robot-space clips / not staged) -> zeros, so the direction
        # reward reads a safe null vector for those frames.
        df = sample_dir / "smpl_dirs.npz"
        dd = np.load(df) if df.exists() else None
        for name in _KEYPOINT_DIR_NAMES:
            if dd is not None and name in dd.files:
                self._all_dirs[name].append(
                    torch.tensor(dd[name], dtype=torch.float32, device=self.device))
            else:
                self._all_dirs[name].append(
                    torch.zeros(T, 3, dtype=torch.float32, device=self.device))

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

        # ── SMPL keypoint reference directions ── {name: (T_tot, 3)}
        self.smpl_dirs: dict[str, torch.Tensor] = {
            name: torch.cat(v) for name, v in self._all_dirs.items()
        }


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

        # Object tracking metrics
        self.metrics["error_object_pos"] = torch.zeros(self.num_envs, device=self.device)
        self.metrics["error_object_ori"] = torch.zeros(self.num_envs, device=self.device)
        # Object goal metrics
        self.metrics["error_object_pos_goal"] = torch.zeros(self.num_envs, device=self.device)
        self.metrics["error_object_ori_goal"] = torch.zeros(self.num_envs, device=self.device)
        self.metrics["at_goal"] = torch.zeros(self.num_envs, device=self.device)

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

    def _update_task(self) -> None:
        """RobotObjectContactGraph: per-body demo-vs-actual contact disparity
        |ref - live|, averaged over envs (logged whenever a graph sensor is
        configured, whatever contact reward, if any, is live)."""
        if not (self.cfg.contact_graph_body_names
                and self.cfg.contact_graph_sensor_name):
            return
        log = self._env.extras.setdefault("log", {})
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
    the tracked-body/anchor defaults, phase annealing, `start_from_zero`) are
    inherited from core. Everything below mentions an object.
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

    # Success thresholds for the `at_goal` metric (pos AND ori, fcrl parity)
    success_pos_threshold: float = 0.15
    """Position error (m) below which the object goal counts as reached."""
    success_ori_threshold: float = 0.35
    """Angular error (rad) below which the object goal counts as reached."""

    def build(self, env: ManagerBasedRlEnv) -> ObjectMotionCommand:
        return ObjectMotionCommand(self, env)


# ---------------------------------------------------------------------------
# ProgressivePoolMotionCommand — Sugar-DRCL growing-pool RSI
# ---------------------------------------------------------------------------
#
# Ported from geometry-aware-policy's src.commands.progressive_pool. On top of
# the real retargeted robot reference (here that IS the base loader's robot
# arrays — orcs already RSIs from the real G1 clips, so no separate retarget
# loader is needed), this grows a **pool of validated start states** over
# training:
#
#   * The pool starts as a clone of the real reference, with only each clip's
#     frame 0 marked valid (`pool_flag`).
#   * Every `update_interval` control steps (after `pool_warmup_steps`) the live
#     state each env was in `validation_k` steps earlier is committed into the
#     pool — but only if that env did NOT reset in the intervening window
#     (survival gate). This validates new (clip, frame) cells.
#   * A protected fraction of envs (`start_init_env_ratio`) always resets to
#     clip frame 0; the rest ("free" envs) draw from the pool with probability
#     `1 - ref_prob`, where `ref_prob` decays linearly from 1 over
#     `[pool_warmup_steps, pool_minref_steps]` down to `pool_minref_ratio`. A
#     cell not yet valid in the pool falls back to the raw reference.
#
# Also implements Sugar's clip-start-only reset noise (`reset_pose_range` /
# `reset_joint_position_range`), applied only when a reset lands exactly on a
# clip's first frame — a separate mechanism from pool growth (the parent's
# pose_range/velocity_range/joint_position_range are left untouched and apply,
# if set, to every resample unconditionally).
#
# Object state/RSI (goal, contact-conditional randomization) is entirely
# untouched — still the parent's logic. Only the robot root+joint state written
# at reset is replaced. Pooling activates only when the clip library is NOT
# object-masked (`_clip_allowed is None`, i.e. the flat/SMPL command space);
# omni/robot datasets fall back to the parent sampler.


class ProgressivePoolMotionCommand(ObjectMotionCommand):
    """ObjectMotionCommand + Sugar-DRCL growing-pool RSI."""

    cfg: "ProgressivePoolMotionCommandCfg"

    def __init__(self, cfg: "ProgressivePoolMotionCommandCfg",
                 env: "ManagerBasedRlEnv") -> None:
        super().__init__(cfg, env)

        # Optional one-shot clip override for deterministic playback. When set,
        # the next reset starts each env from that clip's frame 0, then clears.
        self._forced_clip_ids: torch.Tensor | None = None

        # Optional floating platform to re-place per reset at the resetting env's
        # active clip's last-frame object x,y (see _place_platform). ``None`` (no
        # name, missing entity, or non-mocap) disables it.
        self._platform = None
        name = cfg.platform_entity_name
        if name is not None and name in env.scene.entities:
            plat = env.scene[name]
            if getattr(plat, "is_mocap", False):
                self._platform = plat

        # The REAL robot reference is the base loader's robot arrays (orcs RSIs
        # from the real retargeted G1 clips). Root arrays are the anchor body
        # (index 0) in env-local frame; env origin is added at write time.
        self.ref_root_pos = self.motion.body_pos_w[:, 0]
        self.ref_root_quat = self.motion.body_quat_w[:, 0]
        self.ref_root_lin_vel = self.motion.body_lin_vel_w[:, 0]
        self.ref_root_ang_vel = self.motion.body_ang_vel_w[:, 0]
        self.ref_joint_pos = self.motion.joint_pos
        self.ref_joint_vel = self.motion.joint_vel
        self._pool_J = self.ref_joint_pos.shape[-1]

        # `_init_from_pool` is set per-resample by `_sample_init_frame`, aligned
        # to that call's env_ids. Init empty so an early dry-run resample is safe.
        self._init_from_pool = torch.zeros(0, dtype=torch.bool, device=self.device)

        if not cfg.pool_enabled:
            return

        self.pool_count = 0
        self.start_init_env_count = int(self.num_envs * cfg.start_init_env_ratio)

        # ── the init pool (env-agnostic, origin-free), seeded from the REAL
        # reference; only frame 0 of each clip is valid to start ──
        T = self.ref_root_pos.shape[0]
        self.pool_root_pos = self.ref_root_pos.clone()
        self.pool_root_quat = self.ref_root_quat.clone()
        self.pool_root_lin_vel = self.ref_root_lin_vel.clone()
        self.pool_root_ang_vel = self.ref_root_ang_vel.clone()
        self.pool_joint_pos = self.ref_joint_pos.clone()
        self.pool_joint_vel = self.ref_joint_vel.clone()
        self.pool_flag = torch.zeros(T, dtype=torch.bool, device=self.device)
        self.pool_flag[self.motion.clip_offsets] = True

        # ── candidate buffers (per env), captured `validation_k` before commit ──
        n = self.num_envs
        J = self._pool_J
        self.cand_valid = torch.zeros(n, dtype=torch.bool, device=self.device)
        self.cand_clip_id = torch.zeros(n, dtype=torch.long, device=self.device)
        self.cand_global_step = torch.zeros(n, dtype=torch.long, device=self.device)
        self.cand_root_pos = torch.zeros(n, 3, device=self.device)
        self.cand_root_quat = torch.zeros(n, 4, device=self.device)
        self.cand_root_lin_vel = torch.zeros(n, 3, device=self.device)
        self.cand_root_ang_vel = torch.zeros(n, 3, device=self.device)
        self.cand_joint_pos = torch.zeros(n, J, device=self.device)
        self.cand_joint_vel = torch.zeros(n, J, device=self.device)

        self.metrics["pool_valid_cells"] = torch.zeros(n, device=self.device)
        self.metrics["pool_ref_prob"] = torch.zeros(n, device=self.device)

    # ── ref-vs-pool schedule ────────────────────────────────────────────────

    def _ref_prob(self) -> float:
        """Probability a free env resets from the raw reference (vs the pool)."""
        c = self.pool_count
        w, m, r = (self.cfg.pool_warmup_steps, self.cfg.pool_minref_steps,
                   self.cfg.pool_minref_ratio)
        if c < w or m <= w:
            return 1.0
        if c < m:
            alpha = (c - w) / (m - w)
            return 1.0 - alpha * (1.0 - r)
        return r

    # ── sampling ─────────────────────────────────────────────────────────────

    def _sample_init_frame(
        self, env_ids: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Per-env (clip, global start-frame) with protected/free + pool draw.

        Also records ``self._init_from_pool`` (aligned to ``env_ids``): which
        envs' robot state should come from the pool rather than the raw
        reference. Object-masked (omni) datasets or a disabled pool fall back to
        the parent sampler (pure phase-annealed/uniform reference selection).
        """
        if self._forced_clip_ids is not None:
            clip_ids = self._forced_clip_ids.to(device=self.device, dtype=torch.long)
            if clip_ids.numel() != len(env_ids):
                raise ValueError(
                    f"forced clip count {clip_ids.numel()} does not match "
                    f"env_ids {len(env_ids)}")
            self._forced_clip_ids = None
            self._init_from_pool = torch.zeros(
                len(env_ids), dtype=torch.bool, device=self.device)
            return clip_ids, self.motion.clip_offsets[clip_ids]

        if not self.cfg.pool_enabled or self._clip_allowed is not None:
            clip_ids, steps = super()._sample_init_frame(env_ids)
            self._init_from_pool = torch.zeros(
                len(env_ids), dtype=torch.bool, device=self.device)
            return clip_ids, steps

        n = len(env_ids)
        dev = self.device
        offsets = self.motion.clip_offsets
        lengths = self.motion.clip_lengths

        clip_ids = torch.zeros(n, dtype=torch.long, device=dev)
        global_steps = torch.zeros(n, dtype=torch.long, device=dev)
        use_pool = torch.zeros(n, dtype=torch.bool, device=dev)

        protected = env_ids < self.start_init_env_count
        free = ~protected

        # protected envs: uniform clip, frame 0, always reference.
        if protected.any():
            k = int(protected.sum())
            c = torch.randint(self.motion.n_clips, (k,), device=dev)
            clip_ids[protected] = c
            global_steps[protected] = offsets[c]

        # free envs: uniform candidate frame + ref-vs-pool decision.
        if free.any():
            k = int(free.sum())
            c = torch.randint(self.motion.n_clips, (k,), device=dev)
            # leave `future_steps` of lookahead so there is a trajectory to track
            max_local = (lengths[c].float() - self.cfg.future_steps).clamp(min=1.0)
            local = (torch.rand(k, device=dev) * max_local).long()
            local = torch.minimum(local, (lengths[c] - 1).clamp(min=0))
            gstep = offsets[c] + local

            ref_prob = self._ref_prob()
            take_ref = torch.rand(k, device=dev) < ref_prob
            pool_valid = self.pool_flag[gstep]
            # use pool only when NOT drawing reference AND the cell is validated.
            up = (~take_ref) & pool_valid

            clip_ids[free] = c
            global_steps[free] = gstep
            use_pool[free] = up

        self._init_from_pool = use_pool
        return clip_ids, global_steps

    def force_next_clip_ids(self, clip_ids: torch.Tensor | list[int]) -> None:
        """Force the next reset to start each env from the given clip's frame 0."""
        self._forced_clip_ids = torch.as_tensor(
            clip_ids, dtype=torch.long, device=self.device)

    def _resample_command(self, env_ids: torch.Tensor) -> None:
        """Reset: invalidate stale candidates, delegate clip/time/goal/object RSI
        to the parent (unchanged), then overwrite the robot state with the real
        reference or pool state."""
        if self.cfg.pool_enabled:
            # these envs are resetting -> any candidate captured for them is
            # unproven (they did not survive the validation window).
            self.cand_valid[env_ids] = False

        # parent samples (via our _sample_init_frame), sets clip/time_steps,
        # writes robot + object RSI. The robot side is re-written below (pool or
        # real ref + clip-start noise); the object side is left untouched.
        super()._resample_command(env_ids)
        self._write_robot_state(env_ids)
        self._place_platform(env_ids)

    def _place_platform(self, env_ids: torch.Tensor) -> None:
        """Re-place the floating platform at each resetting env's active clip's
        last-frame object x,y (env-local + env origin), keeping the configured
        box-center height and an upright orientation. No-op when no mocap
        platform was resolved in ``__init__``."""
        if self._platform is None:
            return
        clip_ids = self._clip_ids[env_ids]
        origins = self._env.scene.env_origins[env_ids]
        clip_last = self.motion.clip_ends[clip_ids] - 1
        n = env_ids.shape[0]
        pos = torch.empty(n, 3, device=self.device)
        pos[:, :2] = self.motion.obj_pos[clip_last][:, :2] + origins[:, :2]
        pos[:, 2] = self.cfg.platform_height + origins[:, 2]
        quat = torch.zeros(n, 4, device=self.device)
        quat[:, 0] = 1.0  # identity (w,x,y,z)
        pose = torch.cat([pos, quat], dim=-1)
        self._platform.write_mocap_pose_to_sim(pose, env_ids=env_ids)

    def _write_robot_state(self, env_ids: torch.Tensor) -> None:
        """Overwrite root+joint state from the real reference/pool, with
        Sugar-style reset noise applied only at clip-start resets."""
        t = self.time_steps[env_ids]
        origins = self._env.scene.env_origins[env_ids]
        clip_ids = self._clip_ids[env_ids]
        at_start = t == self.motion.clip_offsets[clip_ids]

        if self.cfg.pool_enabled:
            use_pool = self._init_from_pool.unsqueeze(-1)
            root_pos = torch.where(use_pool, self.pool_root_pos[t], self.ref_root_pos[t])
            root_quat = torch.where(use_pool, self.pool_root_quat[t], self.ref_root_quat[t])
            root_lin_vel = torch.where(
                use_pool, self.pool_root_lin_vel[t], self.ref_root_lin_vel[t])
            root_ang_vel = torch.where(
                use_pool, self.pool_root_ang_vel[t], self.ref_root_ang_vel[t])
            joint_pos = torch.where(use_pool, self.pool_joint_pos[t], self.ref_joint_pos[t])
            joint_vel = torch.where(use_pool, self.pool_joint_vel[t], self.ref_joint_vel[t])
        else:
            root_pos = self.ref_root_pos[t]
            root_quat = self.ref_root_quat[t]
            root_lin_vel = self.ref_root_lin_vel[t]
            root_ang_vel = self.ref_root_ang_vel[t]
            joint_pos = self.ref_joint_pos[t]
            joint_vel = self.ref_joint_vel[t]

        root_pos = root_pos + origins

        rc = self.cfg
        has_pose_noise = bool(rc.reset_pose_range)
        has_joint_noise = tuple(rc.reset_joint_position_range) != (0.0, 0.0)
        if bool(at_start.any()) and (has_pose_noise or has_joint_noise):
            n = len(env_ids)
            mask = at_start.unsqueeze(-1)
            if has_pose_noise:
                s = sample_se3(rc.reset_pose_range, n, self.device)
                root_pos = root_pos + torch.where(
                    mask, s[:, 0:3], torch.zeros_like(s[:, 0:3]))
                d_quat = quat_from_euler_xyz(s[:, 3], s[:, 4], s[:, 5])
                ident = torch.zeros_like(d_quat)
                ident[:, 0] = 1.0
                d_quat = torch.where(mask, d_quat, ident)
                root_quat = quat_mul(d_quat, root_quat)
            if has_joint_noise:
                jn = sample_uniform(
                    rc.reset_joint_position_range[0], rc.reset_joint_position_range[1],
                    joint_pos.shape, device=self.device)
                joint_pos = joint_pos + torch.where(mask, jn, torch.zeros_like(jn))

        self._write_reference_state_to_sim(
            env_ids, root_pos, root_quat, root_lin_vel, root_ang_vel,
            joint_pos, joint_vel)

    # ── step: advance time, grow the pool ─────────────────────────────────────

    def _update_command(self) -> None:
        super()._update_command()   # advances time_steps, logs, relative poses
        if not self.cfg.pool_enabled:
            return

        self.pool_count += 1
        ui, vk = self.cfg.update_interval, self.cfg.validation_k
        # snapshot live states `vk` steps before each update boundary.
        if ui > vk and self.pool_count % ui == (ui - vk):
            self._snapshot_candidates()
        # commit survivors at the boundary, once past warmup.
        if self.pool_count % ui == 0 and self.pool_count > self.cfg.pool_warmup_steps:
            self._commit_candidates()

        self.metrics["pool_valid_cells"][:] = float(self.pool_flag.sum().item())
        self.metrics["pool_ref_prob"][:] = self._ref_prob()
        log = self._env.extras.setdefault("log", {})
        log["ProgressivePool/valid_cells"] = float(self.pool_flag.sum().item())
        log["ProgressivePool/ref_prob"] = self._ref_prob()

    def _snapshot_candidates(self) -> None:
        """Record every env's current live state as a pool candidate."""
        origins = self._env.scene.env_origins
        rd = self.robot.data
        self.cand_clip_id[:] = self._clip_ids
        self.cand_global_step[:] = self.time_steps
        self.cand_root_pos[:] = rd.root_link_pos_w - origins
        self.cand_root_quat[:] = rd.root_link_quat_w
        self.cand_root_lin_vel[:] = rd.root_link_lin_vel_w
        self.cand_root_ang_vel[:] = rd.root_link_ang_vel_w
        self.cand_joint_pos[:] = rd.joint_pos[:, : self._pool_J]
        self.cand_joint_vel[:] = rd.joint_vel[:, : self._pool_J]
        self.cand_valid[:] = True

    def _commit_candidates(self) -> None:
        """Write survived candidates (mid-clip only) into the pool, validating
        their ``(clip, frame)`` cells."""
        local = self.cand_global_step - self.motion.clip_offsets[self.cand_clip_id]
        ok = self.cand_valid & (local > 0)   # frame 0 is protected/seeded
        if not bool(ok.any()):
            return
        g = self.cand_global_step[ok]
        self.pool_root_pos[g] = self.cand_root_pos[ok]
        self.pool_root_quat[g] = self.cand_root_quat[ok]
        self.pool_root_lin_vel[g] = self.cand_root_lin_vel[ok]
        self.pool_root_ang_vel[g] = self.cand_root_ang_vel[ok]
        self.pool_joint_pos[g] = self.cand_joint_pos[ok]
        self.pool_joint_vel[g] = self.cand_joint_vel[ok]
        self.pool_flag[g] = True
        self.cand_valid[:] = False


@dataclass(kw_only=True)
class ProgressivePoolMotionCommandCfg(ObjectMotionCommandCfg):
    """ObjectMotionCommandCfg + Sugar-DRCL pooling knobs."""

    # progressive pooling (control-step units)
    pool_enabled: bool = True
    start_init_env_ratio: float = 0.25
    pool_warmup_steps: int = 24000
    pool_minref_steps: int = 120000
    pool_minref_ratio: float = 0.33
    validation_k: int = 48
    update_interval: int = 2400

    # Sugar-style reset noise, clip-start only (independent of pool_enabled)
    reset_pose_range: dict[str, tuple[float, float]] = field(default_factory=dict)
    reset_joint_position_range: tuple[float, float] = (0.0, 0.0)

    # Floating-platform per-reset placement. When ``platform_entity_name`` names
    # a (mocap) entity in the scene, every reset re-places that platform's x,y at
    # the resetting env's active clip's last-frame object x,y. Left ``None`` the
    # platform keeps its static build-time pose.
    platform_entity_name: str | None = None
    platform_height: float = 0.0

    def build(self, env: ManagerBasedRlEnv) -> ProgressivePoolMotionCommand:
        return ProgressivePoolMotionCommand(self, env)
