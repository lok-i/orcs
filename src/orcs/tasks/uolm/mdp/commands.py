"""ObjectMotionCommand — multi-clip motion tracking with object state.

Extends mjlab's MotionCommand with:
  1. Concatenated multi-clip loader with object tracking (_ConcatMotionLoader)
  2. Reference state init (RSI) for both robot AND object
  3. N-step future reference accessors (robot + object)
  4. Phase annealing curriculum (init near end → init at start)
  5. Object goal from final clip frame
  6. Last-frame freeze: past the clip end the reference holds still; resets
     come from episode events only (steady-state hold + truncation backup)

Functionally 1:1 with fcrl's ObjectMotionCommand, built on mjlab.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Literal

import mujoco
import numpy as np
import torch
from mjlab.tasks.tracking.mdp.commands import MotionCommand, MotionCommandCfg
from mjlab.utils.lab_api.math import (
    matrix_from_quat,
    quat_error_magnitude,
    quat_from_euler_xyz,
    quat_mul,
    sample_uniform,
)

from orcs.tasks.uolm.mdp.contact_schedule import ContactSchedule
from orcs.tasks.uolm.mdp.demo_loader import (
    get_motion_files_for_objects,
    load_field_or_make_zeros,
)

if TYPE_CHECKING:
    from mjlab.envs import ManagerBasedRlEnv

__all__ = ["ObjectMotionCommandCfg", "ObjectMotionCommand", "motion_dirs"]

# Joint/body order maps — canonical copies live in mocke.mdp.joint_maps.
from mocke.mdp.joint_maps import (  # noqa: E402
    G1_TRACKED_BODIES as _G1_TRACKED_BODIES,
)
from mocke.mdp.joint_maps import (
    G1_TRACKED_BODY_NAMES as _G1_BODY_NAMES,
)
from mocke.mdp.joint_maps import (
    IL2MJ as _IL2MJ,
)

_VIZ_FRAME_SCALE = 0.45  # goal/ref frame axis length (m)

# SMPL kinematic tree (24 joints, standard SMPL order: 0=pelvis .. 22/23=hands)
_SMPL_PARENTS = (
    -1, 0, 0, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 9, 9, 12, 13, 14, 16, 17, 18, 19,
    20, 21,
)
_SMPL_GHOST_COLOR = (0.2, 0.8, 0.9, 0.6)

_IL_BODY_IDS = [idx for _, idx in _G1_TRACKED_BODIES]

_SE3_KEYS = ("x", "y", "z", "roll", "pitch", "yaw")


def _sample_se3(
    ranges: dict[str, tuple[float, float]], n: int, device
) -> torch.Tensor:
    """(n, 6) uniform samples from a {x..yaw} -> (lo, hi) dict; missing keys = 0."""
    lims = torch.tensor(
        [ranges.get(k, (0.0, 0.0)) for k in _SE3_KEYS], device=device)
    return sample_uniform(lims[:, 0], lims[:, 1], (n, 6), device=device)


# ---------------------------------------------------------------------------
# Concatenated multi-clip motion loader
# ---------------------------------------------------------------------------

def motion_dirs(dataset_dir: str | list[str]) -> list[Path]:
    """Motion folders (each holds sampleX/motion.npz), found DEPTH-INVARIANTLY.

    `dataset_dir` is one root (str/Path) or a list of paths; each path may be a
    motion folder itself OR any ancestor of them — the omni root whose subdirs
    are motions, a root grouping objects-then-motions, or an explicit list of
    motion folders (the custom layout). We locate every `sampleX/motion.npz`
    beneath each root and take its grandparent as the motion folder, so the two
    layouts differ only in nesting depth and both just work. Sorted by path;
    deduped, first occurrence wins (preserves list order across roots).
    """
    roots = ([Path(dataset_dir)] if isinstance(dataset_dir, (str, Path))
             else [Path(d) for d in dataset_dir])
    seen: dict[Path, None] = {}
    for root in roots:
        for mf in sorted(root.rglob("motion.npz")):
            seen.setdefault(mf.parent.parent, None)  # <motion>/<sampleX>/motion.npz
    return list(seen)


def _scan_flat_dataset(
    dataset_dir: str | list[str],
    exclude_motions: tuple[str, ...] | None = None,
) -> list[str]:
    """<motion>/<sampleX>/motion.npz walk — the single-object layout.

    `dataset_dir` is a root (subdirs = motion folders) or a list of motion
    folders. `exclude_motions` entries match a whole motion folder ("<motion>",
    all samples) or a single clip ("<motion>/<sampleX>").
    """
    excl = set(exclude_motions or ())
    n_skipped = 0
    motion_files: list[str] = []
    for motion_dir in motion_dirs(dataset_dir):
        sample_dirs = sorted(
            (d for d in motion_dir.iterdir()
             if d.is_dir() and d.name.startswith("sample")),
            key=lambda d: int("".join(filter(str.isdigit, d.name)) or "0"),
        )
        for sample_dir in sample_dirs:
            mf = sample_dir / "motion.npz"
            if not mf.exists():
                continue
            if (motion_dir.name in excl
                    or f"{motion_dir.name}/{sample_dir.name}" in excl):
                n_skipped += 1
                continue
            motion_files.append(str(mf))
    if n_skipped:
        print(f"[uolm] excluded {n_skipped} clips "
              f"({len(excl)} exclude_motions entries)")
    if not motion_files:
        raise FileNotFoundError(f"No motion.npz under {dataset_dir}")
    return motion_files


class _ConcatMotionLoader:
    """All demo clips concatenated into one timeline.

    MotionLoader-compatible interface (joint_pos, body_pos_w, etc.) so
    MotionCommand properties (ghost viz, body tracking) work unchanged.
    Adds object tracking and per-clip boundaries for frame clamping.

    `motion_files=None` scans `dataset_dir` flat (single-object layout);
    an explicit list (omni: object-grouped order from
    `get_motion_files_for_objects`) is loaded verbatim — clip index i
    corresponds to motion_files[i].
    """

    def __init__(
        self,
        dataset_dir: str | list[str],
        device: str | torch.device,
        contact_graph_body_names: tuple[str, ...] | None = None,
        motion_files: list[str] | None = None,
    ) -> None:
        root = dataset_dir  # display only (for the max_len message below)
        if motion_files is None:
            motion_files = _scan_flat_dataset(dataset_dir)

        all_jp: list[torch.Tensor] = []
        all_jv: list[torch.Tensor] = []
        all_bp: list[torch.Tensor] = []
        all_bq: list[torch.Tensor] = []
        all_blv: list[torch.Tensor] = []
        all_bav: list[torch.Tensor] = []
        all_sj: list[torch.Tensor] = []
        all_sq: list[torch.Tensor] = []
        all_sv: list[torch.Tensor] = []
        all_op: list[torch.Tensor] = []
        all_oq: list[torch.Tensor] = []
        all_olv: list[torch.Tensor] = []
        all_oav: list[torch.Tensor] = []
        clip_lengths: list[int] = []

        for mf_str in motion_files:
            mf = Path(mf_str)
            sample_dir = mf.parent
            d = np.load(mf)
            T = d["joint_pos"].shape[0]

            def _t(a: np.ndarray) -> torch.Tensor:
                return torch.tensor(a, dtype=torch.float32, device=device)

            all_jp.append(_t(d["joint_pos"])[:, _IL2MJ])
            all_jv.append(_t(d["joint_vel"])[:, _IL2MJ])
            all_bp.append(_t(d["body_pos_w"][:, _IL_BODY_IDS]))
            all_bq.append(_t(d["body_quat_w"][:, _IL_BODY_IDS]))
            all_blv.append(_t(d["body_lin_vel_w"][:, _IL_BODY_IDS]))
            all_bav.append(_t(d["body_ang_vel_w"][:, _IL_BODY_IDS]))

            # SMPL human reference (smpl mode; zeros when absent).
            # Contract: smpl_joints RAW (y-up, root-centered — encoder-exact,
            # SONIC never converts them); smpl_root_quat_w z-up/wxyz/base-rot
            # removed; smpl_joints_viz_w optional z-up world (ghost only).
            sf = sample_dir / "smpl_motion.npz"
            sd = np.load(sf) if sf.exists() else None
            sj, _ = load_field_or_make_zeros(
                sd, "smpl_joints", (T, 24, 3), str(device))
            sq, exist = load_field_or_make_zeros(
                sd, "smpl_root_quat_w", (T, 4), str(device))
            if not exist:
                sq[:, 0] = 1.0
            # ghost-only z-up world track; falls back to the raw joints
            sv, exist = load_field_or_make_zeros(
                sd, "smpl_joints_viz_w", (T, 24, 3), str(device))
            all_sj.append(sj)
            all_sq.append(sq)
            all_sv.append(sv if exist else sj)

            of = sample_dir / "object_motion.npz"
            if of.exists():
                od = np.load(of)
            else:
                od = None
            obp, _ = load_field_or_make_zeros(od, "obj_pos_w", (T, 3), str(device))
            obq, exist = load_field_or_make_zeros(od, "obj_quat_w", (T, 4), str(device))
            if not exist:
                obq[:, 0] = 1.0
            oblv, _ = load_field_or_make_zeros(od, "obj_lin_vel_w", (T, 3), str(device))
            obav, _ = load_field_or_make_zeros(od, "obj_ang_vel_w", (T, 3), str(device))

            all_op.append(obp)
            all_oq.append(obq)
            all_olv.append(oblv)
            all_oav.append(obav)

            clip_lengths.append(T)

        # contact schedule: the single source of contact truth. richer (T,N,N)
        # narrowphase matrix, name-queryable; subsumes the deprecated scalar
        # `object_motion.npz["contact"]` flag (== object_robot_contact_any).
        self.contact = ContactSchedule(motion_files, clip_lengths, device)

        # per-body contact-graph node vector, concatenated onto the same
        # timeline as obj_pos (only if a reward asks for it).
        self.obj_bodywise_contact: torch.Tensor | None = (
            torch.cat([
                self.contact.body_object_contacts(i, contact_graph_body_names)
                for i in range(len(motion_files))
            ])  # (T_tot, K)
            if contact_graph_body_names else None
        )

        # (T_tot,) ANY-robot-body<->object contact — gates the conditional
        # object RSI (twist rand only where the robot has control-authority).
        self.obj_contact_any = torch.cat([
            self.contact.object_robot_contact_any(i)
            for i in range(len(motion_files))
        ])

        # ── MotionLoader-compatible tensors ──
        self.joint_pos = torch.cat(all_jp)           # (T_tot, 29)
        self.joint_vel = torch.cat(all_jv)           # (T_tot, 29)
        self.body_pos_w = torch.cat(all_bp)          # (T_tot, 14, 3)
        self.body_quat_w = torch.cat(all_bq)         # (T_tot, 14, 4)
        self.body_lin_vel_w = torch.cat(all_blv)     # (T_tot, 14, 3)
        self.body_ang_vel_w = torch.cat(all_bav)     # (T_tot, 14, 3)
        self.time_step_total: int = self.joint_pos.shape[0]

        # ── SMPL human reference (smpl command space) ──
        self.smpl_joints = torch.cat(all_sj)         # (T_tot, 24, 3) RAW (encoder)
        self.smpl_root_quat = torch.cat(all_sq)      # (T_tot, 4) z-up, wxyz
        self.smpl_joints_viz = torch.cat(all_sv)     # (T_tot, 24, 3) z-up world (ghost)

        # ── object tracking ──
        self.obj_pos = torch.cat(all_op)             # (T_tot, 3)
        self.obj_quat = torch.cat(all_oq)            # (T_tot, 4)
        self.obj_lin_vel = torch.cat(all_olv)        # (T_tot, 3)
        self.obj_ang_vel = torch.cat(all_oav)        # (T_tot, 3)

        # ── clip boundaries (for frame clamping) ──
        self.clip_lengths = torch.tensor(
            clip_lengths, device=device, dtype=torch.long
        )
        self.clip_offsets = torch.zeros(
            len(clip_lengths), device=device, dtype=torch.long
        )
        if len(clip_lengths) > 1:
            self.clip_offsets[1:] = self.clip_lengths[:-1].cumsum(0)
        self.clip_ends = self.clip_offsets + self.clip_lengths  # exclusive end
        self.n_clips: int = len(clip_lengths)
        self.max_clip_length: int = int(self.clip_lengths.max().item())

        from orcs.tasks.uolm.mdp.demo_loader import last_scan

        n_excl = len(last_scan.get("excluded", ()))
        print(
            f"[uolm] {self.n_clips} clips, {self.time_step_total} frames, "
            f"max_len={self.max_clip_length}"
            + (f", {n_excl} motions excluded" if n_excl else "")
            + f" from {root}"
        )


# ---------------------------------------------------------------------------
# ObjectMotionCommand
# ---------------------------------------------------------------------------

class ObjectMotionCommand(MotionCommand):
    """MotionCommand + object tracking, multi-clip loader, RSI, phase annealing."""

    cfg: ObjectMotionCommandCfg

    def __init__(self, cfg: ObjectMotionCommandCfg, env: ManagerBasedRlEnv):
        super().__init__(cfg, env)

        # Replace single-file MotionLoader with concatenated multi-clip loader.
        # Omni mode (ordered_object_names set): dataset_dir is the multi-dataset
        # root, clips are object-keyed (each sample's metadata.json) and loaded
        # grouped in object order, so clip->object is a repeat_interleave.
        if cfg.ordered_object_names:
            files_by_obj, files_ordered = get_motion_files_for_objects(
                list(cfg.ordered_object_names), cfg.dataset_dir,
                exclude_motions=list(cfg.exclude_motions or ()),
            )
            counts = [len(files_by_obj[n]) for n in cfg.ordered_object_names]
            clip_object_ids = torch.repeat_interleave(
                torch.arange(len(counts), device=self.device),
                torch.tensor(counts, device=self.device),
            )
        else:
            # flat scan honors exclude_motions (folder- or sample-level)
            files_ordered = (
                _scan_flat_dataset(cfg.dataset_dir, cfg.exclude_motions)
                if cfg.exclude_motions else None
            )
            clip_object_ids = None
        self.motion = _ConcatMotionLoader(
            cfg.dataset_dir, self.device,
            contact_graph_body_names=cfg.contact_graph_body_names,
            motion_files=files_ordered,
        )

        # Object entity in the scene
        self.object = env.scene[cfg.object_entity_name]

        # Omni mode: env->object identity from the sim's per-world variant
        # table (single source of truth — VariantEntityCfg assignment, fixed
        # at sim init). Variant order == ordered_object_names order by
        # construction (orcs.assets), so the table IS the
        # object-id map. _clip_allowed masks clip sampling per env.
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
                clip_object_ids[None, :]
                == torch.arange(len(cfg.ordered_object_names),
                                device=self.device)[:, None]
            )  # (n_objects, n_clips)
        else:
            self._env_object_ids = None
            self._clip_allowed = None

        # Per-env clip tracking
        self._clip_ids = torch.zeros(
            self.num_envs, dtype=torch.long, device=self.device
        )
        # Steps spent frozen at the clip's last frame (feeds
        # `exceeded_motion_by_eps`; the episode — not the motion — owns resets).
        self._steps_past_end = torch.zeros(
            self.num_envs, dtype=torch.long, device=self.device
        )

        # Object goal: final-frame orientation per env (set at reset)
        self._object_goal_pos = torch.zeros(
            self.num_envs, 3, device=self.device
        )
        self._object_goal_quat = torch.zeros(
            self.num_envs, 4, device=self.device
        )
        self._object_goal_quat[:, 0] = 1.0

        # Phase annealing: _init_phase_max ∈ [0,1] caps WHERE envs can
        # start (not where they end — clips always run to natural end).
        self._init_phase_max: float = 1.0
        self.metrics["init_phase_max"] = torch.zeros(self.num_envs, device=self.device)

        # Object tracking metrics
        self.metrics["error_object_pos"] = torch.zeros(self.num_envs, device=self.device)
        self.metrics["error_object_ori"] = torch.zeros(self.num_envs, device=self.device)
        # Object goal metrics
        self.metrics["error_object_pos_goal"] = torch.zeros(self.num_envs, device=self.device)
        self.metrics["error_object_ori_goal"] = torch.zeros(self.num_envs, device=self.device)
        self.metrics["at_goal"] = torch.zeros(self.num_envs, device=self.device)

    # ── sampling ──

    def _sample_init_frame(
        self, env_ids: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Sample per-env (clip_id, init_frame) with phase annealing.

        _init_phase_max restricts WHERE the episode can START.
        Clips always run to their natural boundary.
        Omni mode: each env samples only clips of ITS object (_clip_allowed).
        """
        n = len(env_ids)
        L = self.motion.max_clip_length

        allowed = (
            self._clip_allowed[self._env_object_ids[env_ids]].float()
            if self._clip_allowed is not None else None
        )  # (n, n_clips) | None

        # Force frame 0: pick a clip uniformly (within allowance), init at start.
        if self.cfg.start_from_zero:
            if allowed is None:
                clip_ids = torch.randint(
                    self.motion.n_clips, (n,), device=self.device
                )
            else:
                clip_ids = torch.multinomial(allowed, 1).squeeze(1)
            return clip_ids, self.motion.clip_offsets[clip_ids]

        # Init-sampling mask: [0, _init_phase_max * clip_length) per clip
        if self._init_phase_max < 1.0:
            init_lens = (
                self.motion.clip_lengths.float() * self._init_phase_max
            ).long().clamp(min=1)
            frame_indices = torch.arange(L, device=self.device).unsqueeze(0)
            init_mask = frame_indices < init_lens.unsqueeze(1)
        else:
            frame_indices = torch.arange(L, device=self.device).unsqueeze(0)
            init_mask = frame_indices < self.motion.clip_lengths.unsqueeze(1)

        # Pick clip weighted by init length (within allowance), then uniform frame
        clip_weights = init_mask.sum(dim=1).float()
        if allowed is None:
            clip_ids = torch.multinomial(clip_weights, n, replacement=True)
        else:
            clip_ids = torch.multinomial(
                clip_weights[None, :] * allowed, 1
            ).squeeze(1)
        clip_lens = init_mask[clip_ids].sum(dim=1).float()
        local_frames = (torch.rand(n, device=self.device) * clip_lens).long()
        local_frames = local_frames.clamp(max=clip_lens.long() - 1)

        global_steps = self.motion.clip_offsets[clip_ids] + local_frames
        return clip_ids, global_steps

    def _resample_command(self, env_ids: torch.Tensor) -> None:
        """Sample new clip + frame, RSI robot + object, set goal."""
        clip_ids, global_steps = self._sample_init_frame(env_ids)
        self._clip_ids[env_ids] = clip_ids
        self.time_steps[env_ids] = global_steps
        self._steps_past_end[env_ids] = 0

        # Goal = object pose at actual clip end
        clip_end_frames = self.motion.clip_ends[clip_ids] - 1
        self._object_goal_pos[env_ids] = self.motion.obj_pos[clip_end_frames]
        self._object_goal_quat[env_ids] = self.motion.obj_quat[clip_end_frames]

        # RSI: write robot state to sim (MotionCommand pattern with randomization)
        origins = self._env.scene.env_origins[env_ids]
        t = self.time_steps[env_ids]

        root_pos = self.motion.body_pos_w[t, 0] + origins
        root_ori = self.motion.body_quat_w[t, 0].clone()
        root_lin_vel = self.motion.body_lin_vel_w[t, 0].clone()
        root_ang_vel = self.motion.body_ang_vel_w[t, 0].clone()
        joint_pos = self.motion.joint_pos[t].clone()
        joint_vel = self.motion.joint_vel[t]

        # Robot pose/twist/joint randomization (same as base MotionCommand),
        # joints clamped to soft limits.
        s = _sample_se3(self.cfg.pose_range, len(env_ids), self.device)
        root_pos = root_pos + s[:, 0:3]
        root_ori = quat_mul(quat_from_euler_xyz(s[:, 3], s[:, 4], s[:, 5]),
                            root_ori)

        s = _sample_se3(self.cfg.velocity_range, len(env_ids), self.device)
        root_lin_vel = root_lin_vel + s[:, 0:3]
        root_ang_vel = root_ang_vel + s[:, 3:6]

        joint_pos = joint_pos + sample_uniform(
            lower=self.cfg.joint_position_range[0],
            upper=self.cfg.joint_position_range[1],
            size=joint_pos.shape,
            device=joint_pos.device,
        )
        limits = self.robot.data.soft_joint_pos_limits[env_ids]
        joint_pos = joint_pos.clamp(limits[..., 0], limits[..., 1])

        self._write_reference_state_to_sim(
            env_ids, root_pos, root_ori, root_lin_vel, root_ang_vel,
            joint_pos, joint_vel,
        )

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
                s = _sample_se3(
                    self.cfg.object_init_pose_range, int(m.sum()), self.device)
                obj_pos[m] = obj_pos[m] + s[:, 0:3]
                obj_quat[m] = quat_mul(
                    quat_from_euler_xyz(s[:, 3], s[:, 4], s[:, 5]), obj_quat[m])
        if self.cfg.object_in_contact_velocity_range:
            m = ~at_clip_start & (self.motion.obj_contact_any[t] > 0.5)
            if m.any():
                s = _sample_se3(
                    self.cfg.object_in_contact_velocity_range, int(m.sum()),
                    self.device)
                obj_lin_vel[m] = obj_lin_vel[m] + s[:, 0:3]
                obj_ang_vel[m] = obj_ang_vel[m] + s[:, 3:6]

        obj_state = torch.cat(
            [obj_pos, obj_quat, obj_lin_vel, obj_ang_vel], dim=-1)
        self.object.write_root_state_to_sim(obj_state, env_ids=env_ids)

    # ── step ──

    def _update_command(self) -> None:
        # Phase annealing: _init_phase_max 1→0 over N policy updates.
        if self.cfg.init_phase_anneal_iterations > 0 and hasattr(
            self._env, "policy_update_count"
        ):
            k = self._env.policy_update_count
            self._init_phase_max = max(
                0.0, 1.0 - k / self.cfg.init_phase_anneal_iterations
            )
        self.metrics["init_phase_max"][:] = self._init_phase_max
        log = self._env.extras.setdefault("log", {})
        log["PhaseAnnealing/init_phase_max"] = self._init_phase_max

        # RobotObjectContactGraph: per-body demo-vs-actual contact disparity
        # |ref - live|, averaged over envs (logged whenever a graph sensor is
        # configured, whatever contact reward, if any, is live).
        if self.cfg.contact_graph_body_names and self.cfg.contact_graph_sensor_name:
            sensor = self._env.scene.sensors[self.cfg.contact_graph_sensor_name]
            force = torch.norm(sensor.data.force, dim=-1)  # (N, K) sensor order
            cols = [sensor.primary_names.index(b)
                    for b in self.cfg.contact_graph_body_names]
            live = (force[:, cols] > self.cfg.contact_force_threshold).float()
            disparity = (self.object_bodywise_contact - live).abs().mean(dim=0)
            for k, body in enumerate(self.cfg.contact_graph_body_names):
                log[f"RobotObjectContactGraph/{body}"] = disparity[k].item()
            log["RobotObjectContactGraph/mean"] = disparity.mean().item()

        # Advance frame, freeze at the clip's last frame (fcrl parity): the
        # episode — not the motion — owns resets. Past the boundary the
        # reference holds still, so the policy must stabilize into a
        # steady-state hold and `exceeded_motion_by_eps` truncates (time_out
        # → V(s') bootstraps) after ε frozen steps. Resampling happens only
        # via env reset (CommandTerm.reset → _resample_command).
        self.time_steps += 1
        clip_last = self.motion.clip_ends[self._clip_ids] - 1
        overrun = self.time_steps > clip_last
        self._steps_past_end += overrun.long()
        self.time_steps = torch.minimum(self.time_steps, clip_last)

        self.update_relative_body_poses()

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

    # ── command (N-step, matches WBC IL convention) ──

    @property
    def command(self) -> torch.Tensor:
        """N-step [joint_pos, joint_vel] — matches the WBC generated_commands."""
        return torch.cat([
            self.motion_joint_pos_future.reshape(self._env.num_envs, -1),
            self.motion_joint_vel_future.reshape(self._env.num_envs, -1),
        ], dim=1)

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

    # ── N-step future reference (for phase-conditioned obs) ──

    def future_frames(self, steps: int, skip: int = 1) -> torch.Tensor:
        """(B, steps) future time indices at ``skip`` spacing, clamped within clip boundaries.

        Shared accessor with ``mocke.mdp.FutureMotionCommand`` — obs terms
        that own their window shape (e.g. the SONIC tokenizer) are duck-typed on it.
        """
        offsets = torch.arange(steps, device=self.device) * skip
        indices = self.time_steps[:, None] + offsets[None, :]  # (B, steps)
        clip_ends = self.motion.clip_ends[self._clip_ids]  # (B,)
        return indices.clamp(max=(clip_ends - 1)[:, None])

    def _future_time_indices(self) -> torch.Tensor:
        """(B, N) future time indices, clamped within clip boundaries."""
        return self.future_frames(self.cfg.future_steps)

    @property
    def motion_anchor_pos_w_future(self) -> torch.Tensor:
        """Future N-step anchor position, world frame. (B, N, 3)."""
        idx = self._future_time_indices()
        ai = self.motion_anchor_body_index
        return self.motion.body_pos_w[idx, ai] + self._env.scene.env_origins[:, None, :]

    @property
    def motion_anchor_quat_w_future(self) -> torch.Tensor:
        """Future N-step anchor quaternion. (B, N, 4)."""
        ai = self.motion_anchor_body_index
        return self.motion.body_quat_w[self._future_time_indices(), ai]

    @property
    def motion_joint_pos_future(self) -> torch.Tensor:
        """Future N-step reference joint positions. (B, N, J)."""
        return self.motion.joint_pos[self._future_time_indices()]

    @property
    def motion_joint_vel_future(self) -> torch.Tensor:
        """Future N-step reference joint velocities. (B, N, J)."""
        return self.motion.joint_vel[self._future_time_indices()]

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
        """SMPL human ghost — 24-joint stick figure from the reference clip
        (no body model / LBS; spheres + parent-child bones, z-up world)."""
        origin = self._env.scene.env_origins[batch].cpu().numpy()
        joints = (
            self.motion.smpl_joints_viz[self.time_steps[batch]].cpu().numpy() + origin
        )  # (24, 3)
        for j, parent in enumerate(_SMPL_PARENTS):
            visualizer.add_sphere(
                center=joints[j], radius=0.03, color=_SMPL_GHOST_COLOR,
                label=f"smpl_{batch}_j{j}",
            )
            if parent >= 0:
                visualizer.add_cylinder(
                    start=joints[parent], end=joints[j], radius=0.012,
                    color=_SMPL_GHOST_COLOR, label=f"smpl_{batch}_b{j}",
                )
        rot = matrix_from_quat(
            self.motion.smpl_root_quat[self.time_steps[batch]]).cpu().numpy()
        visualizer.add_frame(
            position=joints[0], rotation_matrix=rot,
            scale=0.25, label=f"smpl_root_frame_{batch}", axis_radius=0.004,
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
class ObjectMotionCommandCfg(MotionCommandCfg):
    """MotionCommandCfg + multi-clip object tracking.

    entity_name = "robot" for MotionCommand (ghost viz, body tracking).
    object_entity_name: the scene entity to track/RSI ("object", unified).
    """

    entity_name: str = "robot"
    object_entity_name: str = "object"

    # Robot-motion space of the reference: "robot" = retargeted G1 clips drive
    # tracking (rewards, ghost); "smpl" = human SMPL clips drive the SONIC smpl
    # tokenizer (loader expects smpl_motion.npz per sample; motion.npz then
    # only seeds RSI/wrists) — object plumbing identical in both.
    command_space: Literal["robot", "smpl"] = "robot"

    # Multi-clip dataset root, or a list of motion folders (replaces the
    # single motion_file after init). Depth-invariant either way — see
    # `motion_dirs`.
    dataset_dir: str | list[str] = ""

    # Omni mode: spawn-order object names (must match the scene's
    # VariantEntityCfg variant order — orcs.assets preserves it).
    # When set, dataset_dir is the multi-dataset root (clips object-keyed via
    # each sample's metadata.json) and every env samples only clips of its
    # assigned object (env->object read from sim.world_to_variant).
    # None -> single-object flat scan (e.g. the repose cube task).
    ordered_object_names: tuple[str, ...] | None = None
    # Motions to skip. Omni mode: bare name (any dataset) or "<dataset>/<motion>".
    # Flat mode (single-object): "<motion>" (whole folder) or "<motion>/<sampleX>".
    exclude_motions: tuple[str, ...] | None = None

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

    # N-step future reference lookahead
    future_steps: int = 5

    # MotionCommand defaults — 14 tracked bodies
    anchor_body_name: str = "pelvis"
    body_names: tuple[str, ...] = _G1_BODY_NAMES
    # NOTE: base-class adaptive sampling is NOT wired to the multi-clip
    # sampler; pinned to "start" (per-clip adaptive sampling = future PR).
    sampling_mode: Literal["adaptive", "uniform", "start"] = "start"

    # contact-graph reward: robot bodies (contact-legend link names) whose
    # per-body object-contact reference is loaded as `command.object_bodywise_contact`.
    # Single source of column order for ref + live. None -> not loaded.
    contact_graph_body_names: tuple[str, ...] | None = None
    # object-filtered multi-primary ContactSensor read for the
    # RobotObjectContactGraph logging (ref-vs-live disparity). None -> not logged.
    contact_graph_sensor_name: str | None = None
    contact_force_threshold: float = 0.1

    # Phase annealing: init_phase_max 1→0 over N policy updates.
    # Caps WHERE envs can start; clips always run to natural end.
    # 0 = disabled (init anywhere).
    init_phase_anneal_iterations: int = 0

    # Always init at frame 0 (init_phase=0), ignoring the [0, init_phase_max)
    # sampling window. Set True for play so every clip runs from its start.
    start_from_zero: bool = False

    # Success thresholds for the `at_goal` metric (pos AND ori, fcrl parity)
    success_pos_threshold: float = 0.15
    """Position error (m) below which the object goal counts as reached."""
    success_ori_threshold: float = 0.35
    """Angular error (rad) below which the object goal counts as reached."""

    def build(self, env: ManagerBasedRlEnv) -> ObjectMotionCommand:
        return ObjectMotionCommand(self, env)
