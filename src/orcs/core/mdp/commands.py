"""MultiClipMotionCommand — mjlab's MotionCommand over MANY clips.

mjlab's `MotionCommand` tracks one file. Every orcs task tracks a library, and
needs the same five things on top:

  1. one concatenated timeline (`ConcatMotionLoader`) with per-clip boundaries
  2. reference state init (RSI) with pose/twist/joint randomization
  3. N-step future reference accessors (the WBC tokenizer's window)
  4. phase annealing — init near the clip end, walk back toward the start
  5. last-frame freeze: past the clip end the reference holds still and the
     EPISODE owns the reset (steady-state hold + `exceeded_motion_by_eps`
     truncation), never the motion

What a task adds is an identity — which clips this env is allowed to sample,
and what else rides the timeline. Three hooks, no forking:

    _build_loader()                              which loader, which files
    _clip_allowance(env_ids) -> (n, n_clips)|None   per-env clip mask
    _init_task() / _reset_task() / _update_task()   task state on the timeline

**`_clip_allowance` is called at every reset, never cached.** A task whose
env->identity map is fixed for the run (uolm: object variants) may cache
internally; one whose map moves (a terrain curriculum promoting rows) must not,
or promoted envs keep sampling their old identity's clips.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Literal

import numpy as np
import torch
from mjlab.managers import CommandTerm
from mjlab.tasks.tracking.mdp.commands import MotionCommand, MotionCommandCfg
from mjlab.utils.lab_api.math import (
    quat_from_euler_xyz,
    quat_mul,
    sample_uniform,
)
from mocke.mdp.joint_maps import G1_TRACKED_BODY_NAMES as _G1_BODY_NAMES

from orcs.core.data.loader import ConcatMotionLoader
from orcs.core.data.scan import scan_flat

if TYPE_CHECKING:
    from mjlab.envs import ManagerBasedRlEnv

__all__ = ["MultiClipMotionCommand", "MultiClipMotionCommandCfg", "sample_se3"]

_SE3_KEYS = ("x", "y", "z", "roll", "pitch", "yaw")


def sample_se3(
    ranges: dict[str, tuple[float, float]], n: int, device
) -> torch.Tensor:
    """(n, 6) uniform samples from a {x..yaw} -> (lo, hi) dict; missing keys = 0."""
    lims = torch.tensor(
        [ranges.get(k, (0.0, 0.0)) for k in _SE3_KEYS], device=device)
    return sample_uniform(lims[:, 0], lims[:, 1], (n, 6), device=device)


class MultiClipMotionCommand(MotionCommand):
    """MotionCommand over a concatenated clip library, with RSI + annealing."""

    cfg: MultiClipMotionCommandCfg

    def __init__(self, cfg: MultiClipMotionCommandCfg, env: ManagerBasedRlEnv):
        # MotionCommand.__init__ eagerly loads cfg.motion_file, only for ORCS to
        # discard it on the next line. Initialize its state directly so a
        # multi-clip command has no hidden dependency on a dummy single clip.
        # This mirrors mjlab's constructor fields; tracking properties and
        # debug visualization remain inherited from MotionCommand.
        CommandTerm.__init__(self, cfg, env)
        self.robot = env.scene[cfg.entity_name]
        self.robot_anchor_body_index = self.robot.body_names.index(
            cfg.anchor_body_name
        )
        self.motion_anchor_body_index = cfg.body_names.index(cfg.anchor_body_name)
        self.body_indexes = torch.tensor(
            self.robot.find_bodies(cfg.body_names, preserve_order=True)[0],
            dtype=torch.long,
            device=self.device,
        )
        self.motion = self._build_loader()
        self.time_steps = torch.zeros(
            self.num_envs, dtype=torch.long, device=self.device
        )
        self.body_pos_relative_w = torch.zeros(
            self.num_envs, len(cfg.body_names), 3, device=self.device
        )
        self.body_quat_relative_w = torch.zeros(
            self.num_envs, len(cfg.body_names), 4, device=self.device
        )
        self.body_quat_relative_w[:, :, 0] = 1.0

        self.bin_count = int(self.motion.time_step_total // (1 / env.step_dt)) + 1
        self.bin_failed_count = torch.zeros(
            self.bin_count, dtype=torch.float, device=self.device
        )
        self._current_bin_failed = torch.zeros(
            self.bin_count, dtype=torch.float, device=self.device
        )
        self.kernel = torch.tensor(
            [cfg.adaptive_lambda**i for i in range(cfg.adaptive_kernel_size)],
            device=self.device,
        )
        self.kernel = self.kernel / self.kernel.sum()
        for metric in (
            "error_anchor_pos",
            "error_anchor_rot",
            "error_anchor_lin_vel",
            "error_anchor_ang_vel",
            "error_body_pos",
            "error_body_rot",
            "error_joint_pos",
            "error_joint_vel",
            "sampling_entropy",
            "sampling_top1_prob",
            "sampling_top1_bin",
        ):
            self.metrics[metric] = torch.zeros(self.num_envs, device=self.device)
        self._ghost_model = None
        self._ghost_color = np.array(cfg.viz.ghost_color, dtype=np.float32)

        # Per-env clip tracking
        self._clip_ids = torch.zeros(
            self.num_envs, dtype=torch.long, device=self.device
        )
        # Steps spent frozen at the clip's last frame (feeds
        # `exceeded_motion_by_eps`; the episode — not the motion — owns resets).
        self._steps_past_end = torch.zeros(
            self.num_envs, dtype=torch.long, device=self.device
        )

        # Phase annealing: _init_phase_max ∈ [0,1] caps WHERE envs can
        # start (not where they end — clips always run to natural end).
        self._init_phase_max: float = 1.0
        self.metrics["init_phase_max"] = torch.zeros(self.num_envs, device=self.device)

        self._init_task()

    # ── extension hooks ──

    def _build_loader(self) -> ConcatMotionLoader:
        """The clip library. Override to key the file order to an identity."""
        motion_files = (
            scan_flat(self.cfg.dataset_dir, self.cfg.exclude_motions)
            if self.cfg.exclude_motions else None
        )
        return ConcatMotionLoader(
            self.cfg.dataset_dir, self.device, motion_files=motion_files)

    def _clip_allowance(self, env_ids: torch.Tensor) -> torch.Tensor | None:
        """(n, n_clips) float mask of which clips each env may sample, or None
        for "any clip". Re-read every reset — see the module docstring."""
        return None

    def _init_task(self) -> None:
        """Task state that rides the timeline (entities, goals, metrics)."""

    def _reset_task(
        self,
        env_ids: torch.Tensor,
        clip_ids: torch.Tensor,
        time_steps: torch.Tensor,
        origins: torch.Tensor,
    ) -> None:
        """Task-side RSI, AFTER the robot's — so the RNG draw order is
        robot-then-task and stays stable when a task adds randomization."""

    def _update_task(self) -> None:
        """Per-step task update, BEFORE the frame advances."""

    # ── sampling ──

    def _sample_init_frame(
        self, env_ids: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Sample per-env (clip_id, init_frame) with phase annealing.

        _init_phase_max restricts WHERE the episode can START.
        Clips always run to their natural boundary.
        """
        n = len(env_ids)
        L = self.motion.max_clip_length

        allowed = self._clip_allowance(env_ids)  # (n, n_clips) | None

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
        """Sample a new clip + frame and RSI the robot onto it."""
        clip_ids, global_steps = self._sample_init_frame(env_ids)
        self._clip_ids[env_ids] = clip_ids
        self.time_steps[env_ids] = global_steps
        self._steps_past_end[env_ids] = 0

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
        s = sample_se3(self.cfg.pose_range, len(env_ids), self.device)
        root_pos = root_pos + s[:, 0:3]
        root_ori = quat_mul(quat_from_euler_xyz(s[:, 3], s[:, 4], s[:, 5]),
                            root_ori)

        s = sample_se3(self.cfg.velocity_range, len(env_ids), self.device)
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

        self._reset_task(env_ids, clip_ids, t, origins)

    # ── step ──

    def _update_command(self, env_ids: torch.Tensor | None = None) -> None:
        """Advance all commands on a policy step, or only reset environments.

        mjlab's command hook gained ``env_ids`` so its reset-time update does
        not advance unrelated environments.  Keeping ``None`` as the default
        also supports mjlab releases that still call this hook with no
        argument.
        """
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

        self._update_task()

        # Advance frame, freeze at the clip's last frame: the episode — not the
        # motion — owns resets. Past the boundary the reference holds still, so
        # the policy must stabilize into a steady-state hold and
        # `exceeded_motion_by_eps` truncates (time_out → V(s') bootstraps) after
        # ε frozen steps. Resampling happens only via env reset
        # (CommandTerm.reset → _resample_command).
        selected = slice(None) if env_ids is None else env_ids
        self.time_steps[selected] += 1
        clip_last = self.motion.clip_ends[self._clip_ids[selected]] - 1
        overrun = self.time_steps[selected] > clip_last
        self._steps_past_end[selected] += overrun.long()
        self.time_steps[selected] = torch.minimum(
            self.time_steps[selected], clip_last
        )

        self.update_relative_body_poses()

    # ── command (N-step, matches WBC IL convention) ──

    @property
    def command(self) -> torch.Tensor:
        """N-step [joint_pos, joint_vel] — matches the WBC generated_commands."""
        return torch.cat([
            self.motion_joint_pos_future.reshape(self._env.num_envs, -1),
            self.motion_joint_vel_future.reshape(self._env.num_envs, -1),
        ], dim=1)

    # ── N-step future reference (for phase-conditioned obs) ──

    def future_frames(self, steps: int, skip: int = 1) -> torch.Tensor:
        """(B, steps) future time indices at ``skip`` spacing, clamped within clip
        boundaries.

        Shared accessor with ``mocke.mdp.FutureMotionCommand`` — obs terms that
        own their window shape (e.g. the SONIC tokenizer) are duck-typed on it.
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


@dataclass(kw_only=True)
class MultiClipMotionCommandCfg(MotionCommandCfg):
    """MotionCommandCfg over a clip library instead of one file."""

    entity_name: str = "robot"

    # Multi-clip dataset root, or a list of motion folders (replaces the
    # single motion_file after init). Depth-invariant either way — see
    # `orcs.core.data.scan.motion_dirs`.
    dataset_dir: str | list[str] = ""

    # Motions/clips to skip — grammar in `orcs.core.data.scan.matches_exclude`:
    # "<motion>" or "<dataset>/<motion>" drops a whole motion; append
    # "/<sampleN>" to drop that one clip of it.
    exclude_motions: tuple[str, ...] | None = None

    # N-step future reference lookahead
    future_steps: int = 5

    # MotionCommand defaults — 14 tracked bodies
    anchor_body_name: str = "pelvis"
    body_names: tuple[str, ...] = _G1_BODY_NAMES
    # NOTE: base-class adaptive sampling is NOT wired to the multi-clip
    # sampler; pinned to "start" (per-clip adaptive sampling = future PR).
    sampling_mode: Literal["adaptive", "uniform", "start"] = "start"

    # Phase annealing: init_phase_max 1→0 over N policy updates.
    # Caps WHERE envs can start; clips always run to natural end.
    # 0 = disabled (init anywhere).
    init_phase_anneal_iterations: int = 0

    # Always init at frame 0 (init_phase=0), ignoring the [0, init_phase_max)
    # sampling window. Set True for play so every clip runs from its start.
    start_from_zero: bool = False

    def build(self, env: ManagerBasedRlEnv) -> MultiClipMotionCommand:
        return MultiClipMotionCommand(self, env)
