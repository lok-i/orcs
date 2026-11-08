"""Vectorized first-episode measurements at MJLab's pre-reset metric hook."""

from __future__ import annotations

import torch
from mjlab.utils.lab_api.math import quat_error_magnitude


def tracking_errors(command) -> dict[str, torch.Tensor]:
    """Measure matched points, never robot seed bodies for SMPL references."""
    if hasattr(command, "point_target_pos_w"):
        target, actual = command.point_target_pos_w, command.robot_point_pos_w
        target_v, actual_v = command.point_target_vel_w, command.robot_point_vel_w
    else:
        target, actual = command.body_pos_w, command.robot_body_pos_w
        target_v, actual_v = command.body_lin_vel_w, command.robot_body_lin_vel_w
    target_root, actual_root = command.anchor_pos_w, command.robot_anchor_pos_w
    norm = torch.linalg.vector_norm
    return {
        "global_position_m": norm(actual - target, dim=-1).mean(-1),
        "relative_position_m": norm(
            (actual - actual_root[:, None]) - (target - target_root[:, None]), dim=-1).mean(-1),
        "root_position_m": norm(actual_root - target_root, dim=-1),
        "root_orientation_rad": quat_error_magnitude(
            command.robot_anchor_quat_w, command.anchor_quat_w),
        "point_velocity_m_s": norm(actual_v - target_v, dim=-1).mean(-1),
    }


def capture_step(env):
    """A normal MetricsTerm, run after terminations and before automatic reset."""
    capture = getattr(env, "_motion_repertoire_capture", None)
    if capture is None:
        return torch.zeros(env.num_envs, device=env.device)
    return capture.step(env)


class TrialCapture:
    def __init__(self, env, spec, clips: list[dict], bins: int):
        self.command = env.command_manager.get_term(spec.command_name)
        self.spec, self.bins = spec, bins
        self.n, self.device, self.dt = env.num_envs, env.device, env.step_dt
        self.indices = torch.arange(self.n, device=self.device)
        self.lengths = torch.tensor([c["frames"] for c in clips], device=self.device)
        self.status = torch.zeros(self.n, dtype=torch.long, device=self.device)
        self.progress = torch.zeros(self.n, device=self.device)
        self.steps = torch.zeros(self.n, dtype=torch.long, device=self.device)
        self.end_bin = torch.zeros_like(self.steps)
        self.samples = torch.zeros(self.n, bins + 1, dtype=torch.long, device=self.device)
        self.sums = {key: torch.zeros(self.n, bins + 1, device=self.device)
                     for key in tracking_errors(self.command)}
        self.terms = {key: torch.zeros(self.n, dtype=torch.bool, device=self.device)
                      for key in env.termination_manager.active_terms}
        self.hold = (env.cfg.terminations[spec.completion_term].params["epsilon_steps"]
                     if spec.horizon == "clip" else 0)

    @property
    def active(self):
        return self.status == 0

    def invalidate(self, mask):
        self.status[self.active & mask] = 3

    def step(self, env):
        active = self.active
        cmd = self.command
        progress = (cmd.local_reference_frames / (self.lengths - 1).clamp(min=1)
                    if self.spec.horizon == "clip"
                    else env.episode_length_buf / env.max_episode_length)
        progress = progress.clamp(0, 1)
        bin_index = (progress * self.bins).long().clamp(max=self.bins - 1)
        if self.spec.horizon == "clip":
            bin_index = torch.where(cmd.hold_steps > 0, self.bins, bin_index)
        values = tracking_errors(cmd)
        finite = torch.isfinite(env.reward_buf)
        for value in values.values():
            finite &= torch.isfinite(value)
        self.invalidate(~finite)
        valid = active & finite
        self.progress[active] = progress[active]
        self.steps[active] = env.episode_length_buf[active]
        self.end_bin[active] = bin_index[active]
        self.samples[self.indices, bin_index] += valid.long()
        for key, value in values.items():
            self.sums[key][self.indices, bin_index] += torch.where(valid, value, 0.0)
        tm = env.termination_manager
        done = tm.terminated | tm.time_outs
        first = valid & done
        for key in self.terms:
            self.terms[key] |= first & tm.get_term(key)
        success = tm.get_term(self.spec.completion_term)
        if self.spec.horizon == "clip":
            success = success & (cmd.hold_steps >= self.hold)
        else:
            success = success & (env.episode_length_buf >= env.max_episode_length)
        self.status[first] = 4  # unexpected truncation is censoring
        self.status[first & success & ~tm.terminated] = 1
        self.status[first & tm.terminated] = 2  # failure wins at the boundary
        return progress

    def records(self, assignments, batch_index: int, seed: int) -> list[dict]:
        status = self.status.cpu().tolist()
        progress, steps, end_bins = (v.cpu().tolist() for v in (self.progress, self.steps, self.end_bin))
        samples = self.samples.cpu().tolist()
        sums = {k: v.cpu().tolist() for k, v in self.sums.items()}
        terms = {k: v.cpu().tolist() for k, v in self.terms.items()}
        records = []
        for i, (trial_id, motion_index, repeat) in enumerate(assignments):
            count = sum(samples[i])
            records.append(dict(
                trial_id=trial_id, motion_index=motion_index, repeat=repeat,
                batch=batch_index, env_id=i, seed=seed,
                status=("censored", "completed", "failed", "invalid", "censored")[status[i]],
                progress=progress[i], elapsed_s=steps[i]*self.dt, end_bin=end_bins[i],
                terminations=[key for key, value in terms.items() if value[i]],
                metrics={k: sum(v[i])/count if count else None for k, v in sums.items()},
                bin_samples=samples[i],
                bin_metrics={k: [s/n if n else None for s, n in zip(v[i], samples[i], strict=True)]
                             for k, v in sums.items()},
            ))
        return records
