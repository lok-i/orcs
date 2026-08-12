"""Robot-only termination terms — motion-tracking tubes + base collapse.

The tubes read the reference through mjlab's `MotionCommand` anchor API, so
they work for any task whose command exposes it. Object/terrain tubes are the
owning task's.
"""

from __future__ import annotations

import math

import torch
from mjlab.envs import ManagerBasedRlEnv
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.utils.lab_api.math import quat_error_magnitude

__all__ = [
    "base_collapsed", "bad_anchor_pos", "bad_anchor_ori", "bad_point_anchor_pos",
    "exceeded_motion_by_eps",
]


def base_collapsed(
    env: ManagerBasedRlEnv,
    tilt_deg: float = 75.0,
    height: float = 0.36,
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),  # noqa: B008
) -> torch.Tensor:
    """True where the base is BOTH tilted past tilt_deg AND below height (collapsed)."""
    robot = env.scene[robot_cfg.name]
    tilted = robot.data.projected_gravity_b[:, 2] > -math.cos(math.radians(tilt_deg))
    low = robot.data.root_link_pos_w[:, 2] < height
    return tilted & low


def bad_anchor_pos(
    env: ManagerBasedRlEnv, command_name: str, threshold: float = 0.25
) -> torch.Tensor:
    """Pelvis z-only drift from motion reference > threshold. (B,) bool."""
    cmd = env.command_manager.get_term(command_name)
    err = (cmd.robot.data.root_link_pos_w[:, 2] - cmd.anchor_pos_w[:, 2]).abs()
    return err > threshold


def bad_anchor_ori(
    env: ManagerBasedRlEnv, command_name: str, threshold: float = 0.8
) -> torch.Tensor:
    """Pelvis orientation error from motion reference > threshold. (B,) bool."""
    cmd = env.command_manager.get_term(command_name)
    err = quat_error_magnitude(cmd.robot.data.root_link_quat_w, cmd.anchor_quat_w)
    return err > threshold


def bad_point_anchor_pos(
    env: ManagerBasedRlEnv, command_name: str, threshold: float = 0.75
) -> torch.Tensor:
    """Mapped pelvis distance from its source point exceeds ``threshold``."""
    cmd = env.command_manager.get_term(command_name)
    point_index = cmd.point_body_names.index(cmd.cfg.anchor_body_name)
    err = torch.linalg.vector_norm(
        cmd.robot_point_pos_w[:, point_index]
        - cmd.point_target_pos_w[:, point_index],
        dim=-1,
    )
    return err > threshold


def exceeded_motion_by_eps(
    env: ManagerBasedRlEnv, command_name: str, epsilon_steps: int = 25
) -> torch.Tensor:
    """True after the reference has held the clip's last frame for ε steps.

    The command freezes time_steps at the boundary (no mid-episode resample),
    so the policy must hold steady-state on the final frame through the ε
    padding. Truncation (not failure) — agent survived through the motion,
    so rsl_rl bootstraps V(s') instead of zeroing it.
    """
    cmd = env.command_manager.get_term(command_name)
    return cmd._steps_past_end >= epsilon_steps
