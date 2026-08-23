"""Task-blind rewards for named point-reference tracking."""

from __future__ import annotations

import torch
from mjlab.envs import ManagerBasedRlEnv

__all__ = ["point_position_error_exp", "point_velocity_error_exp"]


def point_position_error_exp(
    env: ManagerBasedRlEnv, command_name: str, std: float
) -> torch.Tensor:
    """Exponential mean squared error over every mapped source point."""
    command = env.command_manager.get_term(command_name)
    error = torch.square(
        command.point_target_pos_w - command.robot_point_pos_w
    ).sum(-1)
    return torch.exp(-error.mean(-1) / std**2)


def point_velocity_error_exp(
    env: ManagerBasedRlEnv, command_name: str, std: float
) -> torch.Tensor:
    """Exponential mean squared world-velocity error over mapped points."""
    command = env.command_manager.get_term(command_name)
    error = torch.square(
        command.point_target_vel_w - command.robot_point_vel_w
    ).sum(-1)
    return torch.exp(-error.mean(-1) / std**2)
