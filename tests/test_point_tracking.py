from __future__ import annotations

from types import SimpleNamespace

import torch

from orcs.core.mdp.rewards import (
    point_position_error_exp,
    point_velocity_error_exp,
)


class _Commands:
    def __init__(self, command) -> None:
        self.command = command

    def get_term(self, name: str):
        assert name == "motion"
        return self.command


def test_point_rewards_use_all_points_and_only_live_robot_state():
    target_pos = torch.zeros(2, 3, 3)
    robot_pos = target_pos.clone()
    robot_pos[1, 2, 0] = 0.3
    target_vel = torch.zeros_like(target_pos)
    robot_vel = target_vel.clone()
    robot_vel[1, 1, 1] = 1.0
    command = SimpleNamespace(
        point_target_pos_w=target_pos,
        robot_point_pos_w=robot_pos,
        point_target_vel_w=target_vel,
        robot_point_vel_w=robot_vel,
    )
    env = SimpleNamespace(command_manager=_Commands(command))

    pos = point_position_error_exp(env, "motion", std=0.3)
    vel = point_velocity_error_exp(env, "motion", std=1.0)

    assert torch.allclose(pos[0], torch.tensor(1.0))
    assert torch.allclose(vel[0], torch.tensor(1.0))
    assert 0.0 < pos[1] < 1.0
    assert 0.0 < vel[1] < 1.0
