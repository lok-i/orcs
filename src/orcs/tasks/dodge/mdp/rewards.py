"""Reward terms — the dodge task signal, and only that.

One term. Posture, balance and naturalness are the frozen base's job (it is
tracking a nominal stand); re-specifying them as rewards would be a second,
weaker copy of a prior that is already structural.
"""

from __future__ import annotations

import torch
from mjlab.envs import ManagerBasedRlEnv

__all__ = ["ball_clearance"]


def ball_clearance(
    env: ManagerBasedRlEnv,
    robot_name: str = "robot",
    ball_name: str = "ball",
    pos_w: float = 0.9,
    vel_w: float = 0.1,
    pos_scale: float = 0.3,
    vel_scale: float = 1.0,
) -> torch.Tensor:
    """SMP/MimicKit's dodgeball reward (arXiv:2512.03028 ``compute_dodge_reward``).

        r = pos_w (1 - exp(-pos_scale d)) + vel_w exp(-vel_scale ||v_xy||^2)

    `d` is the root->ball distance. The distance term SATURATES, so once the ball
    is clear there is no gradient pulling the robot to keep fleeing — it settles;
    the stillness term makes it move only when it must. That pairing is the whole
    "stand, sidestep the throw, settle" behaviour, and it is the entire task
    signal in SMP. Privileged (ground-truth ball position); the vision actor sees
    none of it.

    A parked ball is far, so `d` is large and the distance term is a near
    constant between throws — the robot is then driven only to stand still.
    """
    robot, ball = env.scene[robot_name], env.scene[ball_name]
    d = torch.linalg.norm(
        ball.data.root_link_pos_w - robot.data.root_link_pos_w, dim=-1)
    v = torch.sum(torch.square(robot.data.root_link_lin_vel_w[:, :2]), dim=-1)
    return pos_w * (1.0 - torch.exp(-pos_scale * d)) + vel_w * torch.exp(-vel_scale * v)
