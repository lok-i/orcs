"""Observation terms — the ball, as the privileged side sees it.

THE swap target: a vision consumer replaces exactly these terms with encoder
features. Everything is in the robot's YAW frame, which is what a head camera
also measures in — so the privileged and the visual row differ in modality, not
in frame, and a delta between them is the cost of perception alone.
"""

from __future__ import annotations

import torch
from mjlab.envs import ManagerBasedRlEnv
from mjlab.utils.lab_api.math import quat_apply_inverse, yaw_quat

__all__ = ["ball_pos_b", "ball_vel_b", "ball_radius"]


def _rel(env: ManagerBasedRlEnv, robot_name: str, ball_name: str):
    robot, ball = env.scene[robot_name], env.scene[ball_name]
    yq = yaw_quat(robot.data.root_link_quat_w)
    return robot, ball, yq


def ball_pos_b(
    env: ManagerBasedRlEnv, robot_name: str = "robot", ball_name: str = "ball"
) -> torch.Tensor:
    """Ball position relative to the robot root, yaw frame. (B, 3)."""
    robot, ball, yq = _rel(env, robot_name, ball_name)
    return quat_apply_inverse(
        yq, ball.data.root_link_pos_w - robot.data.root_link_pos_w)


def ball_vel_b(
    env: ManagerBasedRlEnv, robot_name: str = "robot", ball_name: str = "ball"
) -> torch.Tensor:
    """Ball velocity relative to the robot, yaw frame. (B, 3).

    Relative, not absolute: closing rate is what the barrier and the evasion are
    about, and it is what a looming cue in an image encodes.
    """
    robot, ball, yq = _rel(env, robot_name, ball_name)
    return quat_apply_inverse(
        yq, ball.data.root_link_lin_vel_w - robot.data.root_link_lin_vel_w)


def ball_radius(
    env: ManagerBasedRlEnv, ball_name: str = "ball", geom_name: str = "ball_collision"
) -> torch.Tensor:
    """Per-env ball radius. (B, 1). **Not in any group today** — see below.

    Reads the per-world `geom_size`, which is what a size-randomization event
    writes, so this is that knob's read side. Until such an event exists the
    value is a run constant and observing it informs nothing
    (`observation_cfgs.ball_state_terms`), so it is defined and unused on
    purpose rather than wired in advance.
    """
    key = "_dodge_ball_geom_id"
    if not hasattr(env, key):
        setattr(env, key, int(env.sim.mj_model.geom(f"{ball_name}/{geom_name}").id))
    return env.sim.model.geom_size[:, getattr(env, key), 0:1]
