"""Object-manip observation terms — body-relative object state, goals, object tracking refs."""

from __future__ import annotations

from typing import cast

import torch
from mjlab.envs import ManagerBasedRlEnv
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.utils.lab_api.math import (
    matrix_from_quat,
    quat_apply_inverse,
    subtract_frame_transforms,
)
from mocke.mdp.observations import (  # noqa: F401 — re-exported into `mdp.*`
    motion_anchor_ori_b_future,
    motion_anchor_pos_b_future,
)

__all__ = [
    "object_pose_b",
    "object_twist_b",
    # tracking obs (ObjectMotionCommand)
    "object_pos_w_obs",
    "object_ori_mat6d_w",
    "object_lin_vel_w_obs",
    "object_ang_vel_w_obs",
    "object_goal_pos_env",
    "object_goal_ori_mat6d",
    "bodywise_contact_cmd",
    "bodywise_saturated_force",
    "robot_root_lin_vel_cmd",
    "robot_root_ang_vel_cmd",
    "motion_object_pos_b_future",
    "motion_object_ori_b_future",
    "unweighted_reward_vector",
    # re-exported from mocke: the reference-anchor error is the TRACKING layer's,
    # not the object task's. One definition, so orcs and a vision consumer can
    # never silently bind different implementations of the same term.
    "motion_anchor_pos_b_future",
    "motion_anchor_ori_b_future",
]

def _quat_to_mat6d(quat: torch.Tensor) -> torch.Tensor:
    """(B, 4) wxyz quat -> (B, 6) mat6d [col0, col1] of rotation matrix."""
    mat = matrix_from_quat(quat)  # (B, 3, 3)
    return mat[..., :2].reshape(quat.shape[0], -1)  # (B, 6)


def object_pose_b(
    env: ManagerBasedRlEnv,
    object_cfg: SceneEntityCfg,
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),  # noqa: B008
) -> torch.Tensor:
    """Object pose in robot body frame: pos(3) + mat6d(6) -> (B, 9)."""
    robot = env.scene[robot_cfg.name]
    obj = env.scene[object_cfg.name]
    pos_b, quat_b = subtract_frame_transforms(
        robot.data.root_link_pos_w, robot.data.root_link_quat_w,
        obj.data.root_link_pos_w, obj.data.root_link_quat_w,
    )
    return torch.cat([pos_b, _quat_to_mat6d(quat_b)], dim=-1)

def object_twist_b(
    env: ManagerBasedRlEnv,
    object_cfg: SceneEntityCfg,
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),  # noqa: B008
) -> torch.Tensor:
    """Object twist in robot body frame: lin_vel(3) + ang_vel(3) -> (B, 6)."""
    robot = env.scene[robot_cfg.name]
    obj = env.scene[object_cfg.name]
    robot_quat = robot.data.root_link_quat_w
    lin_vel_b = quat_apply_inverse(robot_quat, obj.data.root_link_lin_vel_w)
    ang_vel_b = quat_apply_inverse(robot_quat, obj.data.root_link_ang_vel_w)
    return torch.cat([lin_vel_b, ang_vel_b], dim=-1)

# ---------------------------------------------------------------------------
# Tracking obs (ObjectMotionCommand)
# ---------------------------------------------------------------------------

def _get_omni_cmd(env: ManagerBasedRlEnv, command_name: str):
    from orcs.tasks.uolm.mdp.commands import ObjectMotionCommand
    return cast(ObjectMotionCommand, env.command_manager.get_term(command_name))


# -- augmentation stream: object state + goal --

def object_pos_w_obs(
    env: ManagerBasedRlEnv,
    object_cfg: SceneEntityCfg,
) -> torch.Tensor:
    """Object position in env frame (world - env_origin) -> (B, 3)."""
    obj = env.scene[object_cfg.name]
    return obj.data.root_link_pos_w - env.scene.env_origins


def object_ori_mat6d_w(
    env: ManagerBasedRlEnv,
    object_cfg: SceneEntityCfg,
) -> torch.Tensor:
    """Object orientation as mat6d -> (B, 6)."""
    obj = env.scene[object_cfg.name]
    return _quat_to_mat6d(obj.data.root_link_quat_w)


def object_lin_vel_w_obs(
    env: ManagerBasedRlEnv,
    object_cfg: SceneEntityCfg,
) -> torch.Tensor:
    """Object linear velocity world frame -> (B, 3)."""
    return env.scene[object_cfg.name].data.root_link_lin_vel_w


def object_ang_vel_w_obs(
    env: ManagerBasedRlEnv,
    object_cfg: SceneEntityCfg,
) -> torch.Tensor:
    """Object angular velocity world frame -> (B, 3)."""
    return env.scene[object_cfg.name].data.root_link_ang_vel_w


def unweighted_reward_vector(
    env: ManagerBasedRlEnv, enabled: bool = False, terms: list[str] | None = None
) -> torch.Tensor:
    """(B, K) per-term UNWEIGHTED reward rates — critic-only conditioning,
    V(concat(s, r)). Reads the reward manager's step cache (already computed
    this step from the SAME post-step state the obs describe; zero recompute).

    enabled=False -> zeros of the same shape: critic architecture stays
    byte-identical across the with/without A/B, only information flips.
    Toggle: --env.observations.critic.terms.reward_vec.params.enabled True
    terms=[...] -> only those reward terms, in the given order (e.g. the
    task-layer subset as a MuZero-style aux prediction target).

    Notes:
      - Obs-manager dim probe runs before the reward manager exists -> zeros
        fallback sized from cfg.rewards / terms.
      - First step after reset reads the pre-reset cache (one-frame stale).
      - Zero-weight terms read 0 (reward manager skips them).
    """
    n = len(terms) if terms else len(env.cfg.rewards)
    if not enabled or not hasattr(env, "reward_manager"):
        return torch.zeros(env.num_envs, n, device=env.device)
    rm = env.reward_manager
    w = getattr(env, "_reward_vec_inv_w", None)
    if w is None:
        # _step_reward stores raw*weight -> divide weights back out
        w = torch.tensor(
            [c.weight if c.weight != 0.0 else 1.0 for c in rm._term_cfgs],
            device=env.device,
        )
        env._reward_vec_inv_w = w
    vec = rm._step_reward / w
    if terms:
        sel = getattr(env, "_reward_vec_sel", None)
        if sel is None:
            sel = env._reward_vec_sel = {}
        key = tuple(terms)
        if key not in sel:
            sel[key] = torch.tensor(
                [rm._term_names.index(t) for t in key], device=env.device)
        vec = vec[:, sel[key]]
    return vec


def object_goal_pos_env(env: ManagerBasedRlEnv, command_name: str) -> torch.Tensor:
    """Object goal position in env frame -> (B, 3)."""
    cmd = _get_omni_cmd(env, command_name)
    return cmd.object_goal_pos_env


def object_goal_ori_mat6d(env: ManagerBasedRlEnv, command_name: str) -> torch.Tensor:
    """Object goal orientation as mat6d -> (B, 6)."""
    cmd = _get_omni_cmd(env, command_name)
    return _quat_to_mat6d(cmd.object_goal_quat)


# -- sys1 command stream (SUGAR c_t parity; robot-state language only) --

def bodywise_contact_cmd(env: ManagerBasedRlEnv, command_name: str) -> torch.Tensor:
    """l_cmd_t: reference per-body robot<->object contact flags -> (B, K).

    K = len(cfg.contact_graph_body_names), same column order as the
    contact_consistency reward (ContactSchedule truth). SUGAR streams a
    single scalar; we keep the per-body vector (which-limb information)."""
    cmd = _get_omni_cmd(env, command_name)
    return cmd.object_bodywise_contact.float()


def bodywise_saturated_force(
    env: ManagerBasedRlEnv,
    command_name: str,
    sensor_name: str,
    body_names: tuple[str, ...] | None = None,
    f_max: float = 50.0,
) -> torch.Tensor:
    """tanh(||f|| / f_max): live per-body robot<->object contact force -> (B, K).

    The live twin of ``bodywise_contact_cmd`` (reference flags). ``body_names``
    picks the columns and defaults to the full graph
    (``cmd.cfg.contact_graph_body_names``, 1:1 with the demo ref and the
    ``object_contact_consistency`` reward); a caller may pass a subset.

    Saturating, not binary: a hard threshold manufactures label chatter at the
    boundary, and raw ||f|| is spike-and-slab (a consumer's normalizer std gets
    eaten by the impact tail). tanh is the variance-stabilizing transform —
    linear where the signal is, saturating on impacts.

    ``f_max`` is a ROBOT constant, never per-object: normalizing by object mass
    would inject an unobservable into the signal and silently reweight it per
    env. From the actuation bound max_q J^T(q) tau_lim; G1 stance-pull peak is
    57.4 N (https://arxiv.org/pdf/2505.06776), so 50 N is the tightest value
    leaving the whole arm-achievable range unsaturated. Above it is
    support/impact load, not manipulation. One flat knob, so on a wide
    ``body_names`` a torso/leg column runs saturated and degenerates to a
    near-binary "load-bearing or not". Revisit on a robot swap.
    """
    from orcs.tasks.uolm.mdp.rewards import _bodywise_contact_force

    cmd = _get_omni_cmd(env, command_name)
    force = _bodywise_contact_force(
        env, sensor_name, body_names or cmd.cfg.contact_graph_body_names)
    return torch.tanh(force / f_max)  # (B, K)


def robot_root_lin_vel_cmd(env: ManagerBasedRlEnv, command_name: str) -> torch.Tensor:
    """v_cmd_t: reference anchor linear velocity in the REF anchor's own frame
    -> (B, 3). Pure function of the reference stream (no live-state coupling),
    SUGAR ref_anchor_lin_vel_b convention — sys1 can emit it open-loop."""
    cmd = _get_omni_cmd(env, command_name)
    return quat_apply_inverse(cmd.anchor_quat_w, cmd.anchor_lin_vel_w)


def robot_root_ang_vel_cmd(env: ManagerBasedRlEnv, command_name: str) -> torch.Tensor:
    """w_cmd_t: reference anchor angular velocity in the REF anchor's own frame
    -> (B, 3). Same convention as robot_root_lin_vel_cmd."""
    cmd = _get_omni_cmd(env, command_name)
    return quat_apply_inverse(cmd.anchor_quat_w, cmd.anchor_ang_vel_w)


def motion_object_pos_b_future(env: ManagerBasedRlEnv, command_name: str) -> torch.Tensor:
    """Future N-step object position in robot body frame -> (B, N*3)."""
    cmd = _get_omni_cmd(env, command_name)
    future_pos = cmd.motion_object_pos_w_future   # (B, N, 3)
    future_quat = cmd.motion_object_quat_w_future  # (B, N, 4)
    robot_pos = cmd.robot_anchor_pos_w[:, None, :].expand_as(future_pos)
    robot_quat = cmd.robot_anchor_quat_w[:, None, :].expand_as(future_quat)
    pos_b, _ = subtract_frame_transforms(robot_pos, robot_quat, future_pos, future_quat)
    return pos_b.reshape(env.num_envs, -1)


def motion_object_ori_b_future(env: ManagerBasedRlEnv, command_name: str) -> torch.Tensor:
    """Future N-step object orientation in robot body frame (mat6d) -> (B, N*6)."""
    cmd = _get_omni_cmd(env, command_name)
    future_pos = cmd.motion_object_pos_w_future
    future_quat = cmd.motion_object_quat_w_future  # (B, N, 4)
    robot_pos = cmd.robot_anchor_pos_w[:, None, :].expand_as(future_pos)
    robot_quat = cmd.robot_anchor_quat_w[:, None, :].expand_as(future_quat)
    _, ori_b = subtract_frame_transforms(robot_pos, robot_quat, future_pos, future_quat)
    mat = matrix_from_quat(ori_b)  # (B, N, 3, 3)
    return mat[..., :2].reshape(env.num_envs, -1)


# ---------------------------------------------------------------------------
# Anchor tracking obs (future-window command interface; critic stream)
# ---------------------------------------------------------------------------
