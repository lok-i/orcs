"""Object-manip reward terms — object tracking, goals, contact consistency."""

from __future__ import annotations

import torch
from mjlab.envs import ManagerBasedRlEnv
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.utils.lab_api.math import (
    quat_apply,
    quat_apply_inverse,
    quat_error_magnitude,
)

__all__ = [
    "orientation_success_bonus",
    "object_pos_tracking_reward",
    "object_ori_tracking_reward",
    "object_goal_ori_reward",
    "object_goal_pose_reward",
    "object_contact_consistency",
    "keypoint_position_error_exp",
    "keypoint_direction_error_exp",
]


def _goal_quat(env: ManagerBasedRlEnv, command_name: str) -> torch.Tensor:
    """Get object goal quaternion from either command type."""
    term = env.command_manager.get_term(command_name)
    if hasattr(term, "object_goal_quat"):
        return term.object_goal_quat  # ObjectMotionCommand
    return term.command  # ReorientationCommand (.command = target_quat)


def object_goal_ori_reward(
    env: ManagerBasedRlEnv,
    object_cfg: SceneEntityCfg,
    command_name: str,
    std: float = 2.24,
) -> torch.Tensor:
    """Gaussian kernel on orientation error: exp(-err^2 / std^2). Returns (B,)."""
    obj = env.scene[object_cfg.name]
    target = _goal_quat(env, command_name)
    err = quat_error_magnitude(obj.data.root_link_quat_w, target)
    return torch.exp(-err.square() / (std * std))


def orientation_success_bonus(
    env: ManagerBasedRlEnv,
    object_cfg: SceneEntityCfg,
    command_name: str,
    threshold: float = 0.2,
) -> torch.Tensor:
    """Binary bonus when orientation error < threshold. Returns (B,)."""
    obj = env.scene[object_cfg.name]
    target = _goal_quat(env, command_name)
    err = quat_error_magnitude(obj.data.root_link_quat_w, target)
    return (err < threshold).float()


# ---------------------------------------------------------------------------
# Object tracking rewards (ObjectMotionCommand property API:
# .object, .object_pos_w, .object_quat_w)
# ---------------------------------------------------------------------------

def object_pos_tracking_reward(
    env: ManagerBasedRlEnv, command_name: str, std: float = 0.3
) -> torch.Tensor:
    """exp(-||obj - ref_obj||^2 / std^2). Mirrors motion_global_anchor_position_error_exp."""
    cmd = env.command_manager.get_term(command_name)
    err = (cmd.object.data.root_link_pos_w - cmd.object_pos_w).square().sum(dim=-1)
    return torch.exp(-err / (std * std))


def object_ori_tracking_reward(
    env: ManagerBasedRlEnv, command_name: str, std: float = 0.4
) -> torch.Tensor:
    """exp(-theta^2 / std^2). Mirrors motion_global_anchor_orientation_error_exp."""
    cmd = env.command_manager.get_term(command_name)
    err = quat_error_magnitude(cmd.object.data.root_link_quat_w, cmd.object_quat_w).square()
    return torch.exp(-err / (std * std))


# ---------------------------------------------------------------------------
# Contact-graph rewards (demo ref: command.object_bodywise_contact, live: one
# multi-primary object-filtered ContactSensor)
# ---------------------------------------------------------------------------

def _bodywise_contact_force(
    env: ManagerBasedRlEnv, sensor_name: str, body_names: tuple[str, ...]
) -> torch.Tensor:
    """(B, K) per-body object-contact force magnitude, columns reordered from the
    sensor's model-order primaries to `body_names` (1:1 with the demo ref)."""
    sensor = env.scene.sensors[sensor_name]
    force = torch.norm(sensor.data.force, dim=-1)  # (B, P) sensor order
    cols = [sensor.primary_names.index(b) for b in body_names]
    return force[:, cols]


def object_contact_consistency(
    env: ManagerBasedRlEnv,
    command_name: str,
    sensor_name: str,
    contact_force_threshold: float = 0.1,
) -> torch.Tensor:
    """SUGAR r: fraction of contact-graph nodes matching the demo ref (partial
    credit). Ref/live column order = command.cfg.contact_graph_body_names."""
    cmd = env.command_manager.get_term(command_name)
    ref = cmd.object_bodywise_contact  # (B, K) demo per-body contact (0/1)
    force = _bodywise_contact_force(env, sensor_name, cmd.cfg.contact_graph_body_names)
    live = (force > contact_force_threshold).float()  # (B, K)
    r = (live == ref).float().mean(dim=1)

    # --- debug pretty-print (env 0 only) ---
    # if True:
    #     short = [n[:10].center(10) for n in cmd.cfg.contact_graph_body_names]
    #     header = "  body  | " + " | ".join(short)
    #     sep = "-" * len(header)
    #     def _row(label, t):
    #         vals = " | ".join(f"{v:10.3f}" for v in t[0].tolist())
    #         return f" {label:>5}  | {vals}"
    #     print(f"\n{sep}\n{header}\n{sep}")
    #     print(_row("ref", ref))
    #     print(_row("live", live))
    #     print(_row("force", force))
    #     print(f" {'R':>5}  | {r[0].item():.4f}")
    #     print(sep)

    return r


# ---------------------------------------------------------------------------
# SMPL keypoint tracking (position + body-fixed direction), no retargeting.
# Ported from geometry-aware-policy's src.rewards.keypoint. Each keypoint
# (hands / feet / pelvis / head) is rewarded for matching the SMPL reference in
# two ways — a position term and a body-fixed *direction* term — under a
# configurable frame:
#   * hands  -> palm normal, in the OBJECT frame (robot-vs-sim-object compared
#     to reference-vs-reference-object, so the world offset cancels — the
#     morphology-free interaction signal).
#   * feet / pelvis / head -> forward direction, in the WORLD (env-local) frame.
# The robot side is read live from the sim (body_link_pos_w / body_link_quat_w
# rotated through the per-keypoint local axis). The SMPL reference position
# comes from the command's ghost joints (smpl_joints_viz); the reference
# direction comes from the loader's per-frame smpl_dirs buffer.
# All rewards are exp(-||err||^2 / std^2).
# ---------------------------------------------------------------------------


class _Keypoint:
    """One tracked keypoint: robot body + local axis/offset, SMPL draw joint."""

    __slots__ = ("name", "robot_body", "robot_axis", "robot_offset", "smpl_draw_joint")

    def __init__(self, name, robot_body, robot_axis, smpl_draw_joint, robot_offset):
        self.name = name
        self.robot_body = robot_body
        self.robot_axis = robot_axis
        self.robot_offset = robot_offset
        self.smpl_draw_joint = smpl_draw_joint


# Keypoint calibration (see src.debug.directions.KEYPOINTS). robot_axis is the
# body-fixed unit direction in the link frame; robot_offset lifts the point onto
# the hand plate / head; smpl_draw_joint is the SMPL-24 joint for the ref pos.
_KEYPOINTS = [
    _Keypoint("left_palm", "left_wrist_yaw_link", (0.0, -1.0, 0.0), 22, (0.12, -0.01, 0.0)),
    _Keypoint("right_palm", "right_wrist_yaw_link", (0.0, 1.0, 0.0), 23, (0.12, 0.01, 0.0)),
    _Keypoint("left_foot", "left_ankle_roll_link", (1.0, 0.0, 0.0), 10, (0.0, 0.0, 0.0)),
    _Keypoint("right_foot", "right_ankle_roll_link", (1.0, 0.0, 0.0), 11, (0.0, 0.0, 0.0)),
    _Keypoint("pelvis", "pelvis", (1.0, 0.0, 0.0), 0, (0.0, 0.0, 0.0)),
    _Keypoint("head", "torso_link", (1.0, 0.0, 0.0), 15, (0.0, 0.0, 0.5)),
]
_KP = {k.name: k for k in _KEYPOINTS}
_BODY_IDX: dict[tuple[int, str], int] = {}  # (env id, body name) -> body index


def _robot_body_index(env: ManagerBasedRlEnv, body_name: str) -> int:
    """Resolve (and cache) a robot body index from its name."""
    key = (id(env), body_name)
    if key not in _BODY_IDX:
        ids, _ = env.scene["robot"].find_bodies(body_name)
        if not ids:
            raise KeyError(f"robot body {body_name!r} not found")
        _BODY_IDX[key] = ids[0]
    return _BODY_IDX[key]


def _robot_pos_dir(env: ManagerBasedRlEnv, kp) -> tuple[torch.Tensor, torch.Tensor]:
    """Robot keypoint world position (with local offset) and world direction."""
    robot = env.scene["robot"]
    idx = _robot_body_index(env, kp.robot_body)
    pos = robot.data.body_link_pos_w[:, idx]          # (B, 3)
    quat = robot.data.body_link_quat_w[:, idx]        # (B, 4)
    b = pos.shape[0]
    off = torch.tensor(kp.robot_offset, dtype=pos.dtype, device=pos.device).view(1, 3).expand(b, 3)
    axis = torch.tensor(kp.robot_axis, dtype=pos.dtype, device=pos.device).view(1, 3).expand(b, 3)
    pos = pos + quat_apply(quat, off)
    direction = quat_apply(quat, axis)
    return pos, direction


def _ref_pos_dir(env: ManagerBasedRlEnv, cmd, kp) -> tuple[torch.Tensor, torch.Tensor]:
    """SMPL reference keypoint world position and world direction."""
    t = cmd.time_steps
    origins = env.scene.env_origins
    ref_pos = cmd.motion.smpl_joints_viz[t, kp.smpl_draw_joint] + origins
    ref_dirs = getattr(cmd.motion, "smpl_dirs", None)
    if ref_dirs is not None and kp.name in ref_dirs:
        ref_dir = ref_dirs[kp.name][t]
    else:  # buffer not present (e.g. robot clips) -> safe zeros
        ref_dir = torch.zeros_like(ref_pos)
    return ref_pos, ref_dir


def _object_frames(env: ManagerBasedRlEnv, cmd):
    """(sim_obj_pos, sim_obj_quat, ref_obj_pos, ref_obj_quat), all world frame."""
    obj = env.scene["object"]
    sim_pos, sim_quat = obj.data.root_link_pos_w, obj.data.root_link_quat_w
    t = cmd.time_steps
    ref_pos = cmd.motion.obj_pos[t] + env.scene.env_origins
    ref_quat = cmd.motion.obj_quat[t]
    return sim_pos, sim_quat, ref_pos, ref_quat


def keypoint_position_error_exp(
    env: ManagerBasedRlEnv, command_name: str, keypoint: str,
    frame: str = "world", std: float = 0.3,
) -> torch.Tensor:
    """``exp(-||p_robot - p_ref||^2 / std^2)`` for one keypoint, in ``frame``.

    ``frame="object"`` expresses each side's position relative to *its own*
    object (sim object for the robot, reference object for the SMPL ref) before
    comparing, so the world offset cancels.
    """
    kp = _KP[keypoint]
    cmd = env.command_manager.get_term(command_name)
    rp, _ = _robot_pos_dir(env, kp)
    refp, _ = _ref_pos_dir(env, cmd, kp)
    if frame == "object":
        so, sq, ro, rq = _object_frames(env, cmd)
        rp = quat_apply_inverse(sq, rp - so)
        refp = quat_apply_inverse(rq, refp - ro)
    err = (rp - refp).pow(2).sum(dim=-1)
    return torch.exp(-err / (std * std))


def keypoint_direction_error_exp(
    env: ManagerBasedRlEnv, command_name: str, keypoint: str,
    frame: str = "world", std: float = 0.4,
) -> torch.Tensor:
    """``exp(-||d_robot - d_ref||^2 / std^2)`` for one keypoint direction.

    Unit-vector directions, so ``||d_robot - d_ref||^2 = 2(1 - cos angle)``.
    ``frame="object"`` rotates each direction into its own object frame first.
    """
    kp = _KP[keypoint]
    cmd = env.command_manager.get_term(command_name)
    _, rd = _robot_pos_dir(env, kp)
    _, refd = _ref_pos_dir(env, cmd, kp)
    if frame == "object":
        so, sq, ro, rq = _object_frames(env, cmd)
        rd = quat_apply_inverse(sq, rd)
        refd = quat_apply_inverse(rq, refd)
    err = (rd - refd).pow(2).sum(dim=-1)
    return torch.exp(-err / (std * std))


def object_goal_pose_reward(
    env: ManagerBasedRlEnv,
    object_cfg: SceneEntityCfg,
    command_name: str,
    std_pos: float = 0.3,
    std_quat: float = 0.4,
) -> torch.Tensor:
    """exp(-err) for object pose vs goal (final clip frame)."""
    cmd = env.command_manager.get_term(command_name)
    obj = env.scene[object_cfg.name]
    pos_err = (obj.data.root_link_pos_w - cmd.object_goal_pos_env - env.scene.env_origins).square().sum(dim=-1)
    ori_err = quat_error_magnitude(obj.data.root_link_quat_w, cmd.object_goal_quat).square()
    return torch.exp(-pos_err / (std_pos * std_pos)) + torch.exp(-ori_err / (std_quat * std_quat))
