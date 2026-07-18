"""Object-manip reward terms — object tracking, goals, contact consistency."""

from __future__ import annotations

import torch
from mjlab.envs import ManagerBasedRlEnv
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.utils.lab_api.math import quat_error_magnitude

__all__ = [
    "orientation_success_bonus",
    "object_pos_tracking_reward",
    "object_ori_tracking_reward",
    "object_goal_ori_reward",
    "object_goal_pose_reward",
    "object_contact_consistency",
]


def _goal_quat(env: ManagerBasedRlEnv, command_name: str) -> torch.Tensor:
    """Get object goal quaternion from either command type."""
    term = env.command_manager.get_term(command_name)
    if hasattr(term, "object_goal_quat"):
        return term.object_goal_quat  # OmniObjectMotionCommand
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
# Object tracking rewards (OmniObjectMotionCommand property API:
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
