"""Crude, task-agnostic virtual-force assistance for kinematic retargeting.

A frozen SONIC policy remains the motion generator; this module supplies
temporary world-frame wrenches that pull a simulated G1 toward scaled SMPL
landmarks.  Policy training later refines this kinematic result into a dynamic
retarget.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import torch
from mjlab.utils.lab_api.math import (
    axis_angle_from_quat,
    quat_apply,
    quat_apply_inverse,
    quat_conjugate,
    quat_mul,
)

from orcs.core.data.point_reference import (
    G1_SMPL_BODY_MAP,
    infer_morphology_scale,
    scaled_smpl_targets,
)

if TYPE_CHECKING:
    from mjlab.envs import ManagerBasedRlEnv

__all__ = [
    "G1_SMPL_BODY_MAP",
    "DEFAULT_ASSISTANCE_GAINS",
    "AssistanceGains",
    "AssistanceSnapshot",
    "AssistedMotionController",
    "infer_morphology_scale",
    "robot_relative_object_target",
    "scaled_smpl_targets",
]

_ANCHOR_BODY = "torso_link"
_GRAVITY = 9.81


@dataclass(frozen=True)
class AssistanceGains:
    """Global force-assistance tuning shared by every scene and task.

    ``response_rate`` is the controller's angular frequency in rad/s.  With
    the default critical damping ratio, each point uses
    ``kp = mass * response_rate**2`` and
    ``kd = 2 * damping_ratio * mass * response_rate``.
    """

    response_rate: float = 8.0
    damping_ratio: float = 1.0
    robot_force_budget_g: float = 3.0
    object_force_budget_g: float = 5.0

    def __post_init__(self) -> None:
        if self.response_rate <= 0.0:
            raise ValueError("response_rate must be positive")
        if self.damping_ratio < 0.0:
            raise ValueError("damping_ratio must be non-negative")
        if self.robot_force_budget_g <= 0.0:
            raise ValueError("robot_force_budget_g must be positive")
        if self.object_force_budget_g <= 0.0:
            raise ValueError("object_force_budget_g must be positive")


DEFAULT_ASSISTANCE_GAINS = AssistanceGains()


def _quat_error_world(current: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    q_err = quat_mul(target, quat_conjugate(current))
    q_err = torch.where(q_err[..., 0:1] < 0, -q_err, q_err)
    return 2.0 * q_err[..., 1:4]


def _cap_vectors(vectors: torch.Tensor, budget: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Scale a per-env vector set into one L1-of-norms force budget."""
    used = torch.linalg.vector_norm(vectors, dim=-1).sum(-1, keepdim=True)
    scale = torch.minimum(torch.ones_like(used), budget / used.clamp_min(1e-6))
    return vectors * scale[..., None], scale.squeeze(-1)


def robot_relative_object_target(
    robot_pos_w: torch.Tensor,
    robot_quat_w: torch.Tensor,
    mapped_source_robot_pos_w: torch.Tensor,
    mapped_source_robot_quat_w: torch.Tensor,
    source_object_pos_w: torch.Tensor,
    source_object_quat_w: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Map an authored human-object relation onto the current robot pose.

    Returns ``(target_pos_w, target_quat_w, relative_pos_b, relative_quat)``.
    The source robot pose is the morphology-mapped G1 pelvis target, not the
    raw human pelvis.  Consequently, when the robot is exactly on its mapped
    reference, the returned object target is exactly the authored world pose
    (critical for a cube landing on its authored table).
    """
    relative_pos_b = quat_apply_inverse(
        mapped_source_robot_quat_w,
        source_object_pos_w - mapped_source_robot_pos_w,
    )
    relative_quat = quat_mul(
        quat_conjugate(mapped_source_robot_quat_w), source_object_quat_w
    )
    target_pos_w = robot_pos_w + quat_apply(robot_quat_w, relative_pos_b)
    target_quat_w = quat_mul(robot_quat_w, relative_quat)
    return target_pos_w, target_quat_w, relative_pos_b, relative_quat


@dataclass(frozen=True)
class AssistanceSnapshot:
    body_targets_w: torch.Tensor
    body_forces_w: torch.Tensor
    body_torques_w: torch.Tensor
    body_error: torch.Tensor
    saturation: torch.Tensor
    object_target_pos_w: torch.Tensor | None = None
    object_force_w: torch.Tensor | None = None
    object_torque_w: torch.Tensor | None = None


class AssistedMotionController:
    """Mass-scaled PD wrenches around the frozen SONIC rollout."""

    def __init__(
        self,
        env: ManagerBasedRlEnv,
        *,
        command_name: str = "motion",
        robot_name: str = "robot",
        object_name: str | None = None,
        clip_ids: torch.Tensor | None = None,
        gains: AssistanceGains = DEFAULT_ASSISTANCE_GAINS,
    ) -> None:
        self.env = env
        self.command = env.command_manager.get_term(command_name)
        self.robot = env.scene[robot_name]
        self.object = env.scene[object_name] if object_name else None
        self.gains = gains
        self.body_names = tuple(name for name, _ in G1_SMPL_BODY_MAP)
        missing = [name for name in self.body_names if name not in self.robot.body_names]
        if missing:
            raise ValueError(f"G1 assistance bodies missing from robot: {missing}")
        self.body_ids = [self.robot.body_names.index(name) for name in self.body_names]
        self.anchor_idx = self.body_names.index(_ANCHOR_BODY)

        entity_body_ids = self.robot.data.indexing.body_ids
        model_body_ids = entity_body_ids[self.body_ids]
        model = self.robot.data.model
        self.body_mass = model.body_mass[:, model_body_ids].to(env.device)
        self.total_mass = model.body_mass[:, entity_body_ids].sum(-1).to(env.device)
        anchor_model_id = model_body_ids[self.anchor_idx]
        self.anchor_inertia = model.body_inertia[:, anchor_model_id].mean(-1).to(env.device)

        nominal_body_pos = self.robot.data.body_link_pos_w[:, self.body_ids]
        if clip_ids is None:
            source = self.command.motion.smpl_joints_viz
            self.morphology_scale: float | torch.Tensor = infer_morphology_scale(
                source, nominal_body_pos[0], self.body_names
            )
        else:
            if clip_ids.shape != (env.num_envs,):
                raise ValueError(
                    f"clip_ids has shape {clip_ids.shape}, expected "
                    f"({env.num_envs},)"
                )
            # Batched corpus generation keeps the same per-clip morphology
            # estimate as isolated generation.  A global corpus median would
            # make output depend on which other clips happened to share the
            # batch.
            scales = []
            for env_id, clip_id in enumerate(clip_ids.tolist()):
                start = int(self.command.motion.clip_offsets[clip_id].item())
                end = int(self.command.motion.clip_ends[clip_id].item())
                scales.append(
                    infer_morphology_scale(
                        self.command.motion.smpl_joints_viz[start:end],
                        nominal_body_pos[env_id],
                        self.body_names,
                    )
                )
            self.morphology_scale = torch.tensor(
                scales, dtype=nominal_body_pos.dtype, device=env.device
            )

        self.object_mass = None
        self.object_inertia = None
        if self.object is not None:
            ids = self.object.data.indexing.body_ids
            obj_model = self.object.data.model
            self.object_mass = obj_model.body_mass[:, ids].sum(-1).to(env.device)
            self.object_inertia = obj_model.body_inertia[:, ids].mean(-1).sum(-1).to(env.device)

    def targets(self, frame_idx: torch.Tensor) -> torch.Tensor:
        """Return morphology-scaled SMPL landmark targets for source frames."""
        joints = self.command.motion.smpl_joints_viz[frame_idx]
        return scaled_smpl_targets(
            joints, self.morphology_scale, self.env.scene.env_origins
        )

    def apply(self) -> AssistanceSnapshot:
        frame = self.command.time_steps
        targets = self.targets(frame)
        clip_start = self.command.motion.clip_offsets[self.command._clip_ids]
        prev_frame = torch.maximum(frame - 1, clip_start)
        target_vel = (targets - self.targets(prev_frame)) / self.env.step_dt

        current_pos = self.robot.data.body_link_pos_w[:, self.body_ids]
        current_vel = self.robot.data.body_link_lin_vel_w[:, self.body_ids]
        rate = self.gains.response_rate
        damping = self.gains.damping_ratio
        kp = self.body_mass * rate**2
        kd = 2.0 * damping * self.body_mass * rate
        forces = kp[..., None] * (targets - current_pos) + kd[..., None] * (
            target_vel - current_vel
        )

        # The torso is the 6-D global anchor and therefore carries the robot's
        # total mass.  Other landmarks receive translation-only corrections.
        forces[:, self.anchor_idx] *= (
            self.total_mass / self.body_mass[:, self.anchor_idx].clamp_min(1e-6)
        )[:, None]
        budget = (
            self.gains.robot_force_budget_g * _GRAVITY * self.total_mass
        )[:, None]
        forces, force_scale = _cap_vectors(forces, budget)

        torques = torch.zeros_like(forces)
        source_quat = self.command.motion.smpl_root_quat[frame]
        current_quat = self.robot.data.body_link_quat_w[:, self.body_ids[self.anchor_idx]]
        current_ang_vel = self.robot.data.body_link_ang_vel_w[:, self.body_ids[self.anchor_idx]]
        torque = self.anchor_inertia[:, None] * (
            rate**2 * _quat_error_world(current_quat, source_quat)
            - 2.0 * damping * rate * current_ang_vel
        )
        torque_budget = self.total_mass * _GRAVITY * self.morphology_scale
        torque_norm = torch.linalg.vector_norm(torque, dim=-1)
        torque_scale = torch.minimum(
            torch.ones_like(torque_norm), torque_budget / torque_norm.clamp_min(1e-6)
        )
        torques[:, self.anchor_idx] = torque * torque_scale[:, None]
        object_target = object_force = object_torque = None
        object_scale = torch.ones_like(force_scale)
        if self.object is not None:
            object_target, object_force, object_torque, object_scale = (
                self._apply_object(frame)
            )

        self.robot.write_external_wrench_to_sim(
            forces, torques, body_ids=self.body_ids
        )

        return AssistanceSnapshot(
            body_targets_w=targets,
            body_forces_w=forces,
            body_torques_w=torques,
            body_error=torch.linalg.vector_norm(targets - current_pos, dim=-1),
            saturation=torch.minimum(force_scale, torch.minimum(torque_scale, object_scale)),
            object_target_pos_w=object_target,
            object_force_w=object_force,
            object_torque_w=object_torque,
        )

    def _apply_object(
        self, frame: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        assert self.object is not None
        assert self.object_mass is not None and self.object_inertia is not None
        motion = self.command.motion
        clip_start = motion.clip_offsets[self.command._clip_ids]
        previous = torch.maximum(frame - 1, clip_start)
        pelvis = self.body_names.index("pelvis")
        origins = self.env.scene.env_origins

        mapped_robot_pos = self.targets(frame)[:, pelvis]
        mapped_robot_quat = motion.smpl_root_quat[frame]
        source_object_pos = motion.obj_pos[frame] + origins
        source_object_quat = motion.obj_quat[frame]
        object_target, object_target_q, relative_pos, _ = robot_relative_object_target(
            self.robot.data.root_link_pos_w,
            self.robot.data.root_link_quat_w,
            mapped_robot_pos,
            mapped_robot_quat,
            source_object_pos,
            source_object_quat,
        )

        previous_mapped_pos = self.targets(previous)[:, pelvis]
        previous_mapped_quat = motion.smpl_root_quat[previous]
        previous_source_object_pos = motion.obj_pos[previous] + origins
        previous_source_object_quat = motion.obj_quat[previous]
        _, previous_target_q, previous_relative_pos, _ = robot_relative_object_target(
            self.robot.data.root_link_pos_w,
            self.robot.data.root_link_quat_w,
            previous_mapped_pos,
            previous_mapped_quat,
            previous_source_object_pos,
            previous_source_object_quat,
        )

        # Moving-base feedforward.  The relative trajectory is differentiated
        # in the robot frame, then transported through the current robot pose;
        # the lever-arm term accounts for robot angular velocity.
        relative_lin_vel = (relative_pos - previous_relative_pos) / self.env.step_dt
        relative_pos_w = quat_apply(self.robot.data.root_link_quat_w, relative_pos)
        object_target_lin_vel = (
            self.robot.data.root_link_lin_vel_w
            + torch.linalg.cross(
                self.robot.data.root_link_ang_vel_w, relative_pos_w, dim=-1
            )
            + quat_apply(self.robot.data.root_link_quat_w, relative_lin_vel)
        )
        relative_target_delta = quat_mul(
            object_target_q, quat_conjugate(previous_target_q)
        )
        object_target_ang_vel = (
            self.robot.data.root_link_ang_vel_w
            + axis_angle_from_quat(relative_target_delta) / self.env.step_dt
        )

        rate = self.gains.response_rate
        damping = self.gains.damping_ratio
        kp = self.object_mass * rate**2
        kd = 2.0 * damping * self.object_mass * rate
        force = kp[:, None] * (object_target - self.object.data.root_link_pos_w)
        force += kd[:, None] * (
            object_target_lin_vel - self.object.data.root_link_lin_vel_w
        )
        budget = self.gains.object_force_budget_g * _GRAVITY * self.object_mass
        norm = torch.linalg.vector_norm(force, dim=-1)
        force_scale = torch.minimum(torch.ones_like(norm), budget / norm.clamp_min(1e-6))
        force = force * force_scale[:, None]

        ori_error = _quat_error_world(self.object.data.root_link_quat_w, object_target_q)
        torque = self.object_inertia[:, None] * (
            rate**2 * ori_error
            + 2.0 * damping * rate
            * (object_target_ang_vel - self.object.data.root_link_ang_vel_w)
        )
        torque_budget = self.object_mass * _GRAVITY * self.morphology_scale
        torque_norm = torch.linalg.vector_norm(torque, dim=-1)
        torque_scale = torch.minimum(
            torch.ones_like(torque_norm), torque_budget / torque_norm.clamp_min(1e-6)
        )
        torque = torque * torque_scale[:, None]
        self.object.write_external_wrench_to_sim(
            force[:, None, :], torque[:, None, :]
        )
        return object_target, force, torque, torch.minimum(force_scale, torque_scale)

    def clear(self) -> None:
        zeros = torch.zeros(
            self.env.num_envs, len(self.body_ids), 3, device=self.env.device
        )
        self.robot.write_external_wrench_to_sim(
            zeros, zeros, body_ids=self.body_ids
        )
        if self.object is not None:
            object_zeros = torch.zeros(self.env.num_envs, 1, 3, device=self.env.device)
            self.object.write_external_wrench_to_sim(object_zeros, object_zeros)
