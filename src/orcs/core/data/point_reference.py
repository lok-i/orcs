"""Named point-reference geometry shared by seed synthesis and training.

The source skeleton owns points; the robot owns bodies.  A morphology map is
the only bridge between them.  Rewards never need to know whether the source
was SMPL, a marker suit, or another reconstruction format once a command
exposes the resulting world-frame targets.
"""

from __future__ import annotations

import torch

__all__ = [
    "G1_SMPL_BODY_MAP",
    "infer_morphology_scale",
    "scaled_smpl_targets",
]

# Every homologous landmark available on the G1 tracking model.  This is a
# morphology-level map, not a task-specific reward selection: PerLoco and UOLM
# consume the same set.  The integer is the standardized SMPL-24 joint index.
G1_SMPL_BODY_MAP: tuple[tuple[str, int], ...] = (
    ("pelvis", 0),
    ("left_hip_roll_link", 1),
    ("left_knee_link", 4),
    ("left_ankle_roll_link", 10),
    ("right_hip_roll_link", 2),
    ("right_knee_link", 5),
    ("right_ankle_roll_link", 11),
    ("torso_link", 9),
    ("left_shoulder_roll_link", 16),
    ("left_elbow_link", 18),
    ("left_wrist_yaw_link", 20),
    ("right_shoulder_roll_link", 17),
    ("right_elbow_link", 19),
    ("right_wrist_yaw_link", 21),
)

_LEFT_LEG = (0, 1, 4, 10)
_RIGHT_LEG = (0, 2, 5, 11)


def _chain_length(points: torch.Tensor, indices: tuple[int, ...]) -> torch.Tensor:
    chain = points[..., list(indices), :]
    return torch.linalg.vector_norm(
        chain[..., 1:, :] - chain[..., :-1, :], dim=-1
    ).sum(-1)


def infer_morphology_scale(
    smpl_joints_w: torch.Tensor,
    robot_body_pos_w: torch.Tensor,
    robot_body_names: tuple[str, ...] | list[str],
) -> float:
    """Infer one G1/human scale from median left/right leg-chain lengths."""
    name_to_idx = {name: i for i, name in enumerate(robot_body_names)}
    robot_chains = (
        ("pelvis", "left_hip_roll_link", "left_knee_link", "left_ankle_roll_link"),
        ("pelvis", "right_hip_roll_link", "right_knee_link", "right_ankle_roll_link"),
    )
    robot_lengths = []
    for names in robot_chains:
        p = robot_body_pos_w[..., [name_to_idx[name] for name in names], :]
        robot_lengths.append(
            torch.linalg.vector_norm(p[..., 1:, :] - p[..., :-1, :], dim=-1).sum(-1)
        )
    robot_length = torch.stack(robot_lengths).mean()

    human_lengths = torch.stack(
        (
            _chain_length(smpl_joints_w, _LEFT_LEG),
            _chain_length(smpl_joints_w, _RIGHT_LEG),
        ),
        dim=-1,
    ).mean(-1)
    human_length = human_lengths.median()
    if not torch.isfinite(human_length) or human_length <= 1e-6:
        raise ValueError("cannot infer morphology scale from a degenerate SMPL skeleton")
    return float((robot_length / human_length).item())


def scaled_smpl_targets(
    smpl_joints_w: torch.Tensor,
    scale: float | torch.Tensor,
    origins: torch.Tensor,
) -> torch.Tensor:
    """Map every homologous SMPL point to a G1-sized world target.

    Root xy and the lowest source ankle z remain in the authored scene frame;
    body-relative offsets are uniformly scaled around the pelvis.  ``scale``
    may be one scalar or one value per leading batch element.
    """
    pelvis = smpl_joints_w[..., 0, :]
    foot_z = smpl_joints_w[..., (10, 11), 2].amin(dim=-1)
    scale_t = torch.as_tensor(
        scale, dtype=smpl_joints_w.dtype, device=smpl_joints_w.device
    )
    while scale_t.ndim < pelvis.ndim - 1:
        scale_t = scale_t.unsqueeze(-1)

    root = pelvis.clone()
    root[..., 2] = foot_z + scale_t * (pelvis[..., 2] - foot_z)
    mapped = smpl_joints_w[..., [j for _, j in G1_SMPL_BODY_MAP], :]
    targets = root[..., None, :] + scale_t[..., None, None] * (
        mapped - pelvis[..., None, :]
    )
    return targets + origins[..., None, :]
