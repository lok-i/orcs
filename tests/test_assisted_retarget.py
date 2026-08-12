from __future__ import annotations

import torch

from orcs.cli.pseudo_retarget import _SettlingMonitor
from orcs.core.assisted_retarget import (
    DEFAULT_ASSISTANCE_GAINS,
    G1_SMPL_BODY_MAP,
    AssistanceGains,
    infer_morphology_scale,
    scaled_smpl_targets,
)


def _skeleton() -> torch.Tensor:
    joints = torch.zeros(4, 24, 3)
    joints[:, 0] = torch.tensor([0.0, 0.0, 1.0])
    joints[:, 1] = torch.tensor([0.0, 0.1, 0.9])
    joints[:, 4] = torch.tensor([0.0, 0.1, 0.5])
    joints[:, 10] = torch.tensor([0.1, 0.1, 0.0])
    joints[:, 2] = torch.tensor([0.0, -0.1, 0.9])
    joints[:, 5] = torch.tensor([0.0, -0.1, 0.5])
    joints[:, 11] = torch.tensor([0.1, -0.1, 0.0])
    joints[:, 9] = torch.tensor([0.0, 0.0, 1.3])
    joints[:, 16] = torch.tensor([0.0, 0.25, 1.3])
    joints[:, 18] = torch.tensor([0.0, 0.45, 1.1])
    joints[:, 20] = torch.tensor([0.0, 0.65, 0.9])
    joints[:, 17] = torch.tensor([0.0, -0.25, 1.3])
    joints[:, 19] = torch.tensor([0.0, -0.45, 1.1])
    joints[:, 21] = torch.tensor([0.0, -0.65, 0.9])
    return joints


def test_morphology_scale_uses_leg_chain_lengths():
    smpl = _skeleton()
    mapped = smpl[0, [joint for _, joint in G1_SMPL_BODY_MAP]]
    robot = 0.75 * mapped
    scale = infer_morphology_scale(
        smpl, robot, tuple(name for name, _ in G1_SMPL_BODY_MAP)
    )
    assert abs(scale - 0.75) < 1e-6


def test_scaled_targets_preserve_root_xy_and_foot_height():
    smpl = _skeleton()[:1]
    smpl[:, :, :2] += torch.tensor([2.0, 3.0])
    origin = torch.tensor([[10.0, 20.0, 0.0]])
    targets = scaled_smpl_targets(smpl, 0.5, origin)

    pelvis_idx = [name for name, _ in G1_SMPL_BODY_MAP].index("pelvis")
    left_foot_idx = [name for name, _ in G1_SMPL_BODY_MAP].index(
        "left_ankle_roll_link"
    )
    assert torch.allclose(targets[0, pelvis_idx, :2], torch.tensor([12.0, 23.0]))
    assert torch.allclose(targets[0, pelvis_idx, 2], torch.tensor(0.5))
    assert torch.allclose(targets[0, left_foot_idx, 2], torch.tensor(0.0))


def test_scaled_targets_accept_one_morphology_scale_per_environment():
    smpl = _skeleton()[:2]
    targets = scaled_smpl_targets(
        smpl, torch.tensor([0.5, 0.75]), torch.zeros(2, 3)
    )
    pelvis_idx = [name for name, _ in G1_SMPL_BODY_MAP].index("pelvis")
    assert torch.allclose(targets[:, pelvis_idx, 2], torch.tensor([0.5, 0.75]))


def test_global_assistance_gains_preserve_force_only_defaults():
    assert DEFAULT_ASSISTANCE_GAINS == AssistanceGains(
        response_rate=8.0,
        damping_ratio=1.0,
        robot_force_budget_g=3.0,
        object_force_budget_g=5.0,
    )


def test_settling_monitor_accepts_only_a_low_error_plateau():
    high_error = _SettlingMonitor()
    for _ in range(100):
        high_error.update(0.25)
    assert not high_error.converged

    settling = _SettlingMonitor()
    for step in range(1, 130):
        error = max(0.15, 0.30 - 0.01 * step)
        settling.update(error)
    assert settling.converged
