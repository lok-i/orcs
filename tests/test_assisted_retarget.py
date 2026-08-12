from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import torch

import orcs.cli.pseudo_retarget as pseudo_retarget
from orcs.cli.pseudo_retarget import (
    _seed_is_complete,
    _selected_grail_samples,
    _SettlingMonitor,
)
from orcs.core.assisted_retarget import (
    DEFAULT_ASSISTANCE_GAINS,
    G1_SMPL_BODY_MAP,
    AssistanceGains,
    infer_morphology_scale,
    robot_relative_object_target,
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


def test_object_target_preserves_authored_pose_on_reference():
    robot_pos = torch.tensor([[1.0, 2.0, 0.8]])
    robot_quat = torch.tensor([[1.0, 0.0, 0.0, 0.0]])
    object_pos = torch.tensor([[1.4, 2.2, 1.1]])
    object_quat = torch.tensor([[0.7071068, 0.0, 0.0, 0.7071068]])

    target_pos, target_quat, _, _ = robot_relative_object_target(
        robot_pos, robot_quat, robot_pos, robot_quat, object_pos, object_quat
    )

    assert torch.allclose(target_pos, object_pos)
    assert torch.allclose(target_quat, object_quat)


def test_object_target_is_equivariant_to_robot_motion():
    source_robot_pos = torch.tensor([[0.0, 0.0, 0.8]])
    identity = torch.tensor([[1.0, 0.0, 0.0, 0.0]])
    source_object_pos = torch.tensor([[1.0, 0.0, 1.0]])
    source_object_quat = identity.clone()
    yaw_90 = torch.tensor([[0.7071068, 0.0, 0.0, 0.7071068]])

    target_pos, target_quat, _, _ = robot_relative_object_target(
        torch.tensor([[2.0, 3.0, 0.8]]),
        yaw_90,
        source_robot_pos,
        identity,
        source_object_pos,
        source_object_quat,
    )

    assert torch.allclose(target_pos, torch.tensor([[2.0, 4.0, 1.0]]), atol=1e-6)
    assert torch.allclose(target_quat, yaw_90, atol=1e-6)


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


def test_grail_batch_selection_follows_roster(monkeypatch, tmp_path):
    root = tmp_path / "terrain_motions/grail"
    tile = root / "curb_000/level_0.00"
    for name in ("sample0", "sample1"):
        sample = tile / name
        sample.mkdir(parents=True)
        np.savez(sample / "motion.npz", joint_pos=np.zeros((2, 1)))
        np.savez(sample / "smpl_motion.npz", smpl_joints=np.zeros((2, 24, 3)))

    import orcs.tasks.perloco.roster as roster_module

    monkeypatch.setattr(pseudo_retarget, "DATA_ROOT", tmp_path)
    monkeypatch.setattr(
        roster_module,
        "load_roster",
        lambda *_args, **_kwargs: SimpleNamespace(
            tile_keys=("curb_000/level_0.00",),
            clips={"curb_000/level_0.00": ("sample1",)},
        ),
    )

    assert _selected_grail_samples() == [tile / "sample1"]


def test_complete_seed_requires_valid_source_aligned_frames(monkeypatch, tmp_path):
    sample = tmp_path / "sample0"
    sample.mkdir()
    np.savez(sample / "smpl_motion.npz", smpl_joints=np.zeros((3, 24, 3)))
    (sample / "seed_state.npz").touch()

    monkeypatch.setattr(
        pseudo_retarget,
        "SeedMotion",
        SimpleNamespace(
            load=lambda _path: SimpleNamespace(
                num_frames=3, valid=np.ones(3, dtype=bool)
            )
        ),
    )
    assert _seed_is_complete(sample)

    pseudo_retarget.SeedMotion.load = lambda _path: SimpleNamespace(
        num_frames=2, valid=np.ones(2, dtype=bool)
    )
    assert not _seed_is_complete(sample)


def test_grail_batch_uses_one_vectorized_rollout_without_subprocesses(
    monkeypatch, tmp_path
):
    samples = [tmp_path / "sample0", tmp_path / "sample1"]
    calls: list[tuple[list, str]] = []

    monkeypatch.setattr(pseudo_retarget, "_selected_grail_samples", lambda: samples)
    monkeypatch.setattr(pseudo_retarget, "_seed_is_complete", lambda _sample: False)
    monkeypatch.setattr(
        pseudo_retarget,
        "_run_grail_vectorized",
        lambda pending, *, device: calls.append((pending, device)) or [],
    )
    monkeypatch.setattr(
        pseudo_retarget.subprocess,
        "run",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            AssertionError("successful corpus batch must not spawn workers")
        ),
    )

    pseudo_retarget._run_grail_batch(device="cuda:0", overwrite=False)

    assert calls == [(samples, "cuda:0")]


def test_uolm_batch_uses_one_vectorized_rollout_without_subprocesses(
    monkeypatch, tmp_path
):
    samples = [tmp_path / "sample0", tmp_path / "sample1"]
    calls: list[tuple[list, str]] = []

    import orcs.tasks.uolm.sources.reconstructed as reconstructed

    monkeypatch.setattr(
        reconstructed, "stage_motion_set", lambda *_args, **_kwargs: samples
    )
    monkeypatch.setattr(
        pseudo_retarget, "_object_seed_is_complete", lambda _sample: False
    )
    monkeypatch.setattr(
        pseudo_retarget,
        "_run_uolm_vectorized",
        lambda pending, *, device: calls.append((pending, device)) or [],
    )
    monkeypatch.setattr(
        pseudo_retarget.subprocess,
        "run",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            AssertionError("successful corpus batch must not spawn workers")
        ),
    )

    pseudo_retarget._run_reconstructed_batch(
        motion_sets=("big-cube-floor",), device="cuda:0", overwrite=False
    )

    assert calls == [(samples, "cuda:0")]
