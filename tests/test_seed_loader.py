from __future__ import annotations

import numpy as np
import pytest

from orcs.core.data.seed_loader import SeededSmplMotionLoader
from orcs.core.data.seeds import SeedMotion


def _write_sample(tmp_path, *, valid: np.ndarray | None = None):
    sample = tmp_path / "tile" / "sample0"
    sample.mkdir(parents=True)
    # Deliberately not a valid motion NPZ: the seed loader may use this path to
    # locate the sample but must never consume the upstream robot retarget.
    motion_path = sample / "motion.npz"
    motion_path.write_text("not a teacher trajectory")

    t = 3
    root = np.zeros((t, 3), dtype=np.float32)
    root[:, 0] = np.arange(t)
    quat = np.zeros((t, 4), dtype=np.float32)
    quat[:, 0] = 1.0
    body_pos = np.zeros((t, 2, 3), dtype=np.float32)
    body_pos[:, 1] = root  # seed order is hand, pelvis
    body_quat = np.repeat(quat[:, None], 2, axis=1)
    seed = SeedMotion(
        fps=50.0,
        source_frame_idx=np.arange(t),
        robot_root_pos_w=root,
        robot_root_quat_w=quat,
        robot_root_lin_vel_w=np.ones((t, 3), dtype=np.float32),
        robot_root_ang_vel_w=np.zeros((t, 3), dtype=np.float32),
        joint_pos=np.array([[10, 20], [11, 21], [12, 22]], dtype=np.float32),
        joint_vel=np.zeros((t, 2), dtype=np.float32),
        last_action=np.array([[30, 40], [31, 41], [32, 42]], dtype=np.float32),
        body_pos_w=body_pos,
        body_quat_w=body_quat,
        assist_force_w=np.zeros((t, 2, 3), dtype=np.float32),
        assist_torque_w=np.zeros((t, 2, 3), dtype=np.float32),
        body_tracking_error=np.zeros((t, 2), dtype=np.float32),
        assist_saturation=np.ones(t, dtype=np.float32),
        valid=np.ones(t, dtype=bool) if valid is None else valid,
        joint_names=("j1", "j0"),
        body_names=("hand", "pelvis"),
        morphology_scale=0.75,
    )
    seed.save(sample / "seed_state.npz")

    joints = np.zeros((t, 24, 3), dtype=np.float32)
    joints[:, 0, 0] = np.arange(t)
    np.savez(
        sample / "smpl_motion.npz",
        smpl_joints=joints,
        smpl_root_quat_w=quat,
        smpl_joints_viz_w=joints,
    )
    return motion_path


def test_seed_loader_ignores_retarget_and_permutes_named_state(tmp_path):
    motion_path = _write_sample(tmp_path)
    loader = SeededSmplMotionLoader(
        str(tmp_path),
        "cpu",
        motion_files=[str(motion_path)],
        joint_names=("j0", "j1"),
        body_names=("pelvis", "hand"),
        expected_fps=50.0,
    )

    assert loader.joint_pos[0].tolist() == [20.0, 10.0]
    assert loader.last_action[0].tolist() == [40.0, 30.0]
    assert loader.body_pos_w[2, 0, 0].item() == 2.0
    assert loader.morphology_scale.tolist() == [0.75, 0.75, 0.75]


def test_seed_loader_rejects_invalid_rsi_frames(tmp_path):
    motion_path = _write_sample(
        tmp_path, valid=np.array([True, False, True], dtype=bool)
    )
    with pytest.raises(ValueError, match="invalid RSI frames"):
        SeededSmplMotionLoader(
            str(tmp_path),
            "cpu",
            motion_files=[str(motion_path)],
            joint_names=("j0", "j1"),
            body_names=("pelvis", "hand"),
            expected_fps=50.0,
        )
