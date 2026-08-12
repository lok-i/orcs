from __future__ import annotations

import numpy as np

from orcs.core.data.seeds import SEED_POSITION_FRAME, SeedMotion


def test_seed_motion_round_trip(tmp_path):
    t, j, b = 3, 2, 2
    zeros3 = np.zeros((t, 3), dtype=np.float32)
    root_quat = np.zeros((t, 4), dtype=np.float32)
    root_quat[:, 0] = 1.0
    body_quat = np.zeros((t, b, 4), dtype=np.float32)
    body_quat[..., 0] = 1.0
    seed = SeedMotion(
        fps=50.0,
        source_frame_idx=np.arange(t),
        robot_root_pos_w=zeros3,
        robot_root_quat_w=root_quat,
        robot_root_lin_vel_w=zeros3,
        robot_root_ang_vel_w=zeros3,
        joint_pos=np.zeros((t, j), dtype=np.float32),
        joint_vel=np.zeros((t, j), dtype=np.float32),
        last_action=np.zeros((t, j), dtype=np.float32),
        body_pos_w=np.zeros((t, b, 3), dtype=np.float32),
        body_quat_w=body_quat,
        assist_force_w=np.zeros((t, b, 3), dtype=np.float32),
        assist_torque_w=np.zeros((t, b, 3), dtype=np.float32),
        body_tracking_error=np.zeros((t, b), dtype=np.float32),
        assist_saturation=np.ones(t, dtype=np.float32),
        valid=np.ones(t, dtype=bool),
        joint_names=("j0", "j1"),
        body_names=("b0", "b1"),
        source_path="clip/smpl_motion.npz",
        capture_steps=12,
        morphology_scale=0.75,
    )

    path = seed.save(tmp_path / "seed_state.npz")
    loaded = SeedMotion.load(path)

    assert loaded.joint_names == seed.joint_names
    assert loaded.body_names == seed.body_names
    assert loaded.source_path == seed.source_path
    assert loaded.capture_steps == 12
    assert np.allclose(loaded.source_time_s, [0.0, 0.02, 0.04])
    assert np.array_equal(loaded.valid, seed.valid)
    with np.load(path) as raw:
        assert str(raw["position_frame"]) == SEED_POSITION_FRAME == "env_local"


def test_seed_motion_rejects_shifted_source_index():
    t = 2
    zeros3 = np.zeros((t, 3), dtype=np.float32)
    quat = np.zeros((t, 4), dtype=np.float32)
    quat[:, 0] = 1.0
    body_quat = quat[:, None]
    try:
        SeedMotion(
            fps=50.0,
            source_frame_idx=np.array([1, 2]),
            robot_root_pos_w=zeros3,
            robot_root_quat_w=quat,
            robot_root_lin_vel_w=zeros3,
            robot_root_ang_vel_w=zeros3,
            joint_pos=np.zeros((t, 1)),
            joint_vel=np.zeros((t, 1)),
            last_action=np.zeros((t, 1)),
            body_pos_w=np.zeros((t, 1, 3)),
            body_quat_w=body_quat,
            assist_force_w=np.zeros((t, 1, 3)),
            assist_torque_w=np.zeros((t, 1, 3)),
            body_tracking_error=np.zeros((t, 1)),
            assist_saturation=np.ones(t),
            valid=np.ones(t, dtype=bool),
            joint_names=("j0",),
            body_names=("b0",),
        )
    except ValueError as exc:
        assert "0..T-1" in str(exc)
    else:
        raise AssertionError("shifted source index was accepted")


def test_seed_motion_rejects_nonfinite_object_state():
    t = 1
    zeros3 = np.zeros((t, 3), dtype=np.float32)
    quat = np.zeros((t, 4), dtype=np.float32)
    quat[:, 0] = 1.0
    bad_object_pos = zeros3.copy()
    bad_object_pos[0, 0] = np.nan
    try:
        SeedMotion(
            fps=50.0,
            source_frame_idx=np.arange(t),
            robot_root_pos_w=zeros3,
            robot_root_quat_w=quat,
            robot_root_lin_vel_w=zeros3,
            robot_root_ang_vel_w=zeros3,
            joint_pos=np.zeros((t, 1)),
            joint_vel=np.zeros((t, 1)),
            last_action=np.zeros((t, 1)),
            body_pos_w=np.zeros((t, 1, 3)),
            body_quat_w=quat[:, None],
            assist_force_w=np.zeros((t, 1, 3)),
            assist_torque_w=np.zeros((t, 1, 3)),
            body_tracking_error=np.zeros((t, 1)),
            assist_saturation=np.ones(t),
            valid=np.ones(t, dtype=bool),
            joint_names=("j0",),
            body_names=("b0",),
            object_pos_w=bad_object_pos,
            object_quat_w=quat,
            object_lin_vel_w=zeros3,
            object_ang_vel_w=zeros3,
            object_target_pos_w=zeros3,
            object_assist_force_w=zeros3,
            object_assist_torque_w=zeros3,
        )
    except ValueError as exc:
        assert "object_pos_w contains NaN/Inf" in str(exc)
    else:
        raise AssertionError("non-finite object state was accepted")
