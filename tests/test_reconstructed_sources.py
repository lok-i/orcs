from __future__ import annotations

import numpy as np

from orcs.assets.uolm_scenes import (
    BIG_CUBE_HALF_EXTENT,
    BIG_CUBE_MASS,
    SMALL_CUBE_HALF_EXTENT,
    SMALL_CUBE_MASS,
    TABLE_CENTER_HEIGHT,
    TABLE_SIZE,
)
from orcs.tasks.uolm.sources.reconstructed import (
    MOTION_SETS,
    _ground_human_to_floor,
    _timeline,
    is_current_cache_sample,
)


def test_reconstructed_motion_sets_match_vibe_scenes() -> None:
    small = MOTION_SETS["small-cube-table"]
    big = MOTION_SETS["big-cube-floor"]

    assert small.source_root == "drcl/Box"
    assert small.support == "table"
    assert small.object_half_extent == SMALL_CUBE_HALF_EXTENT == 0.18
    assert SMALL_CUBE_MASS == 0.3
    assert TABLE_SIZE == (0.75, 0.75, 0.05)
    assert TABLE_CENTER_HEIGHT == 1.0

    assert big.source_root == "drcl/LargeCube"
    assert big.support == "floor"
    assert big.object_half_extent == BIG_CUBE_HALF_EXTENT == 0.3048
    assert BIG_CUBE_MASS == 1.5


def test_resampling_grid_is_exact_and_does_not_extrapolate() -> None:
    source_t, target_t = _timeline(num_frames=135, source_fps=30.0)

    np.testing.assert_allclose(np.diff(source_t), 1.0 / 30.0)
    np.testing.assert_allclose(np.diff(target_t), 1.0 / 50.0)
    assert target_t[-1] <= source_t[-1]
    assert source_t[-1] - target_t[-1] < 1.0 / 50.0


def test_human_grounding_removes_only_clip_wide_vertical_bias() -> None:
    joints = np.zeros((20, 24, 3), dtype=np.float32)
    joints[..., 2] = 1.2
    joints[:, 10, 2] = 0.31
    joints[:, 11, 2] = 0.33
    joints[0, 10, 2] = -3.0  # one reconstruction outlier must not own the floor

    grounded, offset = _ground_human_to_floor(joints)

    floor = np.minimum(grounded[:, 10, 2], grounded[:, 11, 2])
    assert np.isclose(np.quantile(floor, 0.05), 0.0, atol=1e-6)
    np.testing.assert_allclose(grounded[..., :2], joints[..., :2])
    np.testing.assert_allclose(grounded[..., 2], joints[..., 2] - offset)


def test_current_cache_requires_grounding_schema(tmp_path) -> None:
    np.savez(tmp_path / "smpl_motion.npz", human_ground_offset_z=np.array(0.2))
    np.savez(tmp_path / "object_motion.npz", obj_pos_w=np.zeros((1, 3)))
    (tmp_path / "metadata.json").write_text('{"schema_version": 1}\n')
    assert not is_current_cache_sample(tmp_path)

    (tmp_path / "metadata.json").write_text('{"schema_version": 2}\n')
    assert is_current_cache_sample(tmp_path)
