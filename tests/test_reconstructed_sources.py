from __future__ import annotations

import numpy as np

from orcs.assets.uolm_scenes import (
    BIG_CUBE_HALF_EXTENT,
    BIG_CUBE_MASS,
    SMALL_CUBE_HALF_EXTENT,
    SMALL_CUBE_MASS,
    TABLE_CENTER_HEIGHT,
    TABLE_SIZE,
    WOODCHAIR2_MESH_SCALE,
)
from orcs.tasks.uolm.sources.reconstructed import (
    DEFAULT_MOTION_SETS,
    MOTION_SETS,
    _cache_sample,
    _ground_human_to_floor,
    _timeline,
    cache_motion_files,
    is_current_cache_sample,
    source_clips,
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

    chair = MOTION_SETS["woodchair2-floor"]
    tire = MOTION_SETS["tire-floor"]
    assert DEFAULT_MOTION_SETS == ("woodchair2-floor", "tire-floor")
    assert chair.source_root == "drcl/Chair"
    assert chair.interaction == "chair_flip"
    assert chair.include_clips == ("Data01_Sub01_chair_flip_cam0",)
    assert chair.object_scale == WOODCHAIR2_MESH_SCALE
    assert chair.support == "floor"
    assert tire.source_root == "drcl/Tire"
    assert tire.interaction == "tire_roll"
    assert tire.include_clips == ("Data01_Sub02_tire_roll_cam0",)
    assert tire.support == "floor"


def test_reconstructed_base_roster_is_exact(tmp_path, monkeypatch) -> None:
    from orcs.tasks.uolm.sources import reconstructed

    chair_names = MOTION_SETS["woodchair2-floor"].include_clips
    assert chair_names is not None
    for subject in ("Sub01", "Sub02", "Sub03", "Sub04"):
        name = f"Data01_{subject}_chair_flip_cam0"
        path = tmp_path / "drcl/Chair" / name / "motion.npz"
        path.parent.mkdir(parents=True)
        path.touch()
    for subject in ("Sub01", "Sub02", "Sub03"):
        path = (
            tmp_path
            / "drcl/Tire"
            / f"Data01_{subject}_tire_roll_cam0"
            / "motion.npz"
        )
        path.parent.mkdir(parents=True)
        path.touch()

    monkeypatch.setattr(reconstructed, "source_root", lambda: tmp_path)
    monkeypatch.setattr(reconstructed, "cache_root", lambda: tmp_path / "cache")

    chairs = source_clips("woodchair2-floor")
    tires = source_clips("tire-floor")
    assert tuple(path.name for path in chairs) == chair_names
    assert [path.name for path in tires] == ["Data01_Sub02_tire_roll_cam0"]
    assert _cache_sample(chairs[0], "woodchair2-floor") == (
        tmp_path / "cache/woodchair2-floor/chair_flip" / chair_names[0]
    )
    assert _cache_sample(tires[0], "tire-floor") == (
        tmp_path
        / "cache/tire-floor/tire_roll/Data01_Sub02_tire_roll_cam0"
    )


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
    (tmp_path / "metadata.json").write_text(
        '{"schema_version": 1, "motion_set": "small-cube-table"}\n'
    )
    assert not is_current_cache_sample(tmp_path)

    # A legacy Cube sample has no per-motion-set version and remains current.
    (tmp_path / "metadata.json").write_text(
        '{"schema_version": 2, "motion_set": "small-cube-table"}\n'
    )
    assert is_current_cache_sample(tmp_path)


def test_cache_motion_files_selects_interaction_directories(
    tmp_path, monkeypatch
) -> None:
    from orcs.tasks.uolm.sources import reconstructed

    root = tmp_path / "small-cube-table"
    carry = root / "carryflip" / "carry-01" / "smpl_motion.npz"
    throw = root / "throw" / "throw-01" / "smpl_motion.npz"
    for path in (carry, throw):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.touch()
    monkeypatch.setattr(reconstructed, "cache_root", lambda: tmp_path)

    assert cache_motion_files("small-cube-table") == [carry, throw]
    assert cache_motion_files("small-cube-table", ("carryflip",)) == [carry]
    assert cache_motion_files("small-cube-table", ("throw",)) == [throw]


def test_cache_motion_files_applies_exact_base_roster(tmp_path) -> None:
    chair_root = tmp_path / "woodchair2-floor" / "chair_flip"
    chair_selected = (
        chair_root / "Data01_Sub01_chair_flip_cam0" / "smpl_motion.npz"
    )
    chair_excluded = (
        chair_root / "Data01_Sub02_chair_flip_cam0" / "smpl_motion.npz"
    )
    tire_root = tmp_path / "tire-floor" / "tire_roll"
    tire_selected = tire_root / "Data01_Sub02_tire_roll_cam0" / "smpl_motion.npz"
    tire_excluded = tire_root / "Data01_Sub01_tire_roll_cam0" / "smpl_motion.npz"
    for path in (chair_selected, chair_excluded, tire_selected, tire_excluded):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.touch()

    assert cache_motion_files("woodchair2-floor", root=tmp_path) == [
        chair_selected
    ]
    assert cache_motion_files("tire-floor", root=tmp_path) == [tire_selected]
