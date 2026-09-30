"""The lean LFS selection is the same roster the UOLM runtime consumes."""

from __future__ import annotations

import json
from pathlib import Path

from orcs.core.data.retargeted import DEFAULT_OBJECT_NAMES, lfs_include


def test_lfs_include_is_derived_from_native_object_roster(tmp_path: Path) -> None:
    root = tmp_path / "retargeted_motions"
    robot = root / "data/unitree_g1"
    expected: list[str] = []
    for index, object_name in enumerate(DEFAULT_OBJECT_NAMES):
        sample = robot / "fixture" / f"motion_{index}" / "sample1"
        sample.mkdir(parents=True)
        (sample / "metadata.json").write_text(
            json.dumps({"object_path": f"objects/{object_name}/model.xml"})
        )
        (sample / "motion.npz").write_text("git-lfs pointer")
        expected.append(
            f"data/unitree_g1/fixture/motion_{index}/sample1/*.npz"
        )

    assert lfs_include(root) == sorted(expected)
