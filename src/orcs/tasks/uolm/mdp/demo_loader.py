"""Object-keyed demo discovery — the one thing about the scan that is uolm's.

The walk, the exclusion grammar and the scan report are task-blind and live in
:mod:`orcs.core.data.scan`; they are re-exported here so the uolm mdp surface
is unchanged. What this file owns is the KEY: which object a clip belongs to,
read from each sample's ``metadata.json`` (`object_path` -> its parent dir).

No mjlab / sim dependency — safe to run standalone for testing.
"""

from __future__ import annotations

import json
from pathlib import Path

from orcs.core.data.scan import (  # noqa: F401 — uolm's public scan surface
    last_scan,
    load_field_or_make_zeros,
    matches_exclude,
    motion_dirs,
    scan_flat,
    scan_grouped,
)

__all__ = [
    "last_scan", "matches_exclude", "load_field_or_make_zeros", "motion_dirs",
    "scan_flat", "load_motion_files_from_datasets", "get_motion_files_for_objects",
]


def _object_of(sample_dir: Path) -> str | None:
    """The object a clip belongs to: metadata.json `object_path`'s parent dir.

    None when there is no metadata.json — the caller buckets those as "N/A"
    and records them in ``last_scan["no_metadata"]``.
    """
    metadata_file = sample_dir / "metadata.json"
    if not metadata_file.exists():
        return None
    with open(metadata_file) as f:
        object_path = json.load(f).get("object_path", "N/A")
    return object_path.split("/")[-2] if object_path != "N/A" else "N/A"


def load_motion_files_from_datasets(
    path_to_datasets: str,
    exclude_motions: list[str] | None = None,
) -> dict[str, list[str]]:
    """Motion files under a robot-level dataset root, keyed by object name.

    Expects ``<path_to_datasets>/<dataset>/<motion>/<sampleX>/motion.npz``. All
    immediate subdirs are treated as datasets (e.g. ``omomo/``, ``physhoi/``).

    Args:
        path_to_datasets: Robot-level root, e.g.
            ``/path/to/retargeted_motions/data/unitree_g1/``.
        exclude_motions: Motions/clips to skip — grammar in
            :func:`orcs.core.data.scan.matches_exclude`.

    Returns:
        dict mapping object_name -> motion.npz paths (ordered by dataset name,
        then motion name, then sample number).
    """
    return scan_grouped(path_to_datasets, _object_of, exclude_motions)


def get_motion_files_for_objects(
    ordered_object_names: list[str],
    path_to_datasets: str,
    exclude_motions: list[str] | None = None,
) -> tuple[dict[str, list[str]], list[str]]:
    """(files per requested object, files concatenated in roster order).

    Roster order IS the object-id space, so the flat list's clip index maps to
    an object by a repeat_interleave of the per-object counts.
    """
    motion_files_by_object = load_motion_files_from_datasets(
        path_to_datasets, exclude_motions=exclude_motions
    )

    motion_files_for_given_objects = {}
    motion_files_object_ordered: list[str] = []
    for object_name in ordered_object_names:
        if object_name not in motion_files_by_object:
            raise ValueError(
                f"No motion files found for object '{object_name}' under "
                f"path_to_datasets={path_to_datasets}. Check your datasets or assets."
            )
        motion_files_object_ordered.extend(motion_files_by_object[object_name])
        motion_files_for_given_objects[object_name] = motion_files_by_object[object_name]

    return motion_files_for_given_objects, motion_files_object_ordered
