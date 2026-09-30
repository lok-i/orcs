"""Metadata-only roster selection for the retargeted-motion dataset.

Setup uses the same object and exclusion roster as the UOLM runtime. Keeping
the resolver below orcs.core lets a pointer-only checkout select its LFS
objects without importing MJLab or registering tasks.
"""

from __future__ import annotations

import json
from pathlib import Path

from orcs.core.data.scan import scan_grouped

__all__ = [
    "DEFAULT_OBJECT_NAMES",
    "EXCLUDE_MOTIONS",
    "get_motion_files_for_objects",
    "lfs_include",
    "load_motion_files_from_datasets",
]

# fcrl's default roster. Order is the UOLM object-id space.
DEFAULT_OBJECT_NAMES = (
    "suitcase",
    "trashcan",
    "largetable",
    "plasticbox",
    "tire",
    "woodchair2",
)

EXCLUDE_MOTIONS = (
    "sub5_suitcase_015",
    "woodchair2_sit",
    # Prefer custom/tire_roll/sample1 over sugar's three faster, longer takes.
    "tire_flip",
    "sugar/tire_roll",
    "custom/woodchair2_flip/sample1",
    "custom/woodchair2_flip/sample3",
)


def _object_of(sample_dir: Path) -> str | None:
    """Object key from metadata.json; None when metadata is absent."""
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
    """Motion files under a robot-level dataset root, keyed by object name."""
    return scan_grouped(path_to_datasets, _object_of, exclude_motions)


def get_motion_files_for_objects(
    ordered_object_names: list[str],
    path_to_datasets: str,
    exclude_motions: list[str] | None = None,
) -> tuple[dict[str, list[str]], list[str]]:
    """Files per requested object and the same files in roster order."""
    motion_files_by_object = load_motion_files_from_datasets(
        path_to_datasets, exclude_motions=exclude_motions
    )

    selected: dict[str, list[str]] = {}
    ordered: list[str] = []
    for object_name in ordered_object_names:
        if object_name not in motion_files_by_object:
            raise ValueError(
                f"No motion files found for object '{object_name}' under "
                f"path_to_datasets={path_to_datasets}. Check your datasets or assets."
            )
        ordered.extend(motion_files_by_object[object_name])
        selected[object_name] = motion_files_by_object[object_name]
    return selected, ordered


def lfs_include(checkout: str | Path) -> list[str]:
    """LFS globs for every NPZ consumed by native non-SMPL ORCS tasks.

    metadata.json is not in LFS, so this works while every NPZ is still a
    pointer. Each selected sample directory is one glob because runtimes may
    consume companion NPZ files in addition to motion.npz.
    """
    root = Path(checkout)
    _, files = get_motion_files_for_objects(
        list(DEFAULT_OBJECT_NAMES),
        str(root / "data/unitree_g1"),
        list(EXCLUDE_MOTIONS),
    )
    return sorted(
        {
            f"{Path(path).parent.relative_to(root).as_posix()}/*.npz"
            for path in files
        }
    )
