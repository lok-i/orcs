"""Pure-Python helpers for loading object-manipulation demo data.

No mjlab / sim dependency — safe to run standalone for testing.
Ported 1:1 from fcrl/tasks/uolm/g1/demo_loader.py for consistency.
"""

import json
from pathlib import Path

import numpy as np
import torch

last_scan: dict[str, list[str]] = {
    "excluded_motions": [], "excluded_clips": [], "no_metadata": []
}
"""Report from the most recent dataset scan — motions dropped whole, individual
clips dropped, and clips missing metadata.json. Inspect after building an env;
see the note in `load_motion_files_from_datasets`."""


def matches_exclude(
    exclude_set: set[str],
    dataset: str,
    motion: str,
    sample: str | None = None,
) -> bool:
    """Does an `exclude_motions` entry cover this motion folder / clip?

    Vocabulary: a MOTION is one demo folder (`<dataset>/<motion>/`); a CLIP is
    one take of it (`<sampleN>/motion.npz`) — one motion holds 1..200+ clips.

    Entry grammar (a trailing `sample*` component is what makes it clip-level):

    | entry | drops |
    |---|---|
    | `<motion>` | every clip of that motion, in any dataset |
    | `<dataset>/<motion>` | ... in that dataset only |
    | `<motion>/<sampleN>` | that one clip, in any dataset |
    | `<dataset>/<motion>/<sampleN>` | that one clip, that dataset |

    `sample=None` asks the motion-level question only, so a folder holding an
    excluded CLIP is still walked.
    """
    if motion in exclude_set or f"{dataset}/{motion}" in exclude_set:
        return True
    if sample is None:
        return False
    return (f"{motion}/{sample}" in exclude_set
            or f"{dataset}/{motion}/{sample}" in exclude_set)


def load_field_or_make_zeros(
    data: np.lib.npyio.NpzFile | None,
    field_name: str,
    shape: tuple[int, ...],
    device: str,
    verbose: bool = False,
) -> tuple[torch.Tensor, bool]:
    if data is not None and field_name in data:
        data_tensor = torch.tensor(data[field_name], device=device)
        if verbose:
            print(f"[INFO]: Field {field_name} found in data. Loading data with shape {data_tensor.shape}.")
        return data_tensor, True
    else:
        data_tensor = torch.zeros(shape, device=device)
        if verbose:
            print(f"[INFO]: Field {field_name} not found in data. Using zeros with shape {data_tensor.shape}.")
        return data_tensor, False


def load_motion_files_from_datasets(
    path_to_datasets: str,
    exclude_motions: list[str] | None = None,
) -> dict[str, list[str]]:
    """Load motion files from a robot-level dataset root, keyed by object name.

    Expects the structure ``<path_to_datasets>/<dataset>/<motion>/<sampleX>/motion.npz``.
    All immediate subdirs of ``path_to_datasets`` are treated as individual
    datasets (e.g. ``omomo/``, ``physhoi/``), and each dataset is walked
    recursively for motion folders containing sample dirs.

    Args:
        path_to_datasets: Root directory for a robot, e.g.
            ``/path/to/retargeted_motions/data/unitree_g1/``.
        exclude_motions: Optional list of motions/clips to skip — grammar in
            :func:`matches_exclude` (whole motion, or one ``sampleN`` of it,
            each optionally dataset-qualified).

    Returns:
        dict mapping object_name -> list of motion.npz paths (sorted by
        motion dir then sample number).
    """
    exclude_set = set(exclude_motions) if exclude_motions else set()
    motion_files_by_object: dict[str, list[str]] = {}

    excl_motions: list[str] = []  # reported by the caller, not printed here:
    excl_clips: list[str] = []    # this runs at IMPORT time (cfg construction)
    no_metadata: list[str] = []

    root = Path(path_to_datasets)
    if not root.exists():
        raise FileNotFoundError(f"path_to_datasets does not exist: {root}")

    dataset_dirs = sorted(
        [d for d in root.iterdir() if d.is_dir()],
        key=lambda d: d.name,
    )
    if not dataset_dirs:
        raise FileNotFoundError(f"No dataset subdirectories found under {root}")

    for dataset_dir in dataset_dirs:
        motion_dirs = sorted(
            [d for d in dataset_dir.iterdir() if d.is_dir()],
            key=lambda d: d.name,
        )
        for motion_dir in motion_dirs:
            qualified_name = f"{dataset_dir.name}/{motion_dir.name}"
            if matches_exclude(exclude_set, dataset_dir.name, motion_dir.name):
                excl_motions.append(qualified_name)
                continue

            sample_dirs = sorted(
                [d for d in motion_dir.iterdir() if d.is_dir() and d.name.startswith("sample")],
                key=lambda d: int("".join(filter(str.isdigit, d.name))) if any(c.isdigit() for c in d.name) else 0,
            )
            for sample_dir in sample_dirs:
                motion_file = sample_dir / "motion.npz"
                if not motion_file.exists():
                    continue
                if matches_exclude(exclude_set, dataset_dir.name,
                                   motion_dir.name, sample_dir.name):
                    excl_clips.append(f"{qualified_name}/{sample_dir.name}")
                    continue

                metadata_file = sample_dir / "metadata.json"
                if metadata_file.exists():
                    with open(metadata_file) as f:
                        metadata = json.load(f)
                    object_path = metadata.get("object_path", "N/A")
                    object_name = object_path.split("/")[-2] if object_path != "N/A" else "N/A"
                else:
                    no_metadata.append(str(sample_dir))
                    object_name = "N/A"

                motion_files_by_object.setdefault(object_name, []).append(str(motion_file))

    # Stash the scan report on the returned mapping's owner instead of printing:
    # this function runs at cfg-CONSTRUCTION time, i.e. at `import orcs`, and a
    # library that talks during import is a library you cannot import quietly.
    # `_ConcatMotionLoader` prints one summary line at env BUILD, which is where
    # a human is actually watching.
    last_scan.clear()
    last_scan.update(excluded_motions=excl_motions, excluded_clips=excl_clips,
                     no_metadata=no_metadata)

    total = sum(len(v) for v in motion_files_by_object.values())
    if total == 0:
        raise FileNotFoundError(
            f"No motion files found under path_to_datasets={root} "
            f"(excluded={exclude_set or 'none'})"
        )

    return motion_files_by_object


def get_motion_files_for_objects(
    ordered_object_names: list[str],
    path_to_datasets: str,
    exclude_motions: list[str] | None = None,
) -> tuple[dict[str, list[str]], list[str]]:
    motion_files_by_object = load_motion_files_from_datasets(
        path_to_datasets, exclude_motions=exclude_motions
    )

    motion_files_for_given_objects = {}
    motion_files_object_ordered: list[str] = []
    for object_name in ordered_object_names:
        if object_name in motion_files_by_object:
            motion_files_object_ordered.extend(motion_files_by_object[object_name])
            motion_files_for_given_objects[object_name] = motion_files_by_object[object_name]
        else:
            raise ValueError(
                f"No motion files found for object '{object_name}' under "
                f"path_to_datasets={path_to_datasets}. Check your datasets or assets."
            )

    return motion_files_for_given_objects, motion_files_object_ordered
