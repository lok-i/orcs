"""Finding motion clips on disk — no sim, no torch-device semantics, no task.

Layout vocabulary, used verbatim throughout orcs:

    <root>/<dataset>/<motion>/<sampleN>/motion.npz

A MOTION is one demo folder; a CLIP is one ``sampleN`` take of it (a motion
holds 1..200+).

Two walks, because tasks ask two different questions:

  scan_flat(root)          "give me every clip" — depth-invariant, order by path
  scan_grouped(root, key)  "give me every clip, bucketed" — the caller supplies
                           the bucket via `key(sample_dir)`. uolm keys on the
                           object recorded in metadata.json; a terrain task keys
                           on the path. Core never decides what a bucket means.
"""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path

import numpy as np
import torch

__all__ = [
    "last_scan", "matches_exclude", "load_field_or_make_zeros",
    "motion_dirs", "scan_flat", "scan_grouped",
]

last_scan: dict[str, list[str]] = {
    "excluded_motions": [], "excluded_clips": [], "no_metadata": []
}
"""Report from the most recent dataset scan — motions dropped whole, individual
clips dropped, and clips whose key could not be resolved. Scans run at cfg
CONSTRUCTION time (i.e. at ``import orcs``), and a library that talks during
import is a library you cannot import quietly — so nothing here prints. The
motion loader prints one summary line at env BUILD, where a human is watching.
"""


def matches_exclude(
    exclude_set: set[str],
    dataset: str,
    motion: str,
    sample: str | None = None,
) -> bool:
    """Does an `exclude_motions` entry cover this motion folder / clip?

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
    """(tensor, existed) — the field, or zeros of `shape` when absent."""
    if data is not None and field_name in data:
        data_tensor = torch.tensor(data[field_name], device=device)
        if verbose:
            print(f"[INFO]: Field {field_name} found in data. "
                  f"Loading data with shape {data_tensor.shape}.")
        return data_tensor, True
    data_tensor = torch.zeros(shape, device=device)
    if verbose:
        print(f"[INFO]: Field {field_name} not found in data. "
              f"Using zeros with shape {data_tensor.shape}.")
    return data_tensor, False


def motion_dirs(dataset_dir: str | list[str]) -> list[Path]:
    """Motion folders (each holds sampleX/motion.npz), found DEPTH-INVARIANTLY.

    `dataset_dir` is one root (str/Path) or a list of paths; each path may be a
    motion folder itself OR any ancestor of them — a root whose subdirs are
    motions, a root grouping datasets-then-motions, or an explicit list of
    motion folders. We locate every `sampleX/motion.npz` beneath each root and
    take its grandparent as the motion folder, so layouts differ only in
    nesting depth and all of them just work. Sorted by path; deduped, first
    occurrence wins (preserves list order across roots).
    """
    roots = ([Path(dataset_dir)] if isinstance(dataset_dir, (str, Path))
             else [Path(d) for d in dataset_dir])
    seen: dict[Path, None] = {}
    for root in roots:
        for mf in sorted(root.rglob("motion.npz")):
            seen.setdefault(mf.parent.parent, None)  # <motion>/<sampleX>/motion.npz
    return list(seen)


def _sample_dirs(motion_dir: Path) -> list[Path]:
    """`sampleN` subdirs, ordered numerically (sample2 before sample10)."""
    return sorted(
        (d for d in motion_dir.iterdir()
         if d.is_dir() and d.name.startswith("sample")),
        key=lambda d: int("".join(filter(str.isdigit, d.name)) or "0"),
    )


def scan_flat(
    dataset_dir: str | list[str],
    exclude_motions: tuple[str, ...] | None = None,
) -> list[str]:
    """Every `motion.npz` under `dataset_dir`, depth-invariantly, path-ordered.

    Fills :data:`last_scan`. Raises when the walk finds nothing — an env that
    silently trains on zero clips is worse than one that fails to build.
    """
    excl = set(exclude_motions or ())
    excl_motions: list[str] = []
    excl_clips: list[str] = []
    motion_files: list[str] = []
    for motion_dir in motion_dirs(dataset_dir):
        dataset = motion_dir.parent.name
        if matches_exclude(excl, dataset, motion_dir.name):
            excl_motions.append(f"{dataset}/{motion_dir.name}")
            continue
        for sample_dir in _sample_dirs(motion_dir):
            mf = sample_dir / "motion.npz"
            if not mf.exists():
                continue
            if matches_exclude(excl, dataset, motion_dir.name, sample_dir.name):
                excl_clips.append(
                    f"{dataset}/{motion_dir.name}/{sample_dir.name}")
                continue
            motion_files.append(str(mf))
    last_scan.clear()
    last_scan.update(excluded_motions=excl_motions, excluded_clips=excl_clips,
                     no_metadata=[])
    if not motion_files:
        raise FileNotFoundError(f"No motion.npz under {dataset_dir}")
    return motion_files


def scan_grouped(
    root: str,
    key: Callable[[Path], str | None],
    exclude_motions: list[str] | None = None,
) -> dict[str, list[str]]:
    """Walk `<root>/<dataset>/<motion>/<sampleN>/motion.npz`, bucketed by `key`.

    `key(sample_dir)` returns the bucket name, or None when it cannot be
    resolved (recorded in ``last_scan["no_metadata"]``, bucketed as "N/A").
    The two axes of order are fixed here and relied upon by callers that turn
    a bucket roster into an id space: datasets by name, motions by name,
    samples numerically.
    """
    exclude_set = set(exclude_motions) if exclude_motions else set()
    files_by_key: dict[str, list[str]] = {}
    excl_motions: list[str] = []
    excl_clips: list[str] = []
    unresolved: list[str] = []

    root_path = Path(root)
    if not root_path.exists():
        raise FileNotFoundError(f"path_to_datasets does not exist: {root_path}")
    dataset_dirs = sorted((d for d in root_path.iterdir() if d.is_dir()),
                          key=lambda d: d.name)
    if not dataset_dirs:
        raise FileNotFoundError(f"No dataset subdirectories found under {root_path}")

    for dataset_dir in dataset_dirs:
        for motion_dir in sorted((d for d in dataset_dir.iterdir() if d.is_dir()),
                                 key=lambda d: d.name):
            qualified = f"{dataset_dir.name}/{motion_dir.name}"
            if matches_exclude(exclude_set, dataset_dir.name, motion_dir.name):
                excl_motions.append(qualified)
                continue
            for sample_dir in _sample_dirs(motion_dir):
                motion_file = sample_dir / "motion.npz"
                if not motion_file.exists():
                    continue
                if matches_exclude(exclude_set, dataset_dir.name,
                                   motion_dir.name, sample_dir.name):
                    excl_clips.append(f"{qualified}/{sample_dir.name}")
                    continue
                bucket = key(sample_dir)
                if bucket is None:
                    unresolved.append(str(sample_dir))
                    bucket = "N/A"
                files_by_key.setdefault(bucket, []).append(str(motion_file))

    last_scan.clear()
    last_scan.update(excluded_motions=excl_motions, excluded_clips=excl_clips,
                     no_metadata=unresolved)

    if sum(len(v) for v in files_by_key.values()) == 0:
        raise FileNotFoundError(
            f"No motion files found under path_to_datasets={root_path} "
            f"(excluded={exclude_set or 'none'})"
        )
    return files_by_key
