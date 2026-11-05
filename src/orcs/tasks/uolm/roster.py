"""Explicit native UOLM selections and their staged-data contract.

Source pins ship with the roster so a wheel can stage the same data without a
repository checkout or a retained upstream Git/LFS cache.
"""

from __future__ import annotations

import hashlib
import json
import tomllib
from dataclasses import dataclass
from importlib.resources import files
from pathlib import Path

from orcs.core.paths import DATA_ROOT

STAGING_VERSION = 1
COMPANIONS = ("motion.npz", "object_motion.npz", "contact_matrix.npz", "metadata.json")


@dataclass(frozen=True)
class MotionRoster:
    name: str
    spec: dict
    sha256: str

    @property
    def motions(self) -> tuple[str, ...]:
        return tuple(self.spec["motions"])

    @property
    def object_names(self) -> tuple[str, ...]:
        return tuple(self.spec["object_names"])

    def root(self, override: str | Path | None = None) -> Path:
        return Path(override) if override is not None else DATA_ROOT / self.spec["root"]


def _relative(value: str) -> bool:
    path = Path(value)
    return bool(value) and not path.is_absolute() and all(
        part not in (".", "..") for part in value.split("/")
    )


def load_roster(name: str = "suitcase") -> MotionRoster:
    path = Path(name)
    resource = path if path.suffix == ".toml" else files("orcs").joinpath(
        f"tasks/uolm/rosters/{name}.toml"
    )
    raw = resource.read_bytes()
    spec = tomllib.loads(raw.decode())
    motions, objects = spec["motions"], spec["object_names"]
    for label, values in (("motions", motions), ("object_names", objects)):
        if not values or not all(isinstance(v, str) for v in values):
            raise ValueError(f"{name}: {label} must be nonempty strings")
        if len(set(values)) != len(values):
            raise ValueError(f"{name}: duplicate {label}")
    if not _relative(spec["root"]):
        raise ValueError(f"{name}: invalid relative data root")
    for motion in motions:
        parts = motion.split("/")
        if not _relative(motion) or len(parts) != 3 or parts[-1] != "sample1":
            raise ValueError(f"{name}: invalid sample path {motion!r}")
    return MotionRoster(path.stem, spec, hashlib.sha256(raw).hexdigest())


def resolve_roster(roster: MotionRoster, root: str | Path | None = None) -> Path:
    """Require a complete, current install before registering the task."""
    root = roster.root(root)
    hint = "run `bash scripts/setup/sync_data.sh hoiretarget`"
    manifest = root / "manifest.json"
    if not manifest.is_file():
        raise FileNotFoundError(f"Native roster is not staged under {root}; {hint}")
    recorded = json.loads(manifest.read_text())
    if (recorded.get("roster_sha256") != roster.sha256
            or recorded.get("staging_version") != STAGING_VERSION
            or recorded.get("motions") != list(roster.motions)):
        raise ValueError(f"Stale roster under {root}; move it aside and {hint}")
    for motion in roster.motions:
        for filename in COMPANIONS:
            path = root / motion / filename
            if not path.is_file():
                raise FileNotFoundError(f"Incomplete roster: {path}; {hint}")
    return root
