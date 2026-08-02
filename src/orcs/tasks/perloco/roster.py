"""Which tiles and clips a run uses. One file, read by the grid AND the command.

Selection is by AXIS (`families` x `levels`) because mjlab's sub-terrain grid is
rectangular — an arbitrary set of cells has no grid to live in.

Include, never exclude: two mechanisms for one decision is how they drift.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import tomllib

__all__ = ["Roster", "load_roster", "staged"]

ROSTER_DIR = Path(__file__).parent / "rosters"

ALL = "*"


def staged(root: Path) -> dict[str, set[float]]:
    """What is on disk: `family -> {level}`, tiles with at least one clip."""
    found: dict[str, set[float]] = {}
    for tile in root.rglob("tile.json"):
        d = tile.parent
        if any(d.glob("sample*/motion.npz")):
            found.setdefault(d.parent.name, set()).add(float(d.name[len("level_"):]))
    return found


@dataclass(frozen=True)
class Roster:
    families: tuple[str, ...]
    levels: tuple[float, ...]
    clips: dict[str, tuple[str, ...]]
    """tile key -> sample names. Absent key = every clip of that tile."""

    @property
    def n_rows(self) -> int:
        return len(self.levels)

    @property
    def tile_keys(self) -> tuple[str, ...]:
        """Tile index -> key, family-major so index == col * n_rows + row."""
        return tuple(f"{f}/level_{level:.2f}"
                     for f in self.families for level in self.levels)


def load_roster(source: str, root: Path, path: str | Path | None = None) -> Roster:
    """Read `rosters/<source>.toml` (or `path`) and check it against `root`.

    Pass `path` to swap the roster without touching anything else — that is how
    an eval run isolates a subset on byte-identical infrastructure.
    """
    path = Path(path) if path else ROSTER_DIR / f"{source}.toml"
    if not path.exists():
        raise FileNotFoundError(f"no roster at {path}")
    spec = tomllib.loads(path.read_text())

    on_disk = staged(root)
    if not on_disk:
        raise FileNotFoundError(
            f"no staged tiles under {root} — run scripts/stage_terrain_motions.py")

    want_f, want_l = spec.get("families", ALL), spec.get("levels", ALL)
    families = tuple(sorted(on_disk if want_f == ALL else want_f))
    levels = tuple(sorted(
        set().union(*on_disk.values()) if want_l == ALL else map(float, want_l)))

    missing = [f"{f}/level_{level:.2f}" for f in families for level in levels
               if level not in on_disk.get(f, ())]
    if missing:
        raise FileNotFoundError(
            f"{path}: {len(missing)} selected tile(s) are not staged, e.g. "
            f"{missing[:3]}")

    clips = {k: tuple(v) for k, v in spec.get("clips", {}).items()}
    unknown = set(clips) - set(Roster(families, levels, {}).tile_keys)
    if unknown:
        raise ValueError(f"{path}: [clips] names tiles not in the grid: {sorted(unknown)}")
    return Roster(families, levels, clips)
