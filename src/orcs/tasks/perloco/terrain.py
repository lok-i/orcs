"""Staged tiles -> an mjlab sub-terrain grid.

The whole mapping, in one line:

    grid COLUMN = family (`terrain_types`)   fixed per env at build
    grid ROW    = level  (`terrain_levels`)  difficulty, promotable at runtime

which is exactly the `<family>/level_<L>/` layout `stage_terrain_motions.py`
writes, so the grid and the clip library are two views of one directory tree.
`TerrainMotionCommand` re-derives the same `(row, col) -> tile` indexing from
:func:`staged_roster`, and both call it — a roster that drifts between the
terrain and the clips is the one bug this file exists to make impossible.

**Tile-local frames.** mjlab hands `function()` a frame whose origin is the
tile's CORNER and expects the spawn origin back in it; every stock terrain
therefore builds around `(size/2, size/2)`. Staged tiles are centred on their
own origin (boxes at ±half), so we place them at that same centre and return it
as `origin` — put them at `(0,0)` instead and half of every tile lands in its
neighbour.

**Boxes, not heightfields.** A staged tile is 1-3 exact oriented boxes
(`terrain_spec.BoxSpec`); MuJoCo's box-box pair is the cheapest it has and the
geometry is exact rather than sampled. `HFieldSpec` is the fallback for a
source whose geometry boxes cannot express — none today.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

import mujoco
import numpy as np
from mjlab.terrains.terrain_generator import (
    SubTerrainCfg,
    TerrainGeneratorCfg,
    TerrainGeometry,
    TerrainOutput,
)

__all__ = [
    "TILE_SIZE", "TileTerrainCfg", "staged_roster", "terrain_generator_cfg",
]

TILE_SIZE = (6.0, 6.0)
"""Tile footprint, metres. Measured over all 145 staged omni tiles: the widest
box half-footprint is 2.20 m and the furthest the reference pelvis travels from
the tile origin is 2.20 m, so 4.4 m is the hard floor and 6.0 leaves a margin
for RSI randomization without a robot ever seeing its neighbour's geometry."""

_FLOOR_DEPTH = 0.5
"""Thickness of the per-tile ground slab. Its TOP is z=0 — the plane every
staged clip's root trajectory is expressed against."""

_FLOOR_RGBA = (0.35, 0.36, 0.38, 1.0)
_BOX_RGBA_LO = np.array([0.24, 0.46, 0.86, 1.0])   # easiest level
_BOX_RGBA_HI = np.array([0.88, 0.34, 0.28, 1.0])   # hardest level


def _tile_dir(root: Path, family: str, level: float) -> Path:
    return root / family / f"level_{level:.2f}"


def staged_roster(
    root: str | Path,
    families: tuple[str, ...] | None = None,
    levels: tuple[float, ...] | None = None,
) -> tuple[tuple[str, ...], tuple[float, ...]]:
    """(families, levels) of a staged source — THE grid axes, sorted.

    Sorted, so column and row indices are a pure function of what is on disk:
    the terrain builder and the command build the same `(row, col) -> tile` map
    without passing one to the other.

    The grid mjlab builds is rectangular, so the roster must be too: an env
    landing on a (family, level) with no clips would have nothing to track.
    A ragged staging directory fails HERE, at cfg construction, rather than as
    an empty multinomial at the first reset.
    """
    root = Path(root)
    found: dict[str, set[float]] = {}
    for tile_json in root.rglob("tile.json"):
        d = tile_json.parent
        if not any(d.glob("sample*/motion.npz")):
            continue  # a tile with no motion is not a tile
        found.setdefault(d.parent.name, set()).add(float(d.name[len("level_"):]))
    if not found:
        raise FileNotFoundError(
            f"no staged tiles under {root} — run scripts/stage_terrain_motions.py")

    fam = tuple(sorted(families if families is not None else found))
    lvl = tuple(sorted(levels if levels is not None else set.union(*found.values())))
    missing = [f"{f}/level_{level:.2f}" for f in fam for level in lvl
               if level not in found.get(f, ())]
    if missing:
        raise FileNotFoundError(
            f"staged roster is not rectangular — {len(missing)} (family, level) "
            f"cell(s) have no clips, e.g. {missing[:3]}. Restage, or pass "
            f"families=/levels= to select a complete sub-grid.")
    return fam, lvl


@dataclass(kw_only=True)
class TileTerrainCfg(SubTerrainCfg):
    """One staged FAMILY as a grid column; `difficulty` selects the level row.

    `difficulty` arrives as mjlab's `row / (num_rows - 1)` over
    `difficulty_range`, which :func:`terrain_generator_cfg` pins to (0, 1) so
    the inverse is exact. The level is an INDEX into a staged roster, not an
    interpolation — there is no tile between z_scale 0.9 and 1.0.
    """

    root: Path = field(default_factory=Path)
    family: str = ""
    levels: tuple[float, ...] = ()

    def function(
        self, difficulty: float, spec: mujoco.MjSpec, rng: np.random.Generator
    ) -> TerrainOutput:
        del rng  # staged geometry — nothing to sample
        row = int(round(difficulty * (len(self.levels) - 1)))
        level = self.levels[row]
        tile = json.loads((_tile_dir(self.root, self.family, level)
                           / "tile.json").read_text())

        body = spec.body("terrain")
        cx, cy = 0.5 * self.size[0], 0.5 * self.size[1]
        t = row / max(len(self.levels) - 1, 1)
        box_rgba = tuple((1 - t) * _BOX_RGBA_LO + t * _BOX_RGBA_HI)

        geoms = [TerrainGeometry(
            geom=body.add_geom(
                type=mujoco.mjtGeom.mjGEOM_BOX,
                size=(cx, cy, 0.5 * _FLOOR_DEPTH),
                pos=(cx, cy, -0.5 * _FLOOR_DEPTH),
                rgba=_FLOOR_RGBA,
            ),
            color=_FLOOR_RGBA,
        )]
        for b in tile["boxes"]:
            geoms.append(TerrainGeometry(
                geom=body.add_geom(
                    type=mujoco.mjtGeom.mjGEOM_BOX,
                    size=tuple(b["half"]),
                    pos=(cx + b["pos"][0], cy + b["pos"][1], b["pos"][2]),
                    quat=tuple(b["quat"]),
                    rgba=box_rgba,
                ),
                color=box_rgba,
            ))
        # Spawn origin == the staged tile's own origin, so RSI is
        # `clip.root_pos + env_origins` and nothing else.
        return TerrainOutput(origin=np.array([cx, cy, 0.0]), geometries=geoms)


def terrain_generator_cfg(
    root: str | Path,
    families: tuple[str, ...],
    levels: tuple[float, ...],
    *,
    size: tuple[float, float] = TILE_SIZE,
) -> TerrainGeneratorCfg:
    """The sub-terrain grid: one column per family, one row per level.

    `curriculum=True` is what pins column == family (mjlab then ignores
    `num_cols` and uses `len(sub_terrains)`), which is the whole reason the
    env->clip map is derivable rather than sampled.
    """
    root = Path(root)
    return TerrainGeneratorCfg(
        size=size,
        curriculum=True,
        num_rows=len(levels),
        num_cols=len(families),  # ignored under curriculum; kept honest
        difficulty_range=(0.0, 1.0),  # TileTerrainCfg inverts this exactly
        border_width=0.0,
        color_scheme="height",  # re-applies TerrainGeometry.color as-is
        sub_terrains={
            f: TileTerrainCfg(root=root, family=f, levels=levels, size=size)
            for f in families
        },
    )
