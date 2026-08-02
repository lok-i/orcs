"""The staging interface: what a (terrain, motion) dataset must produce.

Three source datasets, one runtime. The rule that keeps that from becoming
three code paths: **the runtime loads only orcs-native files, and every dataset
quirk dies in the offline stage.** A source's whole job is to answer two
questions in these types; `scripts/stage_terrain_motions.py` does the rest
(resample, joint permute, FK, write).

Frames and units — all tile-local, metres, z=0 at the tile's ground plane, and
quaternions wxyz to match MuJoCo. A clip's root trajectory is expressed in the
SAME tile-local frame as its tile's geometry, which is what lets RSI place the
robot on the terrain by adding `env_origins` and nothing else.

Terrain geometry is boxes when it can be (exact, and the cheapest primitive
pair MuJoCo has) and a heightfield when it cannot. Both are fields of one
`TileSpec`, not two classes, because a tile may need both — a box obstacle on
a scanned slope is one tile, not two.
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass, field
from typing import Protocol, runtime_checkable

import numpy as np

__all__ = [
    "BoxSpec", "HFieldSpec", "TileSpec", "ClipSpec", "SmplSpec",
    "TerrainMotionSource", "box_from_vertices",
]


@dataclass(frozen=True)
class BoxSpec:
    """An oriented box, tile-local. Straight into `spec.add_geom(type=BOX)`."""

    pos: tuple[float, float, float]
    """Centre of the box."""
    quat: tuple[float, float, float, float]
    """Orientation, wxyz. Yaw-only for every source we have so far."""
    half: tuple[float, float, float]
    """Half-extents along the box's own axes (MuJoCo's `size` for a box)."""


@dataclass(frozen=True)
class HFieldSpec:
    """A heightfield, tile-local — the fallback for geometry boxes cannot express.

    Cannot represent overhangs (it is an elevation map, not a solid), which is
    the whole reason boxes stay tier 1.
    """

    heights: np.ndarray
    """(H, W) float32 elevation, tile-local z."""
    size: tuple[float, float]
    """XY extent covered by the grid, in metres."""
    base: float = 0.1
    """Solid thickness below the minimum height. Only needs to be enough for
    stable collision, not physically meaningful."""


@dataclass(frozen=True)
class TileSpec:
    """One terrain patch, and where it lands in the sub-terrain grid.

    `family` becomes a grid COLUMN (a `TileTerrainCfg` entry, fixed per env at
    build) and `level` becomes a ROW (difficulty, promotable at runtime). For
    OmniRetarget that is (climb_NN, z_scale) — the dataset's own two axes, and
    `level` really is difficulty, so the curriculum is honest rather than an
    index smuggled through a float.
    """

    family: str
    level: float
    boxes: tuple[BoxSpec, ...] = ()
    hfield: HFieldSpec | None = None

    @property
    def key(self) -> str:
        """`<family>/<level>` — the staged directory, and THE pairing.

        Formatted, not raw, so 0.8 and 0.80 cannot become two tiles.
        """
        return f"{self.family}/level_{self.level:.2f}"


@dataclass(frozen=True)
class SmplSpec:
    """The human reference behind a retargeted clip. Frames + contract:
    `orcs.core.data.smpl`.

    Its own frame count, deliberately. A retargeter is free to re-time what it
    retargets, and GRAIL does: 241 recon frames @ 30 Hz become 250 robot frames
    @ 25 Hz — the SAME path (xy arclength 4.625 vs 4.614 m) played 1.245x
    slower. So the two are aligned by PHASE, and staging resamples this onto
    the robot clip's grid. Aligning by real time instead lands 0.74 m off the
    root, against 0.13 m for phase (which is retargeting error: SMPL root vs
    G1 pelvis). A +-12-frame shift sweep finds no lag, so phase is exact, not
    merely better.
    """

    joints: np.ndarray
    """(S, 24, 3) SMPL-native y-up, root-centred, root rotation applied."""
    root_quat: np.ndarray
    """(S, 4) wxyz, z-up world, SMPL base rot removed."""
    joints_viz: np.ndarray
    """(S, 24, 3) z-up world, tile-local — same frame as `ClipSpec.root_pos`."""


@dataclass(frozen=True)
class ClipSpec:
    """One motion, in the source's own joint order and rate.

    `family`/`level` are the entire pairing mechanism — they name the tile this
    motion was retargeted against. Staging writes the clip under that tile's
    directory, so the runtime recovers the pairing from the path and there is
    no manifest to desync.
    """

    name: str
    """Stem used for the staged motion folder. Unique within a tile."""
    root_pos: np.ndarray
    """(T, 3) root position, tile-local."""
    root_quat: np.ndarray
    """(T, 4) root orientation, wxyz."""
    joint_pos: np.ndarray
    """(T, J) joint positions in `joint_names` order."""
    joint_names: tuple[str, ...]
    """Source joint order. REQUIRED, and staging permutes by name.

    There is deliberately no "already in the right order" escape hatch: it
    existed for one commit, GRAIL used it, and 27 of 29 joints were silently
    transposed. A source that cannot read its names off disk spells them out
    (see `sources/grail.py`); an unverifiable claim beats a wrong one only when
    it is written down."""
    fps: float
    family: str
    level: float
    smpl: SmplSpec | None = None
    """The human this clip was retargeted from, when the source ships it."""
    meta: dict = field(default_factory=dict)
    """Provenance carried into the staged metadata.json. Never read at runtime."""

    @property
    def tile_key(self) -> str:
        return f"{self.family}/level_{self.level:.2f}"


@runtime_checkable
class TerrainMotionSource(Protocol):
    """A (terrain, motion) dataset, reduced to two questions.

    Implementations are pure readers: no sim, no torch, no mjlab. That is what
    makes them testable without a GPU and what keeps dataset quirks out of the
    runtime.
    """

    name: str

    def tiles(self) -> Iterable[TileSpec]: ...

    def clips(self) -> Iterable[ClipSpec]: ...


def box_from_vertices(
    verts: np.ndarray,
    scale: tuple[float, float, float] = (1.0, 1.0, 1.0),
    offset: tuple[float, float, float] = (0.0, 0.0, 0.0),
) -> BoxSpec:
    """A z-extruded rectangle's vertices -> an exact oriented box.

    Shared by every source: OmniRetarget ships them as `.obj`, GRAIL inside
    `.usd`, and both reduce to the same 4 unique xy corners, which give the
    half-extents and the yaw directly.

    Orthogonality is asserted, not assumed — a future asset that is a general
    hexahedron must fail here rather than silently become a wrong box.
    """
    v = verts * np.asarray(scale, float) + np.asarray(offset, float)
    z0, z1 = float(v[:, 2].min()), float(v[:, 2].max())
    xy = np.unique(np.round(v[:, :2], 6), axis=0)
    if len(xy) != 4:
        raise ValueError(f"not a z-extruded rectangle: {len(xy)} xy corners")

    p0, rest = xy[0], xy[1:]
    order = np.argsort(np.linalg.norm(rest - p0, axis=1))
    e1, e2 = rest[order[0]] - p0, rest[order[1]] - p0
    n1, n2 = np.linalg.norm(e1), np.linalg.norm(e2)
    ortho = abs(float(e1 @ e2)) / (n1 * n2)
    if ortho > 1e-3:
        raise ValueError(f"box edges not orthogonal (err={ortho:.2e})")

    cx, cy = xy.mean(axis=0)
    yaw = float(np.arctan2(e1[1], e1[0]))
    return BoxSpec(
        pos=(float(cx), float(cy), 0.5 * (z0 + z1)),
        quat=(float(np.cos(yaw / 2)), 0.0, 0.0, float(np.sin(yaw / 2))),
        half=(float(n1 / 2), float(n2 / 2), float(0.5 * (z1 - z0))),
    )
