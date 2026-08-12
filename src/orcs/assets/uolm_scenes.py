"""Primitive object/support scenes for reconstructed UOLM motions.

These are the privileged counterparts of Vibe's repose variants.  The names,
dimensions, masses, and table pose are intentionally one-to-one so a dynamic
retarget trained here can be consumed by either visual scene without changing
its physical task.
"""

from __future__ import annotations

from collections.abc import Callable
from functools import partial
from typing import Literal, Sequence

import mujoco
import numpy as np
from mjlab.entity import EntityCfg
from mjlab.entity.variants import VariantEntityCfg

__all__ = [
    "BIG_CUBE_HALF_EXTENT",
    "BIG_CUBE_MASS",
    "SMALL_CUBE_HALF_EXTENT",
    "SMALL_CUBE_MASS",
    "TABLE_CENTER_HEIGHT",
    "TABLE_SIZE",
    "UolmReconstructedScene",
    "reconstructed_object_entity_cfg",
    "reconstructed_object_variants_entity_cfg",
    "table_entity_cfg",
]

UolmReconstructedScene = Literal["small-cube-table", "big-cube-floor"]

BIG_CUBE_HALF_EXTENT = 0.3048
BIG_CUBE_MASS = 1.5
SMALL_CUBE_HALF_EXTENT = 0.18
SMALL_CUBE_MASS = 0.3

TABLE_SIZE = (0.75, 0.75, 0.05)
TABLE_CENTER_HEIGHT = 1.0


def _cube_spec(*, half_extent: float, mass: float) -> mujoco.MjSpec:
    spec = mujoco.MjSpec()
    body = spec.worldbody.add_body(name="object")
    body.add_freejoint(name="object_joint")
    body.add_geom(
        name="object_collision",
        type=mujoco.mjtGeom.mjGEOM_BOX,
        size=(half_extent,) * 3,
        mass=mass,
        friction=(0.6, 0.6, 0.0001),
        rgba=(0.62, 0.43, 0.24, 1.0),
    )
    return spec


def reconstructed_object_entity_cfg(scene: UolmReconstructedScene) -> EntityCfg:
    """The exact small/big primitive used by Vibe's corresponding scene."""
    if scene == "small-cube-table":
        half_extent, mass = SMALL_CUBE_HALF_EXTENT, SMALL_CUBE_MASS
    elif scene == "big-cube-floor":
        half_extent, mass = BIG_CUBE_HALF_EXTENT, BIG_CUBE_MASS
    else:
        raise ValueError(f"unknown reconstructed UOLM scene: {scene!r}")
    return EntityCfg(
        spec_fn=partial(_cube_spec, half_extent=half_extent, mass=mass),
        init_state=EntityCfg.InitialStateCfg(pos=(0.6, 0.0, half_extent)),
    )


def _cube_mesh_spec(*, half_extent: float, mass: float) -> mujoco.MjSpec:
    """One convex mesh slot, allowing size/mass to vary per simulated world."""
    h = float(half_extent)
    vertices = np.asarray(
        [
            (-h, -h, -h),
            (-h, h, -h),
            (h, -h, -h),
            (h, h, -h),
            (-h, -h, h),
            (-h, h, h),
            (h, -h, h),
            (h, h, h),
        ],
        dtype=np.float32,
    )
    faces = np.asarray(
        [
            (4, 6, 7), (4, 7, 5),
            (0, 1, 3), (0, 3, 2),
            (0, 2, 6), (0, 6, 4),
            (1, 5, 7), (1, 7, 3),
            (0, 4, 5), (0, 5, 1),
            (2, 3, 7), (2, 7, 6),
        ],
        dtype=np.int32,
    )
    spec = mujoco.MjSpec()
    mesh = spec.add_mesh(
        name="cube_mesh",
        uservert=vertices.reshape(-1).tolist(),
        userface=faces.reshape(-1).tolist(),
    )
    body = spec.worldbody.add_body(name="object")
    body.add_freejoint(name="object_joint")
    body.add_geom(
        name="object_collision",
        type=mujoco.mjtGeom.mjGEOM_MESH,
        meshname=mesh.name,
        mass=mass,
        friction=(0.6, 0.6, 0.0001),
        rgba=(0.62, 0.43, 0.24, 1.0),
    )
    return spec


def _round_robin(num_variants: int):
    return lambda num_envs: [index % num_variants for index in range(num_envs)]


def reconstructed_object_variants_entity_cfg(
    scenes: Sequence[UolmReconstructedScene],
    *,
    assignment: Callable[[int], Sequence[int]] | None = None,
) -> VariantEntityCfg:
    """One correctly-sized reconstructed cube per world, in declaration order."""
    scenes = tuple(scenes)
    if not scenes:
        raise ValueError("at least one reconstructed UOLM scene is required")
    params = {
        "small-cube-table": (SMALL_CUBE_HALF_EXTENT, SMALL_CUBE_MASS),
        "big-cube-floor": (BIG_CUBE_HALF_EXTENT, BIG_CUBE_MASS),
    }
    try:
        variants = {
            scene: partial(_cube_mesh_spec, half_extent=params[scene][0], mass=params[scene][1])
            for scene in scenes
        }
    except KeyError as exc:
        raise ValueError(f"unknown reconstructed UOLM scene: {exc.args[0]!r}") from exc
    return VariantEntityCfg(
        variants=variants,
        assignment=assignment or _round_robin(len(variants)),
    )


def _table_spec() -> mujoco.MjSpec:
    spec = mujoco.MjSpec()
    body = spec.worldbody.add_body(name="table")
    body.add_geom(
        name="table_top",
        type=mujoco.mjtGeom.mjGEOM_BOX,
        size=tuple(value / 2.0 for value in TABLE_SIZE),
        friction=(1.0, 0.005, 0.0001),
        rgba=(0.55, 0.57, 0.62, 1.0),
    )
    return spec


def table_entity_cfg(*, xy: tuple[float, float] = (0.0, 0.0)) -> EntityCfg:
    """Fixed tabletop centered beneath a selected clip's final object XY."""
    return EntityCfg(
        spec_fn=_table_spec,
        init_state=EntityCfg.InitialStateCfg(
            pos=(float(xy[0]), float(xy[1]), TABLE_CENTER_HEIGHT)
        ),
    )
