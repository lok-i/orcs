"""Omni-object variant entity.

mjlab analogue of fcrl's MultiAssetSpawner(random_choice=False) — every world
simulates ONE object from `object_names`, assigned round-robin (world i ->
object i % K) and fixed for the whole run. The authoritative env->object table
is `env.sim.world_to_variant["<entity>"]` — consumers (ObjectMotionCommand)
read it from there, never recompute.

Object assets come from the ``assets`` dep: each object dir ships generated
MuJoCo bodies (`<name>_cvx_dcmp.xml` / `<name>_cvx_hull.xml`, see
make_object_models.py) with a freejoint, per-geom convex-part masses, and a
textured visual mesh. The root body is renamed to a common OBJECT_BODY_NAME so
all variants share one kinematic topology (a VariantEntityCfg requirement) and
so contact sensors can match by body, independent of hull-vs-decomposition geom
counts.
"""

from __future__ import annotations

from pathlib import Path
from typing import Callable, Literal, Mapping, Sequence

import mujoco
from mjlab.entity.variants import VariantEntityCfg

from orcs.core.paths import assets_source

_OBJECT_GROUPS = ("omomo_objects", "sugar_objects", "custom_objects")

OBJECT_BODY_NAME = "object"
"""Common root-body name across all object variants (contact-sensor anchor)."""

Collision = Literal["cvx_dcmp", "cvx_hull"]


def _object_xml(name: str, collision: str) -> Path:
    source = assets_source()
    for group in _OBJECT_GROUPS:
        xml = source / group / name / f"{name}_{collision}.xml"
        if xml.exists():
            return xml
    raise FileNotFoundError(
        f"No {name}_{collision}.xml for object '{name}' under "
        f"{source}/{{{','.join(_OBJECT_GROUPS)}}}/{name}/ — generate it "
        "with dependencies/assets/source/omni_objects/make_object_models.py"
    )


def _make_spec_fn(xml: Path) -> Callable[[], mujoco.MjSpec]:
    def spec_fn() -> mujoco.MjSpec:
        spec = mujoco.MjSpec.from_file(str(xml))
        root_bodies = list(spec.worldbody.bodies)
        assert len(root_bodies) == 1, f"{xml}: expected one root body"
        root_bodies[0].name = OBJECT_BODY_NAME
        return spec

    return spec_fn


def _round_robin(num_variants: int) -> Callable[[int], Sequence[int]]:
    return lambda num_envs: [i % num_variants for i in range(num_envs)]


def omni_object_entity_cfg(
    object_names: Sequence[str],
    collision: Collision | Mapping[str, Collision] = "cvx_dcmp",
) -> VariantEntityCfg:
    """VariantEntityCfg spawning one object per world, round-robin over
    `object_names`. Variant index == index into `object_names` (declaration
    order is preserved), so `sim.world_to_variant` doubles as env->object-id.

    `collision` is one representation for all objects, or a per-object
    mapping (missing names default to "cvx_dcmp") — hulls where convexity
    suffices, decompositions where concavity earns its VRAM.
    """
    if isinstance(collision, str):
        collision = {name: collision for name in object_names}
    return VariantEntityCfg(
        variants={
            name: _make_spec_fn(
                _object_xml(name, collision.get(name, "cvx_dcmp")))
            for name in object_names
        },
        assignment=_round_robin(len(object_names)),
    )
