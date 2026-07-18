"""Assets — flat-hand G1 (SONIC WBC base) + omni-object variant entity.

G1: stock model with capsule hand colliders swapped for thin box plates via
MjSpec surgery (stable object contact); joint dynamics untouched.

Objects: mjlab analogue of fcrl's MultiAssetSpawner(random_choice=False) —
every world simulates ONE object from `object_names`, assigned round-robin
(world i -> object i % K) and fixed for the whole run. The authoritative
env->object table is `env.sim.world_to_variant["<entity>"]` — consumers
(OmniObjectMotionCommand) read it from there, never recompute.

Object assets come from the local `assets` dependency: each object dir ships
generated MuJoCo bodies (`<name>_cvx_dcmp.xml` / `<name>_cvx_hull.xml`,
see make_object_models.py) with a freejoint, per-geom convex-part masses,
and a textured visual mesh. The root body is renamed to a common
OBJECT_BODY_NAME so all variants share one kinematic topology (a
VariantEntityCfg requirement) and so contact sensors can match by body,
independent of hull-vs-decomposition geom counts.
"""

from __future__ import annotations

from pathlib import Path
from typing import Callable, Literal, Mapping, Sequence

import mujoco
import numpy as np
from mjlab.asset_zoo.robots.unitree_g1.g1_constants import (
    FULL_COLLISION,
    G1_ARTICULATION,
    G1_XML,
    KNEES_BENT_KEYFRAME,
)
from mjlab.entity import EntityCfg
from mjlab.entity.variants import VariantEntityCfg

_ASSETS_SOURCE = Path(__file__).resolve().parents[2] / "dependencies/assets/source"
_FLAT_HAND_MESH_DIR = _ASSETS_SOURCE / "g1/meshes"

# ---------------------------------------------------------------------------
# Flat-hand G1
# ---------------------------------------------------------------------------

_SIDES = ("left", "right")
_VISUAL_HAND_POS = {"left": (0.0415, 0.003, 0.0), "right": (0.0415, -0.003, 0.0)}
_Y_OFFSET = {"left": -0.01, "right": 0.01}


def _find_body_or_none(root: mujoco.MjsBody, name: str) -> mujoco.MjsBody | None:
    if root.name == name:
        return root
    for child in root.bodies:
        found = _find_body_or_none(child, name)
        if found is not None:
            return found
    return None


def _find_body(root: mujoco.MjsBody, name: str) -> mujoco.MjsBody:
    found = _find_body_or_none(root, name)
    if found is None:
        raise ValueError(f"body {name!r} not found")
    return found


def _replace_hand_with_box(spec: mujoco.MjSpec) -> None:
    """Replace capsule hand colliders with thin box plates + flat_hand visual."""
    for side in _SIDES:
        stl_path = _FLAT_HAND_MESH_DIR / f"{side}_flat_hand.stl"
        mesh = spec.add_mesh()
        mesh.name = f"{side}_flat_hand"
        mesh.file = str(stl_path)

        body = _find_body(spec.worldbody, f"{side}_wrist_yaw_link")

        # Swap collision: capsule -> box
        for g in body.geoms:
            if g.name == f"{side}_hand_collision":
                g.fromto[:] = np.nan
                g.type = mujoco.mjtGeom.mjGEOM_BOX
                g.size[:] = [0.07, 0.005, 0.045]
                g.pos[:] = [0.12, _Y_OFFSET[side], 0.0]
                break

        # Swap visual: rubber_hand mesh -> flat_hand mesh
        for g in body.geoms:
            if g.meshname == f"{side}_rubber_hand":
                g.meshname = f"{side}_flat_hand"
                g.pos[:] = _VISUAL_HAND_POS[side]
                break


def _flat_hand_spec() -> mujoco.MjSpec:
    spec = mujoco.MjSpec.from_file(str(G1_XML))
    _replace_hand_with_box(spec)
    return spec


def get_g1_flat_hand_cfg() -> EntityCfg:
    """G1 with thin box-plate hand collision (stable training)."""
    return EntityCfg(
        init_state=KNEES_BENT_KEYFRAME,
        collisions=(FULL_COLLISION,),
        spec_fn=_flat_hand_spec,
        articulation=G1_ARTICULATION,
    )


# ---------------------------------------------------------------------------
# Omni-object variant entity
# ---------------------------------------------------------------------------

_OBJECT_GROUPS = ("omomo_objects", "sugar_objects", "custom_objects")

OBJECT_BODY_NAME = "object"
"""Common root-body name across all object variants (contact-sensor anchor)."""

Collision = Literal["cvx_dcmp", "cvx_hull"]


def _object_xml(name: str, collision: str) -> Path:
    for group in _OBJECT_GROUPS:
        xml = _ASSETS_SOURCE / group / name / f"{name}_{collision}.xml"
        if xml.exists():
            return xml
    raise FileNotFoundError(
        f"No {name}_{collision}.xml for object '{name}' under "
        f"{_ASSETS_SOURCE}/{{{','.join(_OBJECT_GROUPS)}}}/{name}/ — generate it "
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
