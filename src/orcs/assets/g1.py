"""Flat-hand Unitree G1 — the loco-manipulation base robot.

Stock mjlab G1 with capsule hand colliders swapped for thin box plates via
MjSpec surgery (capsules roll the object; plates give stable contact). Joint
dynamics — friction, armature, actuation — are untouched, so this is a
CONTACT-geometry change and nothing else. The flat_hand visual mesh comes from
the ``assets`` dep (``g1/meshes``).

Not a WBC concern: mocke owns the pre-training envs and takes whatever robot it
is handed (`profile.robot_cfg(base=...)`). Downstream consumers that need their
own variant compose on `flat_hand_spec` rather than re-deriving the surgery —
`find_body` is public for exactly that.
"""

from __future__ import annotations

import mujoco
import numpy as np
from mjlab.asset_zoo.robots.unitree_g1.g1_constants import (
    FULL_COLLISION,
    G1_ARTICULATION,
    G1_XML,
    KNEES_BENT_KEYFRAME,
)
from mjlab.entity import EntityCfg

from orcs.core.paths import assets_source

_SIDES = ("left", "right")
_VISUAL_HAND_POS = {"left": (0.0415, 0.003, 0.0), "right": (0.0415, -0.003, 0.0)}
_Y_OFFSET = {"left": -0.01, "right": 0.01}


def find_body_or_none(root: mujoco.MjsBody, name: str) -> mujoco.MjsBody | None:
    if root.name == name:
        return root
    for child in root.bodies:
        found = find_body_or_none(child, name)
        if found is not None:
            return found
    return None


def find_body(root: mujoco.MjsBody, name: str) -> mujoco.MjsBody:
    found = find_body_or_none(root, name)
    if found is None:
        raise ValueError(f"body {name!r} not found")
    return found


def _replace_hand_with_box(spec: mujoco.MjSpec) -> None:
    """Replace capsule hand colliders with thin box plates + flat_hand visual."""
    mesh_dir = assets_source() / "g1/meshes"
    for side in _SIDES:
        stl_path = mesh_dir / f"{side}_flat_hand.stl"
        mesh = spec.add_mesh()
        mesh.name = f"{side}_flat_hand"
        mesh.file = str(stl_path)

        body = find_body(spec.worldbody, f"{side}_wrist_yaw_link")

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


def flat_hand_spec() -> mujoco.MjSpec:
    """Stock G1 spec with the hand surgery applied — the composition point for
    consumers that need to layer their own spec edits on top."""
    spec = mujoco.MjSpec.from_file(str(G1_XML))
    _replace_hand_with_box(spec)
    return spec


def get_g1_flat_hand_cfg() -> EntityCfg:
    """G1 with thin box-plate hand collision (stable training)."""
    return EntityCfg(
        init_state=KNEES_BENT_KEYFRAME,
        collisions=(FULL_COLLISION,),
        spec_fn=flat_hand_spec,
        articulation=G1_ARTICULATION,
    )


__all__ = ["get_g1_flat_hand_cfg", "flat_hand_spec", "find_body", "find_body_or_none"]
