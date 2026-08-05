"""Assets — robot and object entity configs, one module per asset family.

  g1        flat-hand Unitree G1 (SONIC WBC base robot)
  objects   omni-object variant entity (one object per world, round-robin)
  ball      thrown-ball sphere (generated, not a mesh — see its docstring)

Add a robot by dropping a sibling module and re-exporting its ``get_*_cfg``
below; asset trees resolve through :mod:`orcs.core.paths`, never hardcoded.
"""

from orcs.assets.ball import (
    BALL_BODY_NAME,
    BALL_GEOM_NAME,
    ball_entity_cfg,
)
from orcs.assets.g1 import (
    find_body,
    find_body_or_none,
    flat_hand_spec,
    get_g1_flat_hand_cfg,
)
from orcs.assets.objects import (
    OBJECT_BODY_NAME,
    Collision,
    omni_object_entity_cfg,
)

__all__ = [
    "BALL_BODY_NAME",
    "BALL_GEOM_NAME",
    "OBJECT_BODY_NAME",
    "Collision",
    "ball_entity_cfg",
    "find_body",
    "find_body_or_none",
    "flat_hand_spec",
    "get_g1_flat_hand_cfg",
    "omni_object_entity_cfg",
]
