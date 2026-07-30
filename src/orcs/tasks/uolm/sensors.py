"""UOLM sensor configs — what the env can measure.

Two sensors, both contact:

  object_contact_graph    per-body robot<->object contact (the contact-graph nodes)
  torso_terrain_contact   the fall kill-switch

`_CONTACT_GRAPH_BODY_NAMES` is the single source of column order — 1:1 with the
demo `contact_matrix.npz` legend names, and shared by the reference
(`bodywise_contact_cmd`), the live read (`bodywise_saturated_force`), and the
`object_contact_consistency` reward.
"""

from __future__ import annotations

from mjlab.sensor import ContactSensorCfg
from mjlab.sensor.contact_sensor import ContactMatch

CONTACT_GRAPH_BODY_NAMES = (
    "pelvis", "torso_link",
    "left_shoulder_roll_link", "left_elbow_link", "left_wrist_yaw_link",
    "right_shoulder_roll_link", "right_elbow_link", "right_wrist_yaw_link",
    "left_knee_link", "left_ankle_roll_link",
    "right_knee_link", "right_ankle_roll_link",
)
CONTACT_GRAPH_SENSOR_NAME = "object_contact_graph"

HAND_BODY_NAMES = ("left_wrist_yaw_link", "right_wrist_yaw_link")
"""Hand subset of the graph nodes — hand contact == control-authority over the
object, so it gates the contact-conditional object perturbation."""

TERRAIN_CONTACT_SENSOR_NAME = "torso_terrain_contact"

LOCOMANIP_KILL_BODIES = ("pelvis_collision", "torso_collision", ".*shoulder.*_collision")
"""Upper-body core only. Loco-manipulation legitimately kneels and braces
(knees, thighs, forearms down while lifting), so "everything but feet" would
terminate on normal behavior."""

STRICT_KILL_BODIES = (".*_collision",)
"""Everything but the feet — for tasks where no ground contact beyond the feet
is expected. Pair with ``exclude=(".*foot.*",)``."""


def object_contact_graph_sensor(
    object_entity: str = "object", object_body: str | None = None
) -> ContactSensorCfg:
    """One object-filtered multi-primary contact sensor for all graph nodes.

    mjlab expands `primary` to P=K primaries in a single sensor, so `data.force`
    is already the batched (B, K, 3) per-body vector. Column order is MODEL
    order (find_bodies), not tuple order — consumers reorder by name via
    `sensor.primary_names`.

    Secondary matches by BODY, so it is collision-representation agnostic (box,
    convex hull, or convex decomposition — all geoms of the body participate)
    and works with variant entities whose root body shares one name across
    variants.
    """
    return ContactSensorCfg(
        name=CONTACT_GRAPH_SENSOR_NAME,
        primary=ContactMatch(
            mode="body", pattern=CONTACT_GRAPH_BODY_NAMES, entity="robot"),
        secondary=ContactMatch(
            mode="body", pattern=object_body or object_entity,
            entity=object_entity),
        fields=("found", "force"),
        reduce="netforce",
    )


def terrain_contact_sensor(
    pattern: tuple[str, ...] = LOCOMANIP_KILL_BODIES,
    exclude: tuple[str, ...] = (),
) -> ContactSensorCfg:
    """Robot<->terrain contact on the bodies whose touchdown ends the episode."""
    return ContactSensorCfg(
        name=TERRAIN_CONTACT_SENSOR_NAME,
        primary=ContactMatch(
            mode="geom", pattern=pattern, exclude=exclude or None, entity="robot"),
        secondary=ContactMatch(mode="geom", pattern="terrain"),
        fields=("found",),
    )
