"""UOLM sensor configs — what the env can measure.

The object-filtered contact graph lives here (it mentions an object). The
terrain kill-switch and the kill-body vocabulary are task-blind and live in
:mod:`orcs.core.sensors`; they are re-exported so a uolm reader still sees one
sensor surface.

`CONTACT_GRAPH_BODY_NAMES` is the single source of column order — 1:1 with the
demo `contact_matrix.npz` legend names, and shared by the reference
(`bodywise_contact_cmd`), the live read (`bodywise_saturated_force`), and the
`object_contact_consistency` reward.
"""

from __future__ import annotations

from mjlab.sensor import ContactSensorCfg
from mjlab.sensor.contact_sensor import ContactMatch

from orcs.core.sensors import (  # noqa: F401 — uolm's public sensor surface
    LOCOMANIP_KILL_BODIES,
    ROOT_KILL_BODIES,
    STRICT_KILL_BODIES,
    TERRAIN_CONTACT_SENSOR_NAME,
    terrain_contact_sensor,
)

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

UOLM_KILL_BODIES = ROOT_KILL_BODIES
"""fcrl parity (2026-08-01): the root link ALONE. Carrying a 9.6 kg tire, a
torso or shoulder brush with the ground is a recoverable state, not a fall —
killing on it truncates episodes before the goal earns credit. Narrower than
LOCOMANIP_KILL_BODIES on purpose; that constant stays as-is for consumers
(vibe's repose) whose kill set was never in question."""


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
