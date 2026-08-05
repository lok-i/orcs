"""Sensor configs — what ends an episode.

Two contact sensors, one question each. The ground kill-switch and the kill-body
vocabulary are task-blind and come from :mod:`orcs.core.sensors`, re-exported so
a dodge reader still sees one sensor surface.
"""

from __future__ import annotations

from mjlab.sensor import ContactSensorCfg
from mjlab.sensor.contact_sensor import ContactMatch

from orcs.assets import BALL_BODY_NAME
from orcs.core.sensors import (  # noqa: F401 — dodge's public sensor surface
    GROUND_CONTACT_SENSOR_NAME,
    ROOT_KILL_BODIES,
    ground_contact_sensor,
)

__all__ = [
    "BALL_CONTACT_SENSOR_NAME",
    "DODGE_KILL_BODIES",
    "GROUND_CONTACT_SENSOR_NAME",
    "ball_contact_sensor",
    "ground_contact_sensor",
]

BALL_CONTACT_SENSOR_NAME = "ball_contact"

DODGE_KILL_BODIES = ROOT_KILL_BODIES
"""What counts as a FALL: the root link touching the ground. Narrow on purpose —
an evasion that puts a hand or a knee down is a dodge that worked, not a fall,
and killing on it would terminate the behaviour we are trying to elicit."""

_EVERY_COLLISION_GEOM = (".*_collision",)
"""The any-link hit surface. Spelled here rather than borrowed from a
`*_KILL_BODIES` constant: those name which GROUND contacts end an episode, a
list that legitimately shrinks (feet, then knees, then hands are allowed down).
The ball criterion is the opposite — it must never shrink, feet included."""


def ball_contact_sensor(
    exclude: tuple[str, ...] = (), name: str = BALL_CONTACT_SENSOR_NAME
) -> ContactSensorCfg:
    """Robot <-> ball contact on EVERY collision geom — the any-link hit criterion.

    Strictly harder than a root-clearance test, and deliberately: most misses of
    a pelvis-only score are limb grazes, so a policy that protects only the base
    can look safe while being struck. `found` (not a force threshold) is the
    whole point — any touch is a hit.
    """
    return ContactSensorCfg(
        name=name,
        primary=ContactMatch(
            mode="geom", pattern=_EVERY_COLLISION_GEOM,
            exclude=exclude or None, entity="robot"),
        secondary=ContactMatch(
            mode="body", pattern=BALL_BODY_NAME, entity=BALL_BODY_NAME),
        fields=("found",),
    )
