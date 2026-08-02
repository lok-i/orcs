"""Robot<->terrain contact sensing and the kill-body vocabulary.

Task-blind: every task that puts a G1 on the ground needs to answer "which
touchdown ends the episode", and the answer is a body-pattern tuple, not a
semantics. Object-filtered contact graphs are NOT here — they mention an
object, so they belong to the task that has one.

The secondary match is `geom pattern="terrain"`, which covers both mjlab
terrain flavours: the ground plane geom is named `terrain`, and generated
sub-terrain tiles are named `terrain_<n>`.
"""

from __future__ import annotations

from mjlab.sensor import ContactSensorCfg
from mjlab.sensor.contact_sensor import ContactMatch

__all__ = [
    "TERRAIN_CONTACT_SENSOR_NAME",
    "LOCOMANIP_KILL_BODIES",
    "ROOT_KILL_BODIES",
    "STRICT_KILL_BODIES",
    "terrain_contact_sensor",
]

TERRAIN_CONTACT_SENSOR_NAME = "torso_terrain_contact"

ROOT_KILL_BODIES = ("pelvis_collision",)
"""The root link ALONE. Under loco-manipulation or climbing, a torso or
shoulder brush with the ground is a recoverable state, not a fall — killing on
it truncates episodes before the goal earns credit (fcrl parity, 2026-08-01)."""

LOCOMANIP_KILL_BODIES = ("pelvis_collision", "torso_collision", ".*shoulder.*_collision")
"""Upper-body core. Loco-manipulation legitimately kneels and braces (knees,
thighs, forearms down while lifting), so "everything but feet" would terminate
on normal behavior."""

STRICT_KILL_BODIES = (".*_collision",)
"""Everything but the feet — for tasks where no ground contact beyond the feet
is expected. Pair with ``exclude=(".*foot.*",)``."""


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
