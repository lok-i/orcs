"""Robot<->ground contact sensing and the kill-body vocabulary.

Task-blind: every task that puts a G1 on the ground needs to answer "which
touchdown ends the episode", and the answer is a body-pattern tuple, not a
semantics. Object-filtered contact graphs are NOT here — they mention an
object, so they belong to the task that has one.

**"ground", not "terrain", is the name here on purpose.** What core knows is
that the robot stands on something; whether that something is a plane or a
staged sub-terrain grid is the task's business, and `core` is the layer that
must not be able to tell (docs/ethos.md §4 rule 2 — the rule is a grep, so a
symbol spelling it defeats it). `"terrain"` survives only as the mjlab BODY
name below, which is an mjlab fact, not ours.

That body-mode match is what covers both mjlab flavours: BODY names agree
(`TerrainEntity` puts a plane and a generated grid alike in one body named
`terrain`) while GEOM names do not — the plane's geom is `terrain`, but the
generator renames every tile geom to `terrain_<n>`. A secondary `ContactMatch`
with no `entity` is a LITERAL name, not a regex, so a geom-mode `"terrain"`
silently works on a plane and raises `unrecognized name 'terrain'` the moment
the task grows a sub-terrain grid.
"""

from __future__ import annotations

from mjlab.sensor import ContactSensorCfg
from mjlab.sensor.contact_sensor import ContactMatch

__all__ = [
    "GROUND_CONTACT_SENSOR_NAME",
    "LOCOMANIP_KILL_BODIES",
    "ROOT_KILL_BODIES",
    "STRICT_KILL_BODIES",
    "ground_contact_sensor",
]

GROUND_CONTACT_SENSOR_NAME = "torso_ground_contact"

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


def ground_contact_sensor(
    pattern: tuple[str, ...] = LOCOMANIP_KILL_BODIES,
    exclude: tuple[str, ...] = (),
) -> ContactSensorCfg:
    """Robot<->ground contact on the bodies whose touchdown ends the episode."""
    return ContactSensorCfg(
        name=GROUND_CONTACT_SENSOR_NAME,
        primary=ContactMatch(
            mode="geom", pattern=pattern, exclude=exclude or None, entity="robot"),
        secondary=ContactMatch(mode="body", pattern="terrain"),  # mjlab's body name
        fields=("found",),
    )
