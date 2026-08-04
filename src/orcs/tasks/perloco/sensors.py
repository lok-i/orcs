"""PerLoco sensors — the height scan, and the terrain kill switch it rides with.

The height scan IS perloco's exteroception: the only thing the adapter learns
to read that uolm's does not have. Deliberately mjlab's stock rough-terrain
scan, unmodified in pattern and alignment, so "does an adapter learn to use a
height scan" is not entangled with "is this a good height scan".

`include_geom_groups=(0,)` is load-bearing: terrain geoms are group 0, so the
rays see the terrain and never the robot's own visual meshes.
"""

from __future__ import annotations

from mjlab.sensor import GridPatternCfg, ObjRef, RayCastSensorCfg

from orcs.core.sensors import (  # noqa: F401 — perloco's public sensor surface
    GROUND_CONTACT_SENSOR_NAME,
    LOCOMANIP_KILL_BODIES,
    ROOT_KILL_BODIES,
    STRICT_KILL_BODIES,
    ground_contact_sensor,
)

__all__ = [
    "TERRAIN_SCAN_SENSOR_NAME", "SCAN_MAX_DISTANCE",
    "PERLOCO_KILL_BODIES", "terrain_scan_sensor",
    "GROUND_CONTACT_SENSOR_NAME", "ground_contact_sensor",
    "ROOT_KILL_BODIES", "LOCOMANIP_KILL_BODIES", "STRICT_KILL_BODIES",
]

TERRAIN_SCAN_SENSOR_NAME = "terrain_scan"

SCAN_MAX_DISTANCE = 5.0
"""Ray length, metres. Also the obs scale (1/max_distance) and the miss value,
so a ray off the tile edge reads as "far below" rather than as a hole."""

PERLOCO_KILL_BODIES = ROOT_KILL_BODIES
"""The root link ALONE. Climbing puts hands, forearms, knees and shins on the
terrain BY DESIGN — that is the task, not a fall. Anything wider terminates on
the reference motion itself."""


def terrain_scan_sensor(
    frame: str = "pelvis",
    *,
    size: tuple[float, float] = (1.6, 1.0),
    resolution: float = 0.1,
    debug_vis: bool = True,
) -> RayCastSensorCfg:
    """Downward height scan, yaw-aligned — mjlab's rough-terrain scan verbatim.

    `ray_alignment="yaw"` drops pitch and roll, so the grid stays world-level
    and a reading means "terrain height here", not "terrain height along
    wherever the pelvis is currently pointing" — the latter couples the
    exteroception to the robot's own attitude and is unreadable during the
    large body-pitch excursions climbing produces.

    `frame` is a knob because the right one is an open question: a pelvis-down
    scan sees surfaces to STAND on, which is what locomotion needs, but the
    climb clips also load hands onto surfaces at chest height that a downward
    scan cannot see. Multi-frame is a tuple away (`frame=` accepts one ObjRef
    or several) if that turns out to matter.

    Default 1.6 x 1.0 m at 0.1 m = 17 x 11 = 187 rays.
    """
    return RayCastSensorCfg(
        name=TERRAIN_SCAN_SENSOR_NAME,
        frame=ObjRef(type="body", name=frame, entity="robot"),
        ray_alignment="yaw",
        pattern=GridPatternCfg(size=size, resolution=resolution),
        max_distance=SCAN_MAX_DISTANCE,
        exclude_parent_body=True,
        include_geom_groups=(0,),  # terrain only
        debug_vis=debug_vis,
    )
