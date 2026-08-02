"""Source plugins — one per (terrain, motion) dataset.

Each implements `orcs.tasks.perloco.terrain_spec.TerrainMotionSource`: pure
readers that answer `.tiles()` and `.clips()`, so they are testable without a
GPU and every dataset quirk stays out of the runtime.
"""

from orcs.tasks.perloco.sources.grail import GrailSource  # noqa: F401
from orcs.tasks.perloco.sources.omni import OmniRetargetSource  # noqa: F401

SOURCES = {"omni": OmniRetargetSource, "grail": GrailSource}
