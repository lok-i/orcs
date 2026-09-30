"""UOLM's public motion-scan compatibility surface.

The implementation lives below orcs.core so setup can derive the exact same
roster from metadata without importing MJLab or registering tasks.
"""

from orcs.core.data.retargeted import (
    get_motion_files_for_objects,
    load_motion_files_from_datasets,
)
from orcs.core.data.scan import (  # noqa: F401 — public compatibility surface
    last_scan,
    load_field_or_make_zeros,
    matches_exclude,
    motion_dirs,
    scan_flat,
    scan_grouped,
)

__all__ = [
    "last_scan",
    "matches_exclude",
    "load_field_or_make_zeros",
    "motion_dirs",
    "scan_flat",
    "load_motion_files_from_datasets",
    "get_motion_files_for_objects",
]
