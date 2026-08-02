"""Dataset discovery and motion loading — task-blind.

  scan     find motion.npz files; group them by a caller-supplied key
  loader   concatenate clips into one timeline (the MotionLoader contract)

Knows what a clip and a sample are. Does not know what an object or a terrain
is — that lives in whichever task supplies the grouping key.
"""

from orcs.core.data.scan import (  # noqa: F401
    last_scan,
    load_field_or_make_zeros,
    matches_exclude,
    motion_dirs,
    scan_flat,
    scan_grouped,
)
