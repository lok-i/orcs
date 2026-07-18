"""sortr — SONIC REtarget & REfine: a framework for retargeting the frozen SONIC
WBC to mjlab tasks.

sortr itself is not a task. Each task lives in its own sub-package and
self-registers on import:

  sortr.uolm   Uni-Object Loco-Manipulation (Sortr-Uolm, Sortr-Uolm-Smpl)

Future tasks (locomotion, whole-body control, ...) drop in as sibling packages
and add their line below. The mjlab compat shim (multi-clip motion command,
``--agent initial``, VRAM caps) is applied once here for all tasks.
"""

import sortr.uolm  # noqa: F401 — task registration
from sortr._mjlab_compat import apply as _apply_mjlab_compat

_apply_mjlab_compat()  # let mjlab train/play tolerate the multi-clip "motion" command
