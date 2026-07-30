"""orcs — Oracle Robot Control Synthesis: privileged-observation policies for
humanoid control, and the machinery to distill them into deployable students.

orcs itself is not a task. Three layers, no upward imports:

  core     agnostic infra — paths, mjlab compat. No robot/task semantics.
  assets   robots + objects as mjlab entity cfgs (g1, objects).
  tasks    one self-registering sub-package per task:
             tasks.uolm   Uni-Object Loco-Manipulation
                          (Orcs-Uolm-AdptSonic, Orcs-Uolm-TaRa, Orcs-Uolm-AdptSonic-Smpl)

Adding a task: drop ``tasks/<name>/`` that self-registers on import, then add
its import line below. Philosophy, layer contract and roadmap: docs/ethos.md.
Usage: readme.md.
"""

import orcs.tasks.uolm  # noqa: F401 — task registration
from orcs.core import deps as _deps
from orcs.core._mjlab_compat import apply as _apply_mjlab_compat
from orcs.tasks.uolm.mdp.commands import ObjectMotionCommandCfg

# Shared deps (mocke, rsl_rl) are one editable install per env and the CONSUMER's
# lock wins — drift becomes a printed line, never a silent bug. See core/deps.py.
_deps.check()

# Let mjlab's train/play tolerate task-owned multi-clip motion commands.
_apply_mjlab_compat(multi_clip_cfgs=(ObjectMotionCommandCfg,))
