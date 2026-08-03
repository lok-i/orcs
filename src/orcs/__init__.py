"""orcs — Oracle Robot Control Synthesis: privileged-observation policies for
humanoid control, and the machinery to distill them into deployable students.

orcs itself is not a task. Three layers, no upward imports:

  core     robot-generic, task-blind infra — paths, mjlab compat, clip
           discovery + timeline, the multi-clip motion command, obs atoms,
           the PPO spine, task registration. Knows joints/bodies/clips; never
           objects or terrains.
  assets   robots + objects as mjlab entity cfgs (g1, objects).
  tasks    one self-registering sub-package per task:
             tasks.uolm      Uni-Object Loco-Manipulation
                             (Orcs-Uolm-AdaptSonic, -TaRa, -AdaptSonic-Smpl)
             tasks.perloco   Perceptive Locomotion over staged (terrain, motion)
                             pairs (Orcs-PerLoco-{OmRe,Grail}-AdaptSonic,
                             -TaRa, and Grail's -AdaptSonic-Smpl)
  cli      console entry points (`orcs-stage-terrain`, ...) — above `tasks`,
           imported by nothing.

The two tasks share everything except ONE obs group: uolm's adapter reads
object kinematics, perloco's reads a terrain height scan. That the rest — the
frozen base, the multi-clip command, the agents, the critic — is literally the
same code is the thesis, not a coincidence.

Adding a task: drop ``tasks/<name>/`` that self-registers on import, then add
its import line below. Philosophy, layer contract and roadmap: docs/ethos.md.
Usage: readme.md.
"""

import orcs.tasks.perloco  # noqa: F401 — task registration
import orcs.tasks.uolm  # noqa: F401 — task registration
from orcs.core import deps as _deps
from orcs.core._mjlab_compat import apply as _apply_mjlab_compat
from orcs.tasks.perloco.mdp.commands import TerrainMotionCommandCfg
from orcs.tasks.uolm.mdp.commands import ObjectMotionCommandCfg

# Shared deps (mocke, rsl_rl) are one editable install per env and the CONSUMER's
# lock wins — drift becomes a printed line, never a silent bug. See core/deps.py.
_deps.check()

# Let mjlab's train/play tolerate task-owned multi-clip motion commands. EVERY
# orcs multi-clip cfg goes in this tuple: the sentinel is one global, so a task
# missing from it gets re-classified as single-file the moment another package
# patches after us.
_apply_mjlab_compat(
    multi_clip_cfgs=(ObjectMotionCommandCfg, TerrainMotionCommandCfg))
