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

MULTI_CLIP_CFGS = (ObjectMotionCommandCfg, TerrainMotionCommandCfg)
"""Every orcs command cfg that owns a multi-clip library — **public, because a
consumer needs it.**

mjlab's train/play gate on `isinstance(cmd, MotionCommandCfg)` and force the
single-file `--motion-file` path; the compat shim installs a sentinel that
reports False for these. That sentinel is **one global**, so whoever patches
LAST decides for every task in the process. A downstream package that patches
after us (it imports orcs, so it does) must exempt this union PLUS its own:

    _apply_mjlab_compat(multi_clip_cfgs=(*orcs.MULTI_CLIP_CFGS, MyCfg))

Spelling the union by hand instead is one added orcs task away from silently
breaking `play Orcs-<whatever>` inside the consumer's env.
"""

_apply_mjlab_compat(multi_clip_cfgs=MULTI_CLIP_CFGS)
