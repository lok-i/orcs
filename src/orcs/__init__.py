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
             tasks.dodge     whole-body evasion of a thrown ball
                             (Orcs-Dodge-AdaptSonic)
  cli      console entry points (`orcs-stage-terrain`, ...) — above `tasks`,
           imported by nothing.

The tasks share everything except ONE obs group: uolm's adapter reads object
kinematics, perloco's a terrain height scan, dodge's the ball's. That the rest —
the frozen base, the multi-clip command, the agents, the critic — is literally
the same code is the thesis, not a coincidence.

Dodge adds a second axis to that claim. uolm and perloco both track a demo clip,
so the adapter corrects a reference that already describes the motion; dodge's
reference is a NOMINAL STAND, constant for every frame of every episode, and the
evasion exists only as the adapter's departure from it.

Adding a task: drop ``tasks/<name>/`` that self-registers on import, then add
its import line below. Philosophy, layer contract and roadmap: docs/ethos.md.
Usage: readme.md.
"""

import orcs.tasks.dodge  # noqa: F401 — task registration
import orcs.tasks.perloco  # noqa: F401 — task registration
import orcs.tasks.uolm  # noqa: F401 — task registration
from orcs.core import deps as _deps
from orcs.core._mjlab_compat import apply as _apply_mjlab_compat
from orcs.core.mdp.commands import MultiClipMotionCommandCfg

# Shared deps (mocke, rsl_rl) are one editable install per env and the CONSUMER's
# lock wins — drift becomes a printed line, never a silent bug. See core/deps.py.
_deps.check()

MULTI_CLIP_CFGS = (MultiClipMotionCommandCfg,)
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

**The BASE class, not a list of the leaves.** Owning a multi-clip library is
exactly what subclassing `MultiClipMotionCommandCfg` means, and `isinstance`
covers subclasses — so a new task is exempt the moment it exists. The hand-
listed pair this replaced was the drift this docstring warns about: it went
stale when perloco arrived, and would have again for dodge (which uses the base
cfg directly and appears nowhere in a leaf list).
"""

_apply_mjlab_compat(multi_clip_cfgs=MULTI_CLIP_CFGS)
