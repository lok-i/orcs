"""MJLab task-discovery entry point and ORCS integration wiring.

ORCS is infrastructure plus a family of self-registering task packages. MJLab
imports this module through the ``mjlab.tasks`` entry-point group; ordinary
``import orcs`` intentionally does not. Keeping registration here prevents an
ORCS console script from recursively loading every sibling task package while
the ORCS package is still initializing.

Adding a task: create ``orcs.tasks.<name>`` with a registration table, then
import it below. Philosophy, layer contract, and roadmap live in
``docs/ethos.md``; usage lives in ``readme.md``.
"""

import orcs.tasks.dodge as _dodge
import orcs.tasks.perloco as _perloco
import orcs.tasks.uolm as _uolm
from orcs.core import deps as _deps
from orcs.core._mjlab_compat import apply as _apply_mjlab_compat
from orcs.core.mdp.commands import MultiClipMotionCommandCfg

# Shared deps (mocke, rsl_rl) are one editable install per env and the CONSUMER's
# lock wins — drift becomes a printed line, never a silent bug. See core/deps.py.
_deps.check()

SKIP_REASON = {
    **_dodge.SKIP_REASON,
    **_perloco.SKIP_REASON,
    **_uolm.SKIP_REASON,
}
"""Every task omitted because its optional data is unavailable.

Missing optional datasets are normal for consumers such as Vibe, so import
stays quiet. This aggregate is the explicit diagnostic surface.
"""

MULTI_CLIP_CFGS = (MultiClipMotionCommandCfg,)
"""Every ORCS command cfg that owns a multi-clip library.

MJLab's train/play scripts gate on ``isinstance(cmd, MotionCommandCfg)`` and
otherwise force their single-file ``--motion-file`` path. The compatibility
shim exempts this base class and therefore all task-specific subclasses.

The sentinel is process-global. A downstream package applying its own shim
must include this tuple together with its own command cfgs::

    _apply_mjlab_compat(multi_clip_cfgs=(*orcs.MULTI_CLIP_CFGS, MyCfg))
"""

_apply_mjlab_compat(multi_clip_cfgs=MULTI_CLIP_CFGS)
