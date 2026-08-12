"""core mdp — mjlab's stock term library plus every orcs term that is task-blind.

A task's own ``mdp/__init__.py`` does ``from orcs.core.mdp import *`` and adds
its own terms on top, so ``mdp.<name>`` resolves both.

The line: a term belongs here if it mentions only the ROBOT, a CLIP, or a
REFERENCE. The moment it mentions an object or a terrain, it belongs to the
task that has one.
"""

from mjlab.envs.mdp import *  # noqa: F401, F403

from orcs.core.mdp.events import *  # noqa: F403
from orcs.core.mdp.observations import *  # noqa: F403
from orcs.core.mdp.rewards import *  # noqa: F403
from orcs.core.mdp.terminations import *  # noqa: F403
