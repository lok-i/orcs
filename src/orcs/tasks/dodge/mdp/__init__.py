"""dodge mdp — core's task-blind terms plus the ones that mention a ball.

`from orcs.core.mdp import *` first, so `mdp.<name>` resolves mjlab's stock
library, orcs's robot/clip terms, and dodge's own from one namespace.

There is no `commands.py` and no `terminations.py`: the reference is a nominal
stand clip loaded by the stock `MultiClipMotionCommand`, and both terminations
(hit, fall) are mjlab's `illegal_contact` over two different contact sensors.
"""

from orcs.core.mdp import *  # noqa: F401, F403
from orcs.tasks.dodge.mdp.events import *  # noqa: F403
from orcs.tasks.dodge.mdp.observations import *  # noqa: F403
from orcs.tasks.dodge.mdp.rewards import *  # noqa: F403
