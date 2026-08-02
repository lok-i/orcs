"""perloco mdp — core's task-blind term library plus the terrain-keyed command.

Short by design. PerLoco tracks a robot over terrain, and every term that
describes a robot tracking a reference is already task-blind and lives in
`orcs.core.mdp`. The only thing perloco owns is WHICH clips an env may sample,
which is the command's business — so that is the only local module.

The height scan is not here either: it is an mjlab sensor read through mjlab's
own `height_scan` term. A terrain term would only be needed for something the
sensor cannot express.
"""

from orcs.core.mdp import *  # noqa: F401, F403

from .commands import *  # noqa: F403
