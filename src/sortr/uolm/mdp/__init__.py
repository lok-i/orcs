"""sortr mdp — THE object-manipulation term library (SONIC-only, fcrl lineage).

Aggregates mjlab's stock mdp + every local object-manip term
(obs/rewards/terminations/curriculums/events/commands) so ``mdp.<name>``
resolves all of them.
"""

from mjlab.envs.mdp import *  # noqa: F401, F403

from .commands_omni_object import *  # noqa: F403
from .contact_schedule import ContactSchedule  # noqa: F401
from .curriculums import *  # noqa: F403
from .demo_loader import (  # noqa: F401
    get_motion_files_for_objects,
    load_field_or_make_zeros,
    load_motion_files_from_datasets,
)
from .events import *  # noqa: F403
from .metrics import *  # noqa: F403
from .observations import *  # noqa: F403
from .rewards import *  # noqa: F403
from .terminations import *  # noqa: F403
