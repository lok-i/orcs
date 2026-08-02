"""Register the PerLoco (Perceptive Locomotion) tasks with mjlab.

Task ids read `Orcs-<Task>-<Agent>` — the AGENT is always named, never a
default hiding in a bare id.

  Orcs-PerLoco-AdaptSonic   frozen SONIC base + LoRA adapter reading a terrain
                            height scan. THE task.
  Orcs-PerLoco-TaRa         tabula rasa from-scratch MLP — the no-frozen-base
                            floor to measure the adapter against.

Both ride the OmniRetarget `robot-terrain` staging. A second source (GRAIL) is
a second `sources/` reader plus a second registration line: the env, the
command and the agents do not change, which is the point of the staging
interface.

Registration needs staged data (`scripts/stage_terrain_motions.py`). Absent, it
is SKIPPED, never raised — an incomplete checkout must not break `import orcs`
for every consumer downstream. A missing task is the signal; `SKIP_REASON` is
the explanation:

    python -c "import orcs; print(orcs.tasks.perloco.SKIP_REASON)"
"""

from mjlab.tasks.registry import register_mjlab_task

from orcs.core.rl import adapt_sonic_agent_cfg, tara_agent_cfg
from orcs.tasks.perloco.env_cfg import perloco_env_cfg

SKIP_REASON: str | None = None
"""Why registration was skipped, or None when every task registered."""

# The agents come from `orcs.core.rl` — a task picks one and names its
# experiment, it never declares PPO. See that module's docstring.
_TASKS = (
    ("Orcs-PerLoco-AdaptSonic", {},
     lambda: adapt_sonic_agent_cfg("orcs_perloco")),
    ("Orcs-PerLoco-TaRa", {"agent": "tara"},
     lambda: tara_agent_cfg("orcs_perloco_tara")),
)

try:
    for _task_id, _kw, _rl in _TASKS:
        register_mjlab_task(
            task_id=_task_id,
            env_cfg=perloco_env_cfg(**_kw),
            play_env_cfg=perloco_env_cfg(**_kw, play=True),
            rl_cfg=_rl(),
        )
except (FileNotFoundError, NotADirectoryError, OSError) as e:
    SKIP_REASON = f"{type(e).__name__}: {e}"
