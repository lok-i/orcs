"""Register the Dodge task with mjlab.

  Orcs-Dodge-AdaptSonic    frozen SONIC + LoRA, standing, dodging a thrown ball

Task ids read `Orcs-<Task>-<Agent>` — the AGENT is always named, never a default
hiding in a bare id. There is no `<Source>` slot: one threat model, and a
one-valued axis is not an axis. There is no `-TaRa` floor yet either; the claim
this task makes is about a frozen base, and a from-scratch row belongs with the
run that needs it.

Registration needs the nominal-stand reference (`orcs-make-nominal`). Absent, it
is SKIPPED, never raised — an incomplete checkout must not break `import orcs`
for every consumer downstream:

    python -c "import orcs; print(orcs.tasks.dodge.SKIP_REASON)"
"""

from functools import partial

from orcs.core.registry import register_all
from orcs.core.rl import adapt_sonic_agent_cfg
from orcs.tasks.dodge.env_cfg import dodge_env_cfg

_TASKS = (
    ("Orcs-Dodge-AdaptSonic",
     dodge_env_cfg,
     partial(adapt_sonic_agent_cfg, "orcs_dodge")),
)

SKIP_REASON: dict[str, str] = register_all(_TASKS)
"""{task_id: why it could not register}. Empty when every task registered."""
