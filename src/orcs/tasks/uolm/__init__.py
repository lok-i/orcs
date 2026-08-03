"""Register the UOLM (Uni-Object Loco-Manipulation) tasks with mjlab.

Task ids read `Orcs-<Task>-<Agent>[-<CommandSpace>]`. The AGENT is always named —
a default agent hiding in a bare id is how the suffix slot ends up meaning two
different things — and the command space is an optional suffix that defaults to
the native robot one.

  Orcs-Uolm-AdaptSonic        frozen SONIC base + LoRA adapter. THE task.
  Orcs-Uolm-TaRa             tabula rasa from-scratch MLP — the no-frozen-base
                             floor to measure the adapter against.
  Orcs-Uolm-AdaptSonic-Smpl   same agent, human SMPL command space (SONIC smpl
                             encoder). Rollout-only: rewards + RSI are nullified
                             in the env cfg (PR pending), so `train` on it is
                             meaningless — use scripts/rollout_smpl.py.

Importing orcs is SILENT and never raises. `orcs.core.paths` resolves data and
assets against the nearest repo root that HAS them, so a host project vendoring
orcs under `dependencies/` is found automatically — no env vars, no import-order
coupling. When the data genuinely is absent (a fresh checkout before
`sync_dependencies.sh` / `make_object_models.py`), registration is skipped
rather than raising: an incomplete checkout must not break `import orcs` for
every consumer downstream. Each row registers on its own, so one unstaged
dataset costs one task, not three.

A missing task is the signal; `SKIP_REASON` is the explanation:

    python -c "import orcs; print(orcs.tasks.uolm.SKIP_REASON)"
"""

from functools import partial

from orcs.core.registry import register_all
from orcs.core.rl import SMPL_CKPT, adapt_sonic_agent_cfg, tara_agent_cfg
from orcs.tasks.uolm.env_cfg import uolm_env_cfg

# The agents come from `orcs.core.rl` — a task picks one and names its
# experiment, it never declares PPO. See that module's docstring.
_TASKS = (
    ("Orcs-Uolm-AdaptSonic",
     partial(uolm_env_cfg),
     partial(adapt_sonic_agent_cfg, "orcs_uolm")),
    ("Orcs-Uolm-TaRa",
     partial(uolm_env_cfg, agent="tara"),
     partial(tara_agent_cfg, "orcs_uolm_tara")),
    ("Orcs-Uolm-AdaptSonic-Smpl",
     partial(uolm_env_cfg, command_space="smpl"),
     partial(adapt_sonic_agent_cfg, "orcs_uolm_smpl", base_checkpoint=SMPL_CKPT)),
)

SKIP_REASON: dict[str, str] = register_all(_TASKS)
"""{task_id: why it could not register}. Empty when every task registered."""
