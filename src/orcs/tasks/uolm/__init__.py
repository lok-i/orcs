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
every consumer downstream.

A missing task is the signal; `SKIP_REASON` is the explanation:

    python -c "import orcs; print(orcs.tasks.uolm.SKIP_REASON)"
"""

from mjlab.tasks.registry import register_mjlab_task

from orcs.core.rl import SMPL_CKPT, adapt_sonic_agent_cfg, tara_agent_cfg
from orcs.tasks.uolm.env_cfg import uolm_env_cfg

SKIP_REASON: str | None = None
"""Why registration was skipped, or None when every task registered."""

# The agents come from `orcs.core.rl` — a task picks one and names its
# experiment, it never declares PPO. See that module's docstring.
_TASKS = (
    ("Orcs-Uolm-AdaptSonic", {},
     lambda: adapt_sonic_agent_cfg("orcs_uolm")),
    ("Orcs-Uolm-TaRa", {"agent": "tara"},
     lambda: tara_agent_cfg("orcs_uolm_tara")),
    ("Orcs-Uolm-AdaptSonic-Smpl", {"command_space": "smpl"},
     lambda: adapt_sonic_agent_cfg("orcs_uolm_smpl", base_checkpoint=SMPL_CKPT)),
)

try:
    for _task_id, _kw, _rl in _TASKS:
        register_mjlab_task(
            task_id=_task_id,
            env_cfg=uolm_env_cfg(**_kw),
            play_env_cfg=uolm_env_cfg(**_kw, play=True),
            rl_cfg=_rl(),
        )
except (FileNotFoundError, NotADirectoryError, OSError) as e:
    SKIP_REASON = f"{type(e).__name__}: {e}"
