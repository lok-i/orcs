"""Register the PerLoco (Perceptive Locomotion) tasks with mjlab.

Task ids read `Orcs-<Task>-<Agent>` — the AGENT is always named, never a
default hiding in a bare id.

  Orcs-PerLoco-AdaptSonic          OmniRetarget climb. THE task.
  Orcs-PerLoco-TaRa                its no-frozen-base floor
  Orcs-PerLoco-AdaptSonic-Grail    GRAIL curb
  Orcs-PerLoco-TaRa-Grail          its floor

One source per task, not one task spanning both — the grids differ in shape
(omni has a z_scale difficulty axis, GRAIL has none). What does NOT differ is
the obs, so a checkpoint crosses between them unchanged.

Registration needs staged data (`scripts/stage_terrain_motions.py`). Absent, it
is SKIPPED, never raised — an incomplete checkout must not break `import orcs`
for every consumer downstream. A missing task is the signal; `SKIP_REASON` is
the explanation:

    python -c "import orcs; print(orcs.tasks.perloco.SKIP_REASON)"
"""

from mjlab.tasks.registry import register_mjlab_task

from orcs.core.rl import adapt_sonic_agent_cfg, tara_agent_cfg
from orcs.tasks.perloco.env_cfg import grail_env_cfg, omni_env_cfg

SKIP_REASON: str | None = None
"""Why registration was skipped, or None when every task registered."""

# The agents come from `orcs.core.rl` — a task picks one and names its
# experiment, it never declares PPO. See that module's docstring.
_TASKS = (
    ("Orcs-PerLoco-AdaptSonic", omni_env_cfg, "sonic", "orcs_perloco"),
    ("Orcs-PerLoco-TaRa", omni_env_cfg, "tara", "orcs_perloco_tara"),
    ("Orcs-PerLoco-AdaptSonic-Grail", grail_env_cfg, "sonic", "orcs_perloco_grail"),
    ("Orcs-PerLoco-TaRa-Grail", grail_env_cfg, "tara", "orcs_perloco_grail_tara"),
)

_AGENT = {"sonic": adapt_sonic_agent_cfg, "tara": tara_agent_cfg}

# Each source registers independently: staged data for one must not block the
# other. SKIP_REASON keeps the last failure.
for _task_id, _env_cfg, _agent, _exp in _TASKS:
    try:
        register_mjlab_task(
            task_id=_task_id,
            env_cfg=_env_cfg(agent=_agent),
            play_env_cfg=_env_cfg(agent=_agent, play=True),
            rl_cfg=_AGENT[_agent](_exp),
        )
    except (FileNotFoundError, NotADirectoryError, OSError) as e:
        SKIP_REASON = f"{_task_id}: {type(e).__name__}: {e}"
