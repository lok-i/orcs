"""Register the PerLoco (Perceptive Locomotion) tasks with mjlab.

Task ids read `Orcs-<Task>-<Source>-<Agent>` — the AGENT is always named, never
a default hiding in a bare id.

  Orcs-PerLoco-OmRe-AdaptSonic    OmniRetarget climb. THE task.
  Orcs-PerLoco-OmRe-TaRa          its no-frozen-base floor
  Orcs-PerLoco-Grail-AdaptSonic   GRAIL curb
  Orcs-PerLoco-Grail-TaRa         its floor

One source per task, not one task spanning both — the grids differ in shape
(OmRe has a z_scale difficulty axis, GRAIL has none). What does NOT differ is
the obs, so a checkpoint crosses between them unchanged.

The SOURCE is in the id because provenance is what changes code: reader, file
format, joint order, conventions. The TERRAIN TYPE is not — curb and stair are
the same reader and the same env, so they are a roster line, not a task.

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
    ("Orcs-PerLoco-OmRe-AdaptSonic", omni_env_cfg, "sonic", "orcs_perloco_omre"),
    ("Orcs-PerLoco-OmRe-TaRa", omni_env_cfg, "tara", "orcs_perloco_omre_tara"),
    ("Orcs-PerLoco-Grail-AdaptSonic", grail_env_cfg, "sonic", "orcs_perloco_grail"),
    ("Orcs-PerLoco-Grail-TaRa", grail_env_cfg, "tara", "orcs_perloco_grail_tara"),
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
