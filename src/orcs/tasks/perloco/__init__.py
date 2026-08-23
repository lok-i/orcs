"""Register the PerLoco (Perceptive Locomotion) tasks with mjlab.

Task ids read `Orcs-<Task>-<Source>-<Agent>` — the AGENT is always named, never
a default hiding in a bare id.

  Orcs-PerLoco-OmRe-AdaptSonic         OmniRetarget climb. THE task.
  Orcs-PerLoco-OmRe-TaRa               its no-frozen-base floor
  Orcs-PerLoco-Grail-AdaptSonic        GRAIL curb
  Orcs-PerLoco-Grail-AdaptSonic-Smpl   ...reading the HUMAN, not the retarget
  Orcs-PerLoco-Grail-TaRa              its floor

`-Smpl` is the SONIC retargeting task: the SMPL-X reconstruction drives the
frozen SMPL encoder and all-point tracking rewards, while a phase-1
`seed_state.npz` supplies RSI only.  It never tracks or initializes from the
GRAIL robot retarget.  Its grid contains the seed-complete subset of the GRAIL
roster, so preprocessing can be resumed sample by sample; without any complete
SMPL+seed pair this row skips and the other four still register.

One source per task, not one task spanning both — the grids differ in shape
(OmRe has a z_scale difficulty axis, GRAIL has none). What does NOT differ is
the obs, so a checkpoint crosses between them unchanged.

The SOURCE is in the id because provenance is what changes code: reader, file
format, joint order, conventions. The TERRAIN TYPE is not — curb and stair are
the same reader and the same env, so they are a roster line, not a task.

Registration needs staged data (`scripts/setup/perceptive_locomotion.sh`).
Absent, it is SKIPPED, never raised — an incomplete checkout must not break
`import orcs` for every consumer downstream, and each row registers on its own
(the `-Smpl` row needs `--smpl` staging plus seed preprocessing). A missing task is
the signal; `SKIP_REASON` is the explanation:

    python -c "import orcs; print(orcs.tasks.perloco.SKIP_REASON)"
"""

from functools import partial

from orcs.core.registry import register_all
from orcs.core.rl import SMPL_CKPT, adapt_sonic_agent_cfg, tara_agent_cfg
from orcs.tasks.perloco.env_cfg import grail_env_cfg, grail_smpl_env_cfg, omni_env_cfg

# The agents come from `orcs.core.rl` — a task picks one and names its
# experiment, it never declares PPO. See that module's docstring. Both
# factories arrive pre-bound, so this table has no per-task branch to grow.
_smpl_sonic = partial(adapt_sonic_agent_cfg, base_checkpoint=SMPL_CKPT)

_TASKS = (
    ("Orcs-PerLoco-OmRe-AdaptSonic",
     partial(omni_env_cfg, agent="sonic"),
     partial(adapt_sonic_agent_cfg, "orcs_perloco_omre")),
    ("Orcs-PerLoco-OmRe-TaRa",
     partial(omni_env_cfg, agent="tara"),
     partial(tara_agent_cfg, "orcs_perloco_omre_tara")),
    ("Orcs-PerLoco-Grail-AdaptSonic",
     partial(grail_env_cfg, agent="sonic"),
     partial(adapt_sonic_agent_cfg, "orcs_perloco_grail")),
    ("Orcs-PerLoco-Grail-AdaptSonic-Smpl",
     grail_smpl_env_cfg,
     partial(_smpl_sonic, "orcs_perloco_grail_smpl")),
    ("Orcs-PerLoco-Grail-TaRa",
     partial(grail_env_cfg, agent="tara"),
     partial(tara_agent_cfg, "orcs_perloco_grail_tara")),
)

SKIP_REASON: dict[str, str] = register_all(_TASKS)
"""{task_id: why it could not register}. Empty when every task registered."""
