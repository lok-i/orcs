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

# LoRA size for THIS task's adapter (2026-08-04). rank 16 -> 28, alpha 1 -> 28.
#
# `alpha` is the delta's SCALE and rsl_rl divides it by rank
# (`Adapter.scale = alpha / rank`), so the two move together or not at all:
# raising rank at alpha=1.0 SHRINKS the update. alpha == rank pins scale = 1.0,
# which is what fcrl's working omni-object run had (full-rank, alpha=1.0).
#
# **28 is the ceiling, not a preference.** `Adapter` falls back to a full-rank
# dense delta — scale = alpha, NOT alpha/rank — whenever
# `rank >= min(in_dim, out_dim)`, and two of the decoder's seven layers are
# narrow: the conditioning port is (augmentation=53 -> 2048) and the output is
# (512 -> 29). At rank 64 BOTH silently become full-rank at 64x scale while
# staying bit-exact at construction, so it looks fine and then runs the
# augmentation stream's entry point at 64x gain. min(53, 29) - 1 = 28 is the
# largest rank at which every layer stays low-rank and the scale is uniform.
#
# Which also says the rank was never the bottleneck: 16 of a possible 53 on the
# port is not a severe constraint. The 1/16 SCALE was. This row buys the 16x
# scale and 1.75x rank together; if it moves nothing, adapter capacity is
# excluded and the next suspect is the VOF decay outrunning the learning rate.
#
# Why here and not in `core.rl`: perloco and vibe's repose learn fine at
# scale 1/16 and the core default is task-blind on purpose. uolm is the row
# where the adapter must pull the frozen base OFF its tracking manifold to move
# an object, and the symptom that it cannot is `error_body_pos` pinned at
# 0.047 m while the object never follows (fcrl held 0.45 m and traded it).
_ADAPTER_RANK = 28

# Exploration band (2026-08-04). The frozen std is the one axis where fcrl's
# working run demonstrably differed: it LEARNED std and grew it 1.69x off the
# textop base band (mean 0.265 -> 0.448 peak 0.55). We keep std FIXED (a
# learnable one diffuses the adapter — repose), so the fixed analogue of that
# behavior is a multiplier on the ckpt's converged per-dim vector.
#
# 1.3 lands SONIC's families where fcrl's converged (per-family, both stacks
# share the SAME action scale — mocke's `0.25*effort/stiffness` IS fcrl's rule,
# so sigma is directly comparable and no per-joint correction is warranted):
#
#   family            SONIC   x1.3   fcrl converged (est.)
#   leg               0.350   0.455   ~0.40
#   waist/arm         0.36-0.40  0.46-0.52  ~0.45
#   wrist pitch/yaw   0.500   0.650   ~0.54
#
# Note the wrist pitch/yaw dims sit at 0.500011/0.500010/0.500010/0.500009 —
# a clamp in SONIC's own pretraining, not convergence, so 1.3x there is
# lifting a ceiling rather than exceeding a converged value.
#
# Safe by construction: std is not an input to encoder/FSQ/decoder, so the
# frozen base's mean action and `--agent initial` bit-exactness are untouched.
# Do NOT reach for the action term's `scale` instead — see `sonic_adapter_actor`.
_STD_SCALE = 1.3

# The agents come from `orcs.core.rl` — a task picks one and names its
# experiment, it never declares PPO. See that module's docstring.
_TASKS = (
    # Both levers move together this run — CONFOUNDED by choice: two nights of
    # single-variable rows are not worth spending before either is known to do
    # anything. If it moves, split them; if it does not, both are excluded at
    # once and the next suspect is the VOF decay outrunning the learning rate.
    ("Orcs-Uolm-AdaptSonic",
     partial(uolm_env_cfg),
     partial(adapt_sonic_agent_cfg, "orcs_uolm",
             rank=_ADAPTER_RANK, alpha=_ADAPTER_RANK, std_scale=_STD_SCALE)),
    ("Orcs-Uolm-TaRa",
     partial(uolm_env_cfg, agent="tara"),
     partial(tara_agent_cfg, "orcs_uolm_tara")),
    ("Orcs-Uolm-AdaptSonic-Smpl",
     partial(uolm_env_cfg, command_space="smpl"),
     partial(adapt_sonic_agent_cfg, "orcs_uolm_smpl", base_checkpoint=SMPL_CKPT)),
)

SKIP_REASON: dict[str, str] = register_all(_TASKS)
"""{task_id: why it could not register}. Empty when every task registered."""
