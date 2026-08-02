"""PPO runner configs — uolm's agents.

  sonic_agent_cfg()    frozen SONIC base + LoRA adapter on the decoder
  tara_agent_cfg()     tabula rasa MLP, from scratch — the no-frozen-base floor
  sidecar_agent_cfg()  frozen textop WBC + action-residual sidecar

The runner/algo/critic spine and the actor builders are task-blind and live in
:mod:`orcs.core.rl`; this file only names experiments and picks actors. The
underscored aliases below are re-exported because a downstream consumer (vibe)
imports them from here — the spine is shared, the actor is not.
"""

from __future__ import annotations

from mjlab.rl import RslRlOnPolicyRunnerCfg

from orcs.core.rl import CRITIC_HIDDEN as _CRITIC_HIDDEN  # noqa: F401
from orcs.core.rl import DIST_BASE_BAND as _DIST_BASE_BAND  # noqa: F401
from orcs.core.rl import DIST_LEARNABLE as _DIST_LEARNABLE  # noqa: F401
from orcs.core.rl import MAX_ITERATIONS as _MAX_ITERATIONS  # noqa: F401
from orcs.core.rl import NUM_STEPS_PER_ENV as _NUM_STEPS_PER_ENV  # noqa: F401
from orcs.core.rl import SAVE_INTERVAL as _SAVE_INTERVAL  # noqa: F401
from orcs.core.rl import SMPL_CKPT as _SMPL_CKPT  # noqa: F401
from orcs.core.rl import SONIC_CKPT as _SONIC_CKPT  # noqa: F401
from orcs.core.rl import WBC_CKPT as _WBC_CKPT  # noqa: F401
from orcs.core.rl import WBC_HIDDEN as _WBC_HIDDEN  # noqa: F401
from orcs.core.rl import mlp_actor, sidecar_actor, sonic_adapter_actor
from orcs.core.rl import runner as _runner  # noqa: F401 — consumer-facing


def sonic_agent_cfg(
    experiment_name: str = "orcs_uolm",
    *,
    rank: int = 16,
    alpha: float = 1.0,
    base_checkpoint: str = _SONIC_CKPT,
) -> RslRlOnPolicyRunnerCfg:
    """Frozen SONIC base + LoRA adapter on the decoder (augmentation stream)."""
    cfg = _runner(experiment_name)
    cfg.actor = sonic_adapter_actor(  # type: ignore[assignment]
        rank=rank, alpha=alpha, base_checkpoint=base_checkpoint)
    return cfg


def tara_agent_cfg(experiment_name: str = "orcs_uolm_tara") -> RslRlOnPolicyRunnerCfg:
    """From-scratch MLP over the 2-stream layout (uolm_env_cfg(agent="tara")).

    Same width as the frozen WBC so the comparison is architecture-fair: what
    differs is initialization + what is trainable, not capacity.
    """
    cfg = _runner(experiment_name)
    cfg.actor = mlp_actor(_WBC_HIDDEN)
    return cfg


def sidecar_agent_cfg(
    experiment_name: str = "orcs_uolm_sidecar",
    *,
    sidecar_hidden_dims: tuple[int, ...] = (512, 256, 128),
    base_checkpoint: str = _WBC_CKPT,
) -> RslRlOnPolicyRunnerCfg:
    """Frozen WBC + a sidecar that adds a bounded residual to the base action.

    The other way to adapt a frozen base: the adapter perturbs the base's
    INTERNAL weights (LoRA), the sidecar leaves it untouched and corrects its
    OUTPUT. Rides the textop base (`ModularNormMLP` shaped), not SONIC —
    rsl_rl has no sonic sidecar model.
    """
    cfg = _runner(experiment_name)
    cfg.actor = sidecar_actor(  # type: ignore[assignment]
        sidecar_hidden_dims=sidecar_hidden_dims, base_checkpoint=base_checkpoint)
    return cfg
