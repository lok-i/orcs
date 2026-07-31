"""PPO runner configs — the privileged-agent spine.

  sonic_agent_cfg()    frozen SONIC base + LoRA adapter on the decoder
  tara_agent_cfg()     tabula rasa MLP, from scratch — the no-frozen-base floor
  sidecar_agent_cfg()  frozen textop WBC + action-residual sidecar

`_runner()` is THE single source of runner/algo/critic defaults (iterations, PPO
hyperparams, obs routing); every factory sets only its actor. A downstream
consumer (vibe) imports `_runner` and the distribution dicts rather than
re-declaring them — the spine is shared, the actor is not.
"""

from __future__ import annotations

from mjlab.rl import RslRlModelCfg, RslRlOnPolicyRunnerCfg, RslRlPpoAlgorithmCfg
from mocke import PRETRAINED_DIR

_SONIC_CKPT = str(PRETRAINED_DIR / "sonic/last_ported.pt")
_WBC_CKPT = str(PRETRAINED_DIR / "textop/model_75000_ported.pt")
# untracked — regenerate: python scripts/port_sonic_checkpoint.py --smpl (in mocke)
_SMPL_CKPT = str(PRETRAINED_DIR / "sonic/smpl_ported.pt")

_CRITIC_HIDDEN = (512, 256, 128)
_WBC_HIDDEN = (2048, 1024, 512)

# Adapter agents: std FROZEN at the base ckpt's converged per-dim values
# (sonic 0.30-0.50). learn_std=False keeps std_param an nn.Parameter
# (requires_grad=False), so the base checkpoint still loads over init_std —
# but PPO can never inflate it. Empirically (2026-07 runs) a learnable std
# blows up to ~1.0 within ~1k updates, wrecking base tracking while the LoRA
# delta_w diffuses (sqrt-t growth) instead of converging.
_DIST_BASE_BAND = {
    "class_name": "GaussianDistribution",
    "init_std": 1.0,  # overwritten per-dim by base_checkpoint at load
    "std_type": "scalar",
    "learn_std": False,
}
# From-scratch agents DO learn std — there is no competent base band to stay
# inside, and action-space exploration is the whole job.
_DIST_LEARNABLE = {
    "class_name": "GaussianDistribution",
    "init_std": 1.0,
    "std_type": "scalar",
}

_NUM_STEPS_PER_ENV = 24
_MAX_ITERATIONS = 60_000
_SAVE_INTERVAL = 1500


def _runner(experiment_name: str) -> RslRlOnPolicyRunnerCfg:
    """Common runner cfg: adaptive-KL PPO (byte-identical to mjlab's stock G1
    tracking algo) + MLP critic."""
    return RslRlOnPolicyRunnerCfg(
        experiment_name=experiment_name,
        num_steps_per_env=_NUM_STEPS_PER_ENV,
        max_iterations=_MAX_ITERATIONS,
        save_interval=_SAVE_INTERVAL,
        obs_groups={"actor": ("policy",), "critic": ("critic",)},
        critic=RslRlModelCfg(
            hidden_dims=_CRITIC_HIDDEN,
            obs_normalization=True,
            activation="elu",
        ),
        algorithm=RslRlPpoAlgorithmCfg(
            clip_param=0.2,
            entropy_coef=0.005,
            learning_rate=1e-3,
            schedule="adaptive",
            gamma=0.99,
            lam=0.95,
            desired_kl=0.01,
            max_grad_norm=1.0,
        ),
    )


def sonic_agent_cfg(
    experiment_name: str = "orcs_uolm",
    *,
    rank: int = 16,
    alpha: float = 1.0,
    base_checkpoint: str = _SONIC_CKPT,
) -> RslRlOnPolicyRunnerCfg:
    """Frozen SONIC base + LoRA adapter on the decoder (augmentation stream)."""
    cfg = _runner(experiment_name)
    cfg.actor = {  # type: ignore[assignment]
        "class_name": "rsl_rl.models.SonicWithAdapterModel",
        "distribution_cfg": _DIST_BASE_BAND,
        "adapter_obs_group": "augmentation",
        "rank": rank,
        "alpha": alpha,
        "base_checkpoint": base_checkpoint,
        "freeze_base": True,
    }
    return cfg


# ---------------------------------------------------------------------------
# TaRa — tabula rasa, from-scratch MLP (the no-frozen-base floor)
# ---------------------------------------------------------------------------

def tara_agent_cfg(experiment_name: str = "orcs_uolm_tara") -> RslRlOnPolicyRunnerCfg:
    """From-scratch MLP over the 2-stream layout (uolm_env_cfg(agent="tara")).

    Same width as the frozen WBC so the comparison is architecture-fair: what
    differs is initialization + what is trainable, not capacity.
    """
    cfg = _runner(experiment_name)
    cfg.actor = RslRlModelCfg(
        hidden_dims=_WBC_HIDDEN,
        obs_normalization=True,
        activation="elu",
        distribution_cfg=_DIST_LEARNABLE,
    )
    return cfg


# ---------------------------------------------------------------------------
# Sidecar — frozen textop WBC + action residual
# ---------------------------------------------------------------------------

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
    cfg.actor = {  # type: ignore[assignment]
        "class_name": "rsl_rl.models.ModularNormMLPWithSidecarModel",
        "hidden_dims": list(_WBC_HIDDEN),
        "activation": "elu",
        "obs_normalization": True,
        "distribution_cfg": _DIST_LEARNABLE,
        "sidecar_obs_group": "augmentation",
        "sidecar_hidden_dims": list(sidecar_hidden_dims),
        "sidecar_activation": "elu",
        "sidecar_output_scale": 1.0,
        "sidecar_output_bound": "tanh",
        "condition_on_base_output": True,
        "base_checkpoint": base_checkpoint,
        "freeze_base": True,
    }
    return cfg
