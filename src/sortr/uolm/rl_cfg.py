"""PPO runner config — ONE agent: frozen SONIC base + LoRA adapter.

`_runner()` is THE single source of runner/algo/critic defaults (iterations,
PPO hyperparams, obs routing); `sonic_agent_cfg()` sets only the actor.
"""

from __future__ import annotations

from mjlab.rl import RslRlModelCfg, RslRlOnPolicyRunnerCfg, RslRlPpoAlgorithmCfg
from mocke import PRETRAINED_DIR

_SONIC_CKPT = str(PRETRAINED_DIR / "sonic/last_ported.pt")
# untracked — regenerate: python scripts/port_sonic_checkpoint.py --smpl (in mocke)
_SMPL_CKPT = str(PRETRAINED_DIR / "sonic/smpl_ported.pt")

_CRITIC_HIDDEN = (512, 256, 128)

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

_NUM_STEPS_PER_ENV = 24
_MAX_ITERATIONS = 60_000
_SAVE_INTERVAL = 1000


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
    experiment_name: str = "sortr_uolm",
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
