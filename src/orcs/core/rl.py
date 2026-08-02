"""The PPO runner spine and the actor builders — task-blind.

Every orcs task trains the same way: adaptive-KL PPO over an asymmetric
actor-critic, `policy` in and `critic` in. What differs per task is the ACTOR,
and what differs per agent is which actor. So:

  _runner(name)          THE runner/algo/critic defaults. One source.
  sonic_adapter_actor()  frozen SONIC base + zero-init LoRA on the decoder
  mlp_actor()            from-scratch MLP — the no-frozen-base floor
  sidecar_actor()        frozen textop WBC + bounded action residual

A task's `rl_cfg.py` is then `_runner(...)` plus one actor call, and a
downstream consumer (vibe) imports the spine rather than re-declaring it.

Nothing here names an object, a terrain, or a dataset.
"""

from __future__ import annotations

from mjlab.rl import RslRlModelCfg, RslRlOnPolicyRunnerCfg, RslRlPpoAlgorithmCfg
from mocke import PRETRAINED_DIR

__all__ = [
    "SONIC_CKPT", "SMPL_CKPT", "WBC_CKPT",
    "CRITIC_HIDDEN", "WBC_HIDDEN",
    "DIST_BASE_BAND", "DIST_LEARNABLE",
    "NUM_STEPS_PER_ENV", "MAX_ITERATIONS", "SAVE_INTERVAL",
    "runner", "sonic_adapter_actor", "mlp_actor", "sidecar_actor",
]

SONIC_CKPT = str(PRETRAINED_DIR / "sonic/last_ported.pt")
WBC_CKPT = str(PRETRAINED_DIR / "textop/model_75000_ported.pt")
# untracked — regenerate: python scripts/port_sonic_checkpoint.py --smpl (in mocke)
SMPL_CKPT = str(PRETRAINED_DIR / "sonic/smpl_ported.pt")

CRITIC_HIDDEN = (512, 256, 128)
WBC_HIDDEN = (2048, 1024, 512)

# Adapter agents: std FROZEN at the base ckpt's converged per-dim values
# (sonic 0.30-0.50). learn_std=False keeps std_param an nn.Parameter
# (requires_grad=False), so the base checkpoint still loads over init_std —
# but PPO can never inflate it. Empirically (2026-07 runs) a learnable std
# blows up to ~1.0 within ~1k updates, wrecking base tracking while the LoRA
# delta_w diffuses (sqrt-t growth) instead of converging.
DIST_BASE_BAND = {
    "class_name": "GaussianDistribution",
    "init_std": 1.0,  # overwritten per-dim by base_checkpoint at load
    "std_type": "scalar",
    "learn_std": False,
}
# From-scratch agents DO learn std — there is no competent base band to stay
# inside, and action-space exploration is the whole job.
DIST_LEARNABLE = {
    "class_name": "GaussianDistribution",
    "init_std": 1.0,
    "std_type": "scalar",
}

NUM_STEPS_PER_ENV = 24
MAX_ITERATIONS = 60_000
SAVE_INTERVAL = 1500


def runner(experiment_name: str) -> RslRlOnPolicyRunnerCfg:
    """Common runner cfg: adaptive-KL PPO (byte-identical to mjlab's stock G1
    tracking algo) + MLP critic. The caller sets `.actor`."""
    return RslRlOnPolicyRunnerCfg(
        experiment_name=experiment_name,
        num_steps_per_env=NUM_STEPS_PER_ENV,
        max_iterations=MAX_ITERATIONS,
        save_interval=SAVE_INTERVAL,
        obs_groups={"actor": ("policy",), "critic": ("critic",)},
        critic=RslRlModelCfg(
            hidden_dims=CRITIC_HIDDEN,
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


def sonic_adapter_actor(
    *,
    rank: int = 16,
    alpha: float = 1.0,
    base_checkpoint: str = SONIC_CKPT,
    adapter_obs_group: str = "augmentation",
) -> dict:
    """Frozen SONIC base + zero-init LoRA adapter on the decoder.

    Construction reproduces the base bit-exact; PPO moves only the adapter.
    `adapter_obs_group` is the task's conditioning stream — THE extension seam.
    """
    return {
        "class_name": "rsl_rl.models.SonicWithAdapterModel",
        "distribution_cfg": DIST_BASE_BAND,
        "adapter_obs_group": adapter_obs_group,
        "rank": rank,
        "alpha": alpha,
        "base_checkpoint": base_checkpoint,
        "freeze_base": True,
    }


def mlp_actor(hidden_dims: tuple[int, ...] = WBC_HIDDEN) -> RslRlModelCfg:
    """From-scratch MLP, WBC width so the comparison is architecture-fair:
    what differs from the adapter row is initialization and what is trainable,
    not capacity."""
    return RslRlModelCfg(
        hidden_dims=hidden_dims,
        obs_normalization=True,
        activation="elu",
        distribution_cfg=DIST_LEARNABLE,
    )


def sidecar_actor(
    *,
    sidecar_hidden_dims: tuple[int, ...] = (512, 256, 128),
    base_checkpoint: str = WBC_CKPT,
    sidecar_obs_group: str = "augmentation",
) -> dict:
    """Frozen WBC + a sidecar that adds a bounded residual to the base action.

    The other way to adapt a frozen base: the adapter perturbs the base's
    INTERNAL weights (LoRA), the sidecar leaves it untouched and corrects its
    OUTPUT. Rides the textop base (`ModularNormMLP` shaped), not SONIC —
    rsl_rl has no sonic sidecar model.
    """
    return {
        "class_name": "rsl_rl.models.ModularNormMLPWithSidecarModel",
        "hidden_dims": list(WBC_HIDDEN),
        "activation": "elu",
        "obs_normalization": True,
        "distribution_cfg": DIST_LEARNABLE,
        "sidecar_obs_group": sidecar_obs_group,
        "sidecar_hidden_dims": list(sidecar_hidden_dims),
        "sidecar_activation": "elu",
        "sidecar_output_scale": 1.0,
        "sidecar_output_bound": "tanh",
        "condition_on_base_output": True,
        "base_checkpoint": base_checkpoint,
        "freeze_base": True,
    }
