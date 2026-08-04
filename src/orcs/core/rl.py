"""THE agent zoo — task-blind. Every orcs agent is generated from here.

Every orcs task trains the same way: adaptive-KL PPO over an asymmetric
actor-critic, `policy` in and `critic` in. What differs per task is the ACTOR,
and what differs per agent is which actor. Two tiers:

  runner(name)              THE runner/algo/critic defaults. One source.
  *_actor()                 the actor dicts — composable parts

  adapt_sonic_agent_cfg()   frozen SONIC base + zero-init LoRA on the decoder
  tara_agent_cfg()          from-scratch MLP — the no-frozen-base floor
  sidecar_agent_cfg()       frozen textop WBC + bounded action residual

**A task does not own an agent.** It picks one by name and passes an
experiment name — `adapt_sonic_agent_cfg("orcs_perloco")` is the whole of a
task's RL config. That is why there is no `rl_cfg.py` under any task: a second
place to spell "PPO" is a second place for the two to drift.

A task that genuinely needs a different actor adds a generator HERE (if it is
task-blind) or overrides `.actor` on the returned cfg (if it is not). Consumers
outside orcs do the same — import the generator, override one field.

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
    "adapt_sonic_agent_cfg", "tara_agent_cfg", "sidecar_agent_cfg",
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
    std_scale: float | dict[str, float] = 1.0,
) -> dict:
    """Frozen SONIC base + zero-init LoRA adapter on the decoder.

    Construction reproduces the base bit-exact; PPO moves only the adapter.
    `adapter_obs_group` is the task's conditioning stream — THE extension seam.

    ⚠ `alpha` is the delta's SCALE and rsl_rl divides it by rank
    (`Adapter.scale = alpha / rank`), so the two are NOT independent: raising
    rank at fixed alpha SHRINKS the update. Pass `alpha=rank` to hold the scale
    at 1.0. The defaults below are scale 1/16 — every task that has trained on
    them is calibrated to that, so change them per-task, not here.

    `std_scale` multiplies the ckpt's per-dim `action_std` (scalar, or
    {joint-name regex: factor}). It is the ONLY sanctioned way to buy
    exploration off a frozen base: std never enters encoder/FSQ/decoder, so the
    base's mean action and construction bit-exactness are untouched. The action
    term's `scale` is NOT an alternative — the base's output is calibrated to
    it, and on the wrists it is actuator-bound (5 Nm / kp) anyway, so a larger
    target only saturates.
    """
    return {
        "class_name": "rsl_rl.models.SonicWithAdapterModel",
        "distribution_cfg": dict(DIST_BASE_BAND),
        "adapter_obs_group": adapter_obs_group,
        "rank": rank,
        "alpha": alpha,
        "base_checkpoint": base_checkpoint,
        "freeze_base": True,
        # Omitted at 1.0 (the no-op) so every task that does not ask for it
        # keeps a byte-identical actor cfg — and a byte-identical wandb config.
        **({"std_scale": std_scale} if std_scale != 1.0 else {}),
    }


def mlp_actor(hidden_dims: tuple[int, ...] = WBC_HIDDEN) -> RslRlModelCfg:
    """From-scratch MLP, WBC width so the comparison is architecture-fair:
    what differs from the adapter row is initialization and what is trainable,
    not capacity."""
    return RslRlModelCfg(
        hidden_dims=hidden_dims,
        obs_normalization=True,
        activation="elu",
        distribution_cfg=dict(DIST_LEARNABLE),
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
        "distribution_cfg": dict(DIST_LEARNABLE),
        "sidecar_obs_group": sidecar_obs_group,
        "sidecar_hidden_dims": list(sidecar_hidden_dims),
        "sidecar_activation": "elu",
        "sidecar_output_scale": 1.0,
        "sidecar_output_bound": "tanh",
        "condition_on_base_output": True,
        "base_checkpoint": base_checkpoint,
        "freeze_base": True,
    }


# ---------------------------------------------------------------------------
# The agent generators — what a task actually calls
# ---------------------------------------------------------------------------

def adapt_sonic_agent_cfg(
    experiment_name: str,
    *,
    rank: int = 16,
    alpha: float = 1.0,
    base_checkpoint: str = SONIC_CKPT,
    adapter_obs_group: str = "augmentation",
    std_scale: float | dict[str, float] = 1.0,
) -> RslRlOnPolicyRunnerCfg:
    """AdaptSonic — frozen SONIC base + LoRA adapter on the decoder. THE agent.

    Task-blind by construction: the ONLY thing a task varies is what rides
    `adapter_obs_group`. uolm feeds it object kinematics, perloco a height
    scan, a vision consumer image features — the agent is the same bytes.
    """
    cfg = runner(experiment_name)
    cfg.actor = sonic_adapter_actor(  # type: ignore[assignment]
        rank=rank, alpha=alpha, base_checkpoint=base_checkpoint,
        adapter_obs_group=adapter_obs_group, std_scale=std_scale)
    return cfg


def tara_agent_cfg(
    experiment_name: str, *, hidden_dims: tuple[int, ...] = WBC_HIDDEN
) -> RslRlOnPolicyRunnerCfg:
    """TaRa — tabula rasa, from scratch. The no-frozen-base floor.

    WBC width so the comparison against AdaptSonic is architecture-fair: what
    differs is initialization and what is trainable, not capacity. Pair it with
    a 2-stream obs layout — with no frozen base there is no tokenizer stream,
    so the actor must be handed the motion reference directly.
    """
    cfg = runner(experiment_name)
    cfg.actor = mlp_actor(hidden_dims)
    return cfg


def sidecar_agent_cfg(
    experiment_name: str,
    *,
    sidecar_hidden_dims: tuple[int, ...] = (512, 256, 128),
    base_checkpoint: str = WBC_CKPT,
    sidecar_obs_group: str = "augmentation",
) -> RslRlOnPolicyRunnerCfg:
    """Sidecar — the OTHER way to adapt a frozen base.

    The adapter perturbs the base's INTERNAL weights (LoRA); the sidecar leaves
    it untouched and corrects its OUTPUT. Rides the textop base
    (`ModularNormMLP` shaped), not SONIC — rsl_rl has no sonic sidecar model.
    """
    cfg = runner(experiment_name)
    cfg.actor = sidecar_actor(  # type: ignore[assignment]
        sidecar_hidden_dims=sidecar_hidden_dims,
        base_checkpoint=base_checkpoint,
        sidecar_obs_group=sidecar_obs_group)
    return cfg
