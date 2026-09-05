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

import dataclasses

from mjlab.rl import (
    RslRlBaseRunnerCfg,
    RslRlModelCfg,
    RslRlOnPolicyRunnerCfg,
    RslRlPpoAlgorithmCfg,
)
from mocke import PRETRAINED_DIR

__all__ = [
    "SONIC_CKPT", "SMPL_CKPT", "WBC_CKPT",
    "CRITIC_HIDDEN", "WBC_HIDDEN",
    "DIST_BASE_BAND", "DIST_LEARNABLE",
    "NUM_STEPS_PER_ENV", "MAX_ITERATIONS", "SAVE_INTERVAL",
    "runner", "sonic_adapter_actor", "mlp_actor", "sidecar_actor",
    "adapt_sonic_agent_cfg", "tara_agent_cfg", "sidecar_agent_cfg",
    "DistillAlgorithmCfg", "DistillRunnerCfg",
    "distill_agent_cfg", "priv_distill_agent_cfg",
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


# ---------------------------------------------------------------------------
# Distillation — the SECOND tier: a gradient that comes from a teacher, not
# from returns. Everything above trains with PPO; nothing below declares one.
# ---------------------------------------------------------------------------


@dataclasses.dataclass
class DistillAlgorithmCfg:
    """Online behavior cloning (`rsl_rl.algorithms.Distillation`), BC only.

    The student rolls out and the teacher labels the SAME states, so this is DAgger
    by construction rather than offline BC — there is no fixed dataset, and the
    covariate shift the student creates is exactly what gets labelled.

    `gradient_length=1` is not the upstream default (15) and is the one number here
    that must not be raised carelessly: the storage generator hands out one stored
    timestep at a time, so N accumulates N live graphs before stepping. With 1, an
    iteration is `num_steps_per_env` gradient steps over full-width batches (24 vs
    PPO's 5 epochs x 4 minibatches = 20), which is the parity that makes a `-Bcd`
    row readable against its PPO twin.

    `loss_type` stays rsl_rl's `mse` — unweighted mean over action dims. `"kl"` is
    the sigma-weighted alternative (measured span 2.84x across SONIC's joints), and
    it is BOTH KL directions at once: student and teacher share one FROZEN sigma, so
    forward and reverse collapse to the same quadratic in the means. It does not
    move the optimum, only the gradient allocation — a variable to test, not a
    default to assume. `ZDistill/gap_sigma` reports the residual in sigma units
    regardless of which is chosen.
    """

    class_name: str = "rsl_rl.algorithms.Distillation"
    num_learning_epochs: int = 1
    gradient_length: int = 1
    learning_rate: float = 1.0e-3
    """Fixed. PPO's adaptive-KL schedule needs an old policy to measure against;
    BC has none, so this is PPO's starting LR held constant."""
    max_grad_norm: float = 1.0
    loss_type: str = "mse"
    kl_direction: str = "reverse"
    """"forward" = KL(T||S) mode-covering, "reverse" = KL(S||T) mode-seeking. Inert
    while student and teacher share one FROZEN sigma (measured element-wise identical
    on SONIC): the log and trace terms cancel and both directions are the same
    quadratic in the means. It bites only if a learnable std or a different
    `std_scale` ever puts the two bands apart — exactly when a silently hard-coded
    direction would have been wrong."""
    optimizer: str = "adam"


@dataclasses.dataclass
class DistillRunnerCfg(RslRlBaseRunnerCfg):
    """Student/teacher runner cfg — mjlab's BASE runner cfg, not the on-policy one.

    `RslRlOnPolicyRunnerCfg` adds `actor`/`critic`; a distillation run has neither
    (`Distillation.construct_algorithm` reads `student`/`teacher`), and carrying two
    dead fields into every dumped `agent.yaml` is how a reader learns to stop
    trusting the config. mjlab splits the base out for exactly this, and its train
    script types `TrainConfig.agent` as the base.

    Declares no amp / compile / nan-guard knob. rsl_rl reads all three with a
    `cfg.get(...)` default, so absent means `amp_dtype=None` and
    `torch_compile_mode=None` (both inert, and autocasting a proprio MLP buys
    nothing — `Distillation.set_amp` touches the student only) while
    `check_for_nan` stays at its `True` default. A consumer that wants them
    declared mixes its own knobs in; vibe does, because a vision student is where
    bf16 is worth 1.3x.
    """

    student: dict = dataclasses.field(default_factory=dict)
    teacher: dict = dataclasses.field(default_factory=dict)
    algorithm: DistillAlgorithmCfg = dataclasses.field(default_factory=DistillAlgorithmCfg)


def distill_agent_cfg(
    experiment_name: str,
    *,
    student: dict,
    teacher: dict,
    obs_groups: dict | None = None,
    **algorithm_kw,
) -> DistillRunnerCfg:
    """Frozen teacher -> student, behavior cloning only.

    `student` and `teacher` are actor dicts, and the pairing IS the experiment: pass
    the PPO row's actor verbatim as the student, so the only difference between a
    `-Bcd` row and its PPO twin is where the gradient comes from.

    No teacher path is declared here. The teacher is whatever checkpoint gets loaded
    over it — `train --agent.resume True --wandb-run-path <run>` routes an RL
    checkpoint's `actor_state_dict` into the teacher and nothing else
    (`Distillation.load`), and `DistillRunner` refuses to train without one.

    Both models read `policy` as their proprio stream; their adapter and any
    extractor groups are read BY NAME off the observation dict, so `obs_groups`
    never mentions them.
    """
    return DistillRunnerCfg(
        experiment_name=experiment_name,
        num_steps_per_env=NUM_STEPS_PER_ENV,
        max_iterations=MAX_ITERATIONS,
        save_interval=SAVE_INTERVAL,
        obs_groups=obs_groups or {"student": ("policy",), "teacher": ("policy",)},
        student=student,
        teacher=teacher,
        algorithm=DistillAlgorithmCfg(**algorithm_kw),
    )


def priv_distill_agent_cfg(
    experiment_name: str,
    *,
    rank: int = 16,
    alpha: float = 1.0,
    base_checkpoint: str = SONIC_CKPT,
    adapter_obs_group: str = "augmentation",
    std_scale: float | dict[str, float] = 1.0,
    **algorithm_kw,
) -> DistillRunnerCfg:
    """The IDENTIFIABILITY control: a PRIVILEGED student cloned from its own teacher.

    Student and teacher are the same architecture reading the same obs group, so the
    student can represent the teacher EXACTLY — its LoRA has a target it can reach to
    machine precision. That makes the BC loss a test of the machinery and nothing
    else:

        loss -> ~0   the distillation path (storage layout, teacher load, optimizer,
                     gradient flow) is sound, and a vision student's plateau is then
                     about what a camera can and cannot resolve.
        loss flat    the plumbing is broken, and no amount of extractor tuning on the
                     exteroceptive row will find it.

    ⚠ The actor kwargs must MATCH THE TEACHER'S. They default to
    `sonic_adapter_actor`'s (rank 16, alpha 1.0), which is what perloco trains with;
    uolm passes rank 28 / alpha 28 / `std_scale` 1.3. Mismatch them and the student
    cannot represent the teacher, which is the one thing this row exists to rule out.
    """
    actor = dict(rank=rank, alpha=alpha, base_checkpoint=base_checkpoint,
                 adapter_obs_group=adapter_obs_group, std_scale=std_scale)
    return distill_agent_cfg(
        experiment_name,
        student=sonic_adapter_actor(**actor),
        teacher=sonic_adapter_actor(**actor),
        **algorithm_kw,
    )
