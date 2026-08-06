"""observation_cfgs.py — THE dodge signal library.

Read this against `orcs.tasks.{uolm,perloco}.observation_cfgs`: all three differ
in ONE group, and that is the claim these tasks exist to test.

    [base]      policy + tokenizer streams   — mocke (frozen SONIC contract)
    [adapter]   AUGMENTATION                 — uolm:    object kinematics
                                               perloco: terrain height scan
                                               dodge:   BALL kinematics
    [critic]    privileged full state        — same shape, different extras

Everything else — proprio, the sys1 root-twist command, the reward-vector
conditioning — is `orcs.core.obs`, shared verbatim.

What dodge does NOT carry, and why:

  `bodywise_contact_cmd`  uolm's per-body contact schedule comes from an object
                          contact graph. A stand has none staged, so the slot
                          stays empty rather than get a zero tensor that looks
                          like data.
  a task goal             the ball is the task. There is nothing to command:
                          `robot_motion_cmd_terms` is a CONSTANT here (the
                          reference is a held stand), which is exactly why a
                          visual consumer spends no attention row on it.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from mjlab.managers.observation_manager import ObservationGroupCfg
from mocke.sonic import profile

from orcs.core.obs import (
    T as _T,
)
from orcs.core.obs import (
    grp as _grp,
)
from orcs.core.obs import (
    proprio_terms,
    robot_root_state_terms,
    robot_root_twist_cmd_terms,
)
from orcs.tasks.dodge import mdp

__all__ = ["ObsCtx", "ball_state_terms", "sonic_obs"]


@dataclass(frozen=True)
class ObsCtx:
    """Assembly context threaded through the group factories (uniform signature)."""

    p: dict = field(default_factory=lambda: {"command_name": "motion"})


# ---------------------------------------------------------------------------
# Atomic term bundles
# ---------------------------------------------------------------------------

def ball_state_terms() -> dict:
    """THE exteroception: where the ball is and how fast it is closing. 6 dims.

    Yaw frame, so it is the same frame a head camera measures in — the visual
    row swaps modality, not geometry. Position AND velocity, because a single
    camera frame carries position only; that asymmetry is the experiment, and
    hiding it here by dropping velocity would answer a different question.

    `mdp.ball_radius` is deliberately NOT here. One ball spec means it is a
    per-run constant, and a constant observation is absorbed into the first
    layer's bias — it informs nothing and costs a dim in two groups. It earns a
    slot the day size is randomized (then a small near ball and a large far one
    stop being distinguishable without it), and not before.
    """
    return {
        "ball_pos_b": _T(mdp.ball_pos_b),
        "ball_vel_b": _T(mdp.ball_vel_b),
    }


def robot_motion_cmd_terms(p: dict) -> dict:
    """sys1's command stream — the root twist alone (uolm prepends a contact
    schedule; dodge has none staged, so this is the shared half by itself)."""
    return robot_root_twist_cmd_terms(p)


def tracking_ref_terms(p: dict) -> dict:
    """The motion reference an actor needs when no frozen base supplies it."""
    return {
        "command": _T(mdp.generated_commands, p),
        "motion_anchor_pos_b": _T(mdp.motion_anchor_pos_b_future, p),
        "motion_anchor_ori_b": _T(mdp.motion_anchor_ori_b_future, p),
    }


# ---------------------------------------------------------------------------
# The named groups
# ---------------------------------------------------------------------------

def policy_group() -> ObservationGroupCfg:
    """The frozen SONIC decoder's proprio stream. Term set owned by mocke.

    Assembled CLEAN; `orcs.core.obs.apply_obs_noise` stamps the sensor noise
    post-assembly, so one table covers this group and a consumer's own deployed
    streams from one edit (mocke's `noisy=` flag reaches only this one).
    """
    return _grp(profile.policy_obs_terms())


def tokenizer_groups(command_name: str = "motion") -> ObservationGroupCfg:
    """The frozen SONIC encoder's reference window — a held stand, every frame."""
    return profile.extra_obs_groups(command_name, mode="g1")


def augmentation_group(c: ObsCtx) -> ObservationGroupCfg:
    """The adapter's conditioning stream — THE extension seam.

    feedback:    ball kinematics + robot root state (the odometry that says how
                 far the robot has drifted from its station).
    feedforward: sys1's root-twist command — constant zero here, kept because it
                 is the base's contract, not because it carries information.

    Swap this one group for encoder features and the frozen base, critic,
    rewards and terminations all carry over unchanged.
    """
    return _grp({
        **ball_state_terms(),
        **robot_root_state_terms(),
        **robot_motion_cmd_terms(c.p),
    })


def critic_group(c: ObsCtx) -> ObservationGroupCfg:
    """Privileged critic obs: full proprio + reference + ball + reward vector.

    Never deployed, so it may read anything the sim knows — and it keeps the
    ball state even when the actor loses it. That asymmetry is deliberate: actor
    vision-only, value function privileged, so a delta against the privileged
    row is the cost of seeing the ball through a camera and nothing else.
    """
    return _grp({
        **tracking_ref_terms(c.p),
        **proprio_terms(),
        # explicit, NOT via proprio_terms: base_lin_vel left the deployable
        # bundle 1Aug2026 (no state estimator on hw), but the critic never
        # deploys — dropping it there was a pure value-function downgrade.
        "base_lin_vel": _T(mdp.base_lin_vel),
        **robot_root_state_terms(),
        "actions": _T(mdp.last_action),
        **ball_state_terms(),
        **robot_motion_cmd_terms(c.p),
        "reward_vec": _T(mdp.unweighted_reward_vector, {"enabled": True}),
    })


# ---------------------------------------------------------------------------
# Agent layouts
# ---------------------------------------------------------------------------

def sonic_obs(c: ObsCtx) -> dict[str, ObservationGroupCfg]:
    """3-stream (frozen base): SONIC policy + tokenizer / augmentation / critic."""
    return {
        "policy": policy_group(),
        **tokenizer_groups(c.p["command_name"]),
        "augmentation": augmentation_group(c),
        "critic": critic_group(c),
    }
