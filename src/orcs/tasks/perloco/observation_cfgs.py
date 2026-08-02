"""observation_cfgs.py — THE perloco signal library.

Read this against `orcs.tasks.uolm.observation_cfgs`: the two differ in ONE
group, and that is the claim the task exists to test.

    [base]      policy + tokenizer streams   — mocke (frozen SONIC contract)
    [adapter]   AUGMENTATION                 — uolm: object kinematics
                                               perloco: TERRAIN height scan
    [critic]    privileged full state        — same shape, different extras

Everything else — proprio, the sys1 root-twist command, the reward-vector
conditioning — is `orcs.core.obs` and is shared verbatim. If a term here
duplicates one there, that is a bug, not a convenience.

What perloco does NOT carry, and why:

  `bodywise_contact_cmd`  uolm's per-body contact schedule is derived from an
                          object contact graph. Terrain contact has no such
                          schedule staged, so the slot stays empty rather than
                          get filled with a zero tensor that looks like data.
  object anything         there is no object.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from mjlab.envs import mdp as envs_mdp
from mjlab.managers.observation_manager import ObservationGroupCfg, ObservationTermCfg
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
from orcs.tasks.perloco import mdp
from orcs.tasks.perloco.sensors import SCAN_MAX_DISTANCE, TERRAIN_SCAN_SENSOR_NAME

__all__ = ["ObsCtx", "height_scan_terms", "sonic_obs", "tara_obs"]


@dataclass(frozen=True)
class ObsCtx:
    """Assembly context threaded through the group factories (uniform signature)."""

    p: dict = field(default_factory=lambda: {"command_name": "motion"})


# ---------------------------------------------------------------------------
# Atomic term bundles
# ---------------------------------------------------------------------------

def height_scan_terms(sensor_name: str = TERRAIN_SCAN_SENSOR_NAME) -> dict:
    """THE exteroception: 187 downward ray clearances under the pelvis.

    Scaled by `1/max_distance` (mjlab's rough-terrain convention) rather than
    normalized: the scale is a CONSTANT, so a value keeps the same meaning
    across runs, tasks and checkpoints. A running normalizer would make the
    channel's meaning depend on the terrain roster it was trained on, which is
    exactly the thing a curriculum changes underneath it.
    """
    return {
        "height_scan": ObservationTermCfg(
            func=envs_mdp.height_scan,
            params={"sensor_name": sensor_name},
            scale=1.0 / SCAN_MAX_DISTANCE,
        ),
    }


def robot_motion_cmd_terms(p: dict) -> dict:
    """sys1's command stream — the root twist alone.

    uolm prepends a per-body contact schedule here; perloco has none staged, so
    this is the shared half by itself and NOT a narrower copy of uolm's.
    """
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
    """The frozen SONIC decoder's proprio stream. Term set owned by mocke."""
    return _grp(profile.policy_obs_terms(), corrupt=True)


def tokenizer_groups(command_name: str = "motion") -> dict:
    """The frozen SONIC encoder's reference-window stream. Owned by mocke."""
    return profile.extra_obs_groups(command_name, mode="g1")


def augmentation_group(c: ObsCtx) -> ObservationGroupCfg:
    """The adapter's conditioning stream — THE extension seam.

    feedback:    terrain height scan + robot root state (env-frame position,
                 body-frame twist). The scan is ego-centric and yaw-aligned;
                 the root state is the odometry that makes it placeable.
    feedforward: sys1's root-twist command.

    Swap this one group and the frozen base, critic, rewards and terminations
    all carry over unchanged — which is how uolm's ObjKin adapter and this one
    stay comparable.
    """
    return _grp({
        **height_scan_terms(),
        **robot_root_state_terms(),
        **robot_motion_cmd_terms(c.p),
    })


def critic_group(c: ObsCtx) -> ObservationGroupCfg:
    """Privileged critic obs: full proprio + reference + terrain + reward vector.

    Never deployed, so it may read anything the sim knows. It reads the SAME
    height scan as the adapter — the scan is the only terrain description that
    exists, and inventing a privileged twin (true tile geometry, level index)
    would make the critic's job depend on the staging layout.
    """
    return _grp({
        # tracking command + anchor (privileged: not zeroed)
        **tracking_ref_terms(c.p),
        # proprio
        **proprio_terms(),
        # explicit, NOT via proprio_terms: base_lin_vel left the deployable
        # bundle 1Aug2026 (no state estimator on hw), but the critic never
        # deploys — dropping it there was a pure value-function downgrade.
        "base_lin_vel": _T(mdp.base_lin_vel),
        **robot_root_state_terms(),
        "actions": _T(mdp.last_action),
        # exteroception
        **height_scan_terms(),
        # sys1 command stream
        **robot_motion_cmd_terms(c.p),
        # reward conditioning: V(concat(s, r_vec))
        "reward_vec": _T(mdp.unweighted_reward_vector, {"enabled": True}),
    })


# ---------------------------------------------------------------------------
# Agent layouts — one per agent architecture
# ---------------------------------------------------------------------------

def sonic_obs(c: ObsCtx) -> dict[str, ObservationGroupCfg]:
    """3-stream (frozen base): SONIC policy + tokenizer / augmentation / critic."""
    return {
        "policy": policy_group(),
        **tokenizer_groups(c.p["command_name"]),
        "augmentation": augmentation_group(c),
        "critic": critic_group(c),
    }


def tara_obs(c: ObsCtx) -> dict[str, ObservationGroupCfg]:
    """2-stream (tabula rasa): policy = everything deployable / critic = privileged.

    No frozen base, so the actor must be handed the motion reference itself —
    `tracking_ref_terms` is what the WBC's tokenizer stream would otherwise
    carry. Everything else is the augmentation content flattened into one
    stream, so the baseline solves the same problem, not a harder one.
    """
    return {
        "policy": _grp({
            **tracking_ref_terms(c.p),
            **proprio_terms(),
            "base_lin_vel": _T(mdp.base_lin_vel),
            **robot_root_state_terms(),
            "actions": _T(mdp.last_action),
            **height_scan_terms(),
            **robot_motion_cmd_terms(c.p),
        }, corrupt=True),
        "critic": critic_group(c),
    }
