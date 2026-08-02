"""observation_cfgs.py — THE uolm signal library.

One place for every uolm-owned observation group and the atomic term bundles they
share, so an experiment is a re-pick of groups, never a re-plumb. The frozen-base
streams (`policy`, `tokenizer`) come from `mocke.sonic.profile` — that is the WBC
contract and orcs never redefines it. Everything downstream of the frozen base is
here:

    [base]      policy + tokenizer streams          — mocke (frozen SONIC contract)
    [adapter]   AUGMENTATION (object kin + sys1 cmd) — this file
    [critic]    privileged full state               — this file

A consumer that wants a different feedback modality (e.g. vision) swaps ONE group
— `augmentation` — and inherits the rest. That is the whole extension seam.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from mjlab.managers.observation_manager import ObservationGroupCfg, ObservationTermCfg
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mocke.sonic import profile

from orcs.assets import OBJECT_BODY_NAME
from orcs.tasks.uolm import mdp


@dataclass(frozen=True)
class ObsCtx:
    """Assembly context threaded through the group factories (uniform signature)."""
    obj: SceneEntityCfg = field(
        default_factory=lambda: SceneEntityCfg(OBJECT_BODY_NAME))
    p: dict = field(default_factory=lambda: {"command_name": "motion"})


def _T(func, params: dict | None = None) -> ObservationTermCfg:
    return ObservationTermCfg(func=func) if params is None else ObservationTermCfg(
        func=func, params=params)


def _grp(terms: dict, *, concat: bool = True, corrupt: bool = False) -> ObservationGroupCfg:
    """Uniform group: nan-sanitized per term. concat=False -> dict group."""
    return ObservationGroupCfg(
        terms=terms, concatenate_terms=concat,
        enable_corruption=corrupt, nan_policy="sanitize", nan_check_per_term=True,
    )


# ---------------------------------------------------------------------------
# Atomic term bundles — defined ONCE, composed into many groups
# ---------------------------------------------------------------------------

def proprio_terms() -> dict:
    """Robot self-state (deployable): gravity dir + base twist + joint pos/vel."""
    return {
        "projected_gravity": _T(mdp.projected_gravity),
        # "base_lin_vel": _T(mdp.base_lin_vel),  #NOTE (lok-i) 1Aug2026: found insensitive 
        "base_ang_vel": _T(mdp.base_ang_vel),
        "joint_pos": _T(mdp.joint_pos_rel),
        "joint_vel": _T(mdp.joint_vel_rel),
    }


def robot_motion_cmd_terms(p: dict) -> dict:
    """sys1 command stream (SUGAR c_t parity): per-body contact + root twist cmd.

    Robot-state language only — no time-varying object refs, which stay
    critic-side and are thrown away post-training.
    """
    return {
        "bodywise_contact_cmd": _T(mdp.bodywise_contact_cmd, p),
        "robot_root_lin_vel_cmd": _T(mdp.robot_root_lin_vel_cmd, p),
        "robot_root_ang_vel_cmd": _T(mdp.robot_root_ang_vel_cmd, p),
    }


def object_goal_terms(p: dict) -> dict:
    """Task goal from the level ABOVE sys1 — fixed per episode. Full pose;
    orientation-only tasks pop `object_goal_pos`."""
    return {
        "object_goal_ori": _T(mdp.object_goal_ori_mat6d, p),
        "object_goal_pos": _T(mdp.object_goal_pos_env, p),
    }


def object_state_terms(obj: SceneEntityCfg) -> dict:
    """Ego-observable object state (base-frame pose + twist).

    Base frame, not world: info-parity with an FPV image stream, since world
    frame needs odometry — unavailable on hw and un-decodable from images.

    NOT what the privileged uolm agents read (that is `object_state_w_terms`);
    this atom exists for the DEPLOYABLE side — a vision consumer's aux
    prediction target, which is ego-observable-only by contract. Do not
    "unify" the two: the frame difference IS the privilege boundary.
    """
    return {
        "object_pose_b": _T(mdp.object_pose_b, {"object_cfg": obj}),
        "object_twist_b": _T(mdp.object_twist_b, {"object_cfg": obj}),
    }


def object_state_w_terms(obj: SceneEntityCfg) -> dict:
    """Privileged object state, ENV frame (fcrl parity) — pose + twist.

    Env frame, not base: the task goal (`object_goal_*`) is an env-frame
    constant, so a base-frame object state cannot be differenced against it
    without odometry. Mixing the two silently made the goal channel
    unactionable (2026-08-01) — every privileged pose in one group now shares
    one frame, and `robot_root_pos_env` supplies the odometry that closes it.
    """
    return {
        "object_pos_w": _T(mdp.object_pos_w_obs, {"object_cfg": obj}),
        "object_ori_mat6d_w": _T(mdp.object_ori_mat6d_w, {"object_cfg": obj}),
        "object_lin_vel_w": _T(mdp.object_lin_vel_w_obs, {"object_cfg": obj}),
        "object_ang_vel_w": _T(mdp.object_ang_vel_w_obs, {"object_cfg": obj}),
    }


def robot_root_state_terms(p: dict) -> dict:
    """Privileged robot-root state: env-frame position + body-frame twist.

    Orientation is already in `proprio_terms`/the frozen base's stream, so only
    the odometry half lives here.
    """
    del p
    return {
        "robot_root_pos_env": _T(mdp.robot_root_pos_env),
        "robot_root_lin_vel_b": _T(mdp.base_lin_vel),
        "robot_root_ang_vel_b": _T(mdp.base_ang_vel),
    }


def object_identity_terms(p: dict, obj: SceneEntityCfg, *, desc: bool = False) -> dict:
    """Which object this world is simulating.

    One-hot over the roster (fcrl streamed a scalar index; one-hot avoids the
    false ordinal). The roster spans 0.56-9.6 kg with one clip each, so without
    it the CRITIC has to average V over K indistinguishable regimes — that, not
    the actor, is why this term is load-bearing.

    `desc=True` adds the roster-free inertial descriptor — the axis that
    survives a roster change. Off by default: one variable at a time.
    """
    terms = {"object_id": _T(mdp.object_id_onehot, p)}
    if desc:
        terms["object_desc"] = _T(mdp.object_inertial_desc, {"object_cfg": obj})
    return terms


# ---------------------------------------------------------------------------
# The named groups
# ---------------------------------------------------------------------------

def policy_group() -> ObservationGroupCfg:
    """The frozen SONIC decoder's proprio stream (history-10, no odometry by
    construction). Term set owned by mocke."""
    return _grp(profile.policy_obs_terms(), corrupt=True)


def tokenizer_groups(mode: str = "g1", command_name: str = "motion") -> dict:
    """The frozen SONIC encoder's reference-window stream. Owned by mocke."""
    return profile.extra_obs_groups(command_name, mode=mode)


def augmentation_group(c: ObsCtx) -> ObservationGroupCfg:
    """The adapter's conditioning stream — ObjKin feedback + sys1 feedforward.

    feedback:    env-frame object kinematics + object identity + robot root
                 (env-frame pos, body-frame twist).  fcrl parity.
    feedforward: task goal (fixed per episode) + sys1 command stream
                 {l, v, w}_cmd_t (orcs addition — robot-state language, and
                 already frame-clean).

    ONE frame for every pose in the group. The 2026-08-01 regression was a
    base-frame object state sharing a group with an env-frame goal and no root
    pose at all: `goal ⊖ object` was not computable from the stream, so the
    goal reward and the goal obs were both dead channels.

    THE extension seam: a vision consumer replaces this one group and inherits
    the frozen base, the critic, rewards and terminations unchanged.
    """
    return _grp({
        **object_state_w_terms(c.obj),
        **object_identity_terms(c.p, c.obj),
        **robot_root_state_terms(c.p),
        **object_goal_terms(c.p),
        **robot_motion_cmd_terms(c.p),
    })


def critic_group(c: ObsCtx) -> ObservationGroupCfg:
    """Privileged critic obs: full proprio + object + goal + reference (un-zeroed).

    Never deployed, so it may read anything the sim knows.
    """
    return _grp({
        # tracking command + anchor (privileged: not zeroed)
        "command": _T(mdp.generated_commands, c.p),
        "motion_anchor_pos_b": _T(mdp.motion_anchor_pos_b_future, c.p),
        "motion_anchor_ori_b": _T(mdp.motion_anchor_ori_b_future, c.p),
        # proprio
        **proprio_terms(),
        # explicit, NOT via proprio_terms: base_lin_vel left the deployable
        # bundle 1Aug2026 (no state estimator on hw), but the critic never
        # deploys — dropping it there was a pure value-function downgrade.
        "base_lin_vel": _T(mdp.base_lin_vel),
        # env-frame root position — the odometry half of the privileged root
        # state (fcrl's robot_root_pos_w). Same reason as in `augmentation`:
        # without it an env-frame goal is not reachable-relative.
        "robot_root_pos_env": _T(mdp.robot_root_pos_env),
        "actions": _T(mdp.last_action),
        # object state (env frame) + which object this world is
        **object_state_w_terms(c.obj),
        **object_identity_terms(c.p, c.obj),
        # object goal (full pose; ori-only tasks pop object_goal_pos)
        **object_goal_terms(c.p),
        # object reference trajectory (N-step future, body frame)
        "object_ref_pos_b": _T(mdp.motion_object_pos_b_future, c.p),
        "object_ref_ori_b": _T(mdp.motion_object_ori_b_future, c.p),
        # sys1 command stream (reward-relevant: contact_consistency)
        **robot_motion_cmd_terms(c.p),
        # reward conditioning: V(concat(s, r_vec))
        "reward_vec": _T(mdp.unweighted_reward_vector, {"enabled": True}),
    })


def tracking_ref_terms(p: dict) -> dict:
    """The motion reference an actor needs when no frozen base supplies it."""
    return {
        "command": _T(mdp.generated_commands, p),
        "motion_anchor_pos_b": _T(mdp.motion_anchor_pos_b_future, p),
        "motion_anchor_ori_b": _T(mdp.motion_anchor_ori_b_future, p),
    }


# ---------------------------------------------------------------------------
# Agent layouts — one per agent architecture
# ---------------------------------------------------------------------------

def sonic_obs(c: ObsCtx, mode: str = "g1") -> dict[str, ObservationGroupCfg]:
    """3-stream (frozen base): SONIC policy + tokenizer / augmentation / critic.

    The adapter injects into `augmentation`; the base streams are mocke's
    contract and must not be touched.
    """
    return {
        "policy": policy_group(),
        **tokenizer_groups(mode, c.p["command_name"]),
        "augmentation": augmentation_group(c),
        "critic": critic_group(c),
    }


def tara_obs(c: ObsCtx) -> dict[str, ObservationGroupCfg]:
    """2-stream (tabula rasa): policy = everything deployable / critic = privileged.

    No frozen base, so the actor must be handed the motion reference itself —
    `tracking_ref_terms` is what the WBC's tokenizer stream would otherwise
    carry. Everything else is the same augmentation content the adapter reads,
    flattened into one stream — env frame included, or the baseline would be
    solving a strictly harder problem than the thing it is a baseline for.
    """
    return {
        "policy": _grp({
            **tracking_ref_terms(c.p),
            **proprio_terms(),
            "base_lin_vel": _T(mdp.base_lin_vel),
            "robot_root_pos_env": _T(mdp.robot_root_pos_env),
            "actions": _T(mdp.last_action),
            **object_state_w_terms(c.obj),
            **object_identity_terms(c.p, c.obj),
            **object_goal_terms(c.p),
            **robot_motion_cmd_terms(c.p),
        }, corrupt=True),
        "critic": critic_group(c),
    }
