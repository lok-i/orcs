"""Robustness domain, robot half — the training domain (we never train nominal).

Two families, prefix-encoded in the event keys so later filtering (e.g.
play-time stripping) is a prefix match, not a name list:

  perturb_* : STATE variations — sweep states across the basin of attraction
              so the policy learns to recover rather than memorize one
              trajectory. Runtime pushes here; the reset-time half (RSI
              randomization) lives on the motion command and is wired via its
              cfg fields.
  rand_*    : PARAM variations — per-env-constant world parameters the policy
              can neither infer nor control (sim2real invariance).

The split is by WHAT A TERM TOUCHES, not by which task wants it. A G1 is a G1
in every task, so the robot half lives here and every task calls it; the object
half needs an object and lives with the task that has one
(:mod:`orcs.tasks.uolm.robustness`).

fcrl parity notes (isaaclab -> mjlab):
  - material static/dynamic friction + restitution
      -> dr.geom_friction (MuJoCo: one tangential friction; restitution is
         solref territory — dropped).
  - add_joint_default_pos -> dr.encoder_bias: the honest model (biased obs
    via joint_pos_rel(biased=True) + PD target shifted -bias, both halves of
    a mis-zeroed encoder). mjlab's qpos0 term never reaches obs/action
    offsets (they snapshot default_joint_pos at entity init).
"""

from __future__ import annotations

from mjlab.envs import ManagerBasedRlEnvCfg
from mjlab.envs.mdp import dr
from mjlab.managers.event_manager import EventTermCfg
from mjlab.managers.scene_entity_config import SceneEntityCfg

from orcs.core import mdp
from orcs.core.mdp.commands import MultiClipMotionCommandCfg
from orcs.core.obs import NOISY_GROUPS

__all__ = ["ROOT_VELOCITY_RANGE", "ROBOT_STATE_VARIATION",
           "ROBOT_PARAM_VARIATION", "apply_robot_robustness",
           "DOMAIN_PREFIXES", "strip_domain"]

DOMAIN_PREFIXES = ("perturb_", "rand_")
"""What a domain event is called. `strip_domain` matches on these, so a term
added to either half is stripped from play without editing a list."""


# ---------------------------------------------------------------------------
# Ranges (fcrl parity unless noted)
# ---------------------------------------------------------------------------

ROOT_VELOCITY_RANGE = {  # m/s, rad/s — same as the low-level WBC MDP
    "x": (-0.3, 0.3),
    "y": (-0.3, 0.3),
    "z": (-0.15, 0.15),
    "roll": (-0.3, 0.3),
    "pitch": (-0.3, 0.3),
    "yaw": (-0.5, 0.5),
}

ROBOT_STATE_VARIATION = {
    # ── RSI (consumed by MultiClipMotionCommand._resample_command) ──
    "robot_pose_range": {},  # zeros — RSI pose is exact, like the WBC's MDP
    "robot_velocity_range": ROOT_VELOCITY_RANGE,
    "robot_joint_position_range": (-0.05, 0.05),  # rad
    # ── runtime pushes ──
    "perturb_robot_velocity_range": ROOT_VELOCITY_RANGE,
}

ROBOT_PARAM_VARIATION = {
    "robot_friction_range": (0.3, 1.6),
    "robot_com_range": {  # torso_link, m
        0: (-0.025, 0.025),
        1: (-0.05, 0.05),
        2: (-0.05, 0.05),
    },
    "encoder_bias_range": (-0.01, 0.01),  # rad
}


# ---------------------------------------------------------------------------
# apply_robot_robustness — mutates cfg in place
# ---------------------------------------------------------------------------

def apply_robot_robustness(
    cfg: ManagerBasedRlEnvCfg,
    *,
    state: bool = True,
    param: bool = True,
) -> None:
    """Wire the robot-only robustness domain into any orcs/vibe cfg.

    state: RSI randomization (motion-command fields) + the perturb_robot push.
    param: rand_* startup events + biased joint_pos on the actor stream
           (encoder bias must reach obs AND action, or it's a half-model).

    Gated on `MultiClipMotionCommandCfg`, the base every orcs motion command
    derives from — NOT on a task's subclass, or the RSI half silently skips
    every task but the one it was written for.
    """
    sv, pv = ROBOT_STATE_VARIATION, ROBOT_PARAM_VARIATION

    if state:
        mc = cfg.commands.get("motion")
        if isinstance(mc, MultiClipMotionCommandCfg):
            mc.pose_range = dict(sv["robot_pose_range"])
            mc.velocity_range = dict(sv["robot_velocity_range"])
            mc.joint_position_range = sv["robot_joint_position_range"]

        cfg.events["perturb_robot"] = EventTermCfg(
            func=mdp.push_by_setting_velocity,
            mode="interval",
            interval_range_s=(0.0, cfg.episode_length_s),
            params={
                "velocity_range": sv["perturb_robot_velocity_range"],
                "asset_cfg": SceneEntityCfg("robot"),
            },
        )

    if param:
        cfg.events["rand_robot_friction"] = EventTermCfg(
            func=dr.geom_friction,
            mode="startup",
            params={
                "asset_cfg": SceneEntityCfg("robot"),
                "operation": "abs",
                "ranges": pv["robot_friction_range"],
            },
        )
        cfg.events["rand_robot_com"] = EventTermCfg(
            func=dr.body_com_offset,
            mode="startup",
            params={
                "asset_cfg": SceneEntityCfg("robot", body_names="torso_link"),
                "operation": "add",
                "ranges": pv["robot_com_range"],
            },
        )
        cfg.events["rand_encoder_bias"] = EventTermCfg(
            func=dr.encoder_bias,
            mode="startup",
            params={
                "asset_cfg": SceneEntityCfg("robot"),
                "bias_range": pv["encoder_bias_range"],
            },
        )
        # Encoder bias reaches the PD target automatically (JointPositionAction
        # subtracts it); the actor's joint_pos obs must opt in. Critic stays
        # true-state (privileged).
        pol = cfg.observations.get("policy")
        jp = pol.terms.get("joint_pos") if pol is not None else None
        if jp is not None:
            jp.params = {**(jp.params or {}), "biased": True}


def strip_domain(
    cfg: ManagerBasedRlEnvCfg,
    *,
    obs_groups: tuple[str, ...] = NOISY_GROUPS,
    keep: tuple[str, ...] = (),
) -> None:
    """Undo the domain for a play/eval build — the exact inverse of the applies.

    Prefix match on `DOMAIN_PREFIXES`, so a knob added to either half is
    subtracted here without editing a list. `keep` exempts events that share the
    `rand_` prefix but are a TASK CHANNEL rather than a domain (vibe's repose
    randomizes the cube's face colors to DEFINE its goal — stripping that would
    not clean the eval, it would delete the task).
    """
    for key in [k for k in cfg.events
                if k.startswith(DOMAIN_PREFIXES) and k not in keep]:
        cfg.events.pop(key)
    for name in obs_groups:
        group = cfg.observations.get(name)
        if group is not None:
            group.enable_corruption = False
    pol = cfg.observations.get("policy")
    jp = pol.terms.get("joint_pos") if pol is not None else None
    if jp is not None:
        jp.params = {**(jp.params or {}), "biased": False}
