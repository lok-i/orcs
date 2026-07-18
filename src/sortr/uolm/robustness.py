"""Robustness domain — the training domain (we never train nominal).

Two families, prefix-encoded in the event keys so later filtering (e.g.
play-time stripping) is a prefix match, not a name list:

  perturb_* : STATE variations — sweep states across the basin of attraction
              so the policy learns to recover rather than memorize one
              trajectory. Runtime pushes here; the reset-time half (RSI
              randomization) lives in OmniObjectMotionCommand and is wired
              via the command cfg fields.
  rand_*    : PARAM variations — per-env-constant world parameters the policy
              can neither infer nor control (sim2real invariance). Physical
              terms only.

fcrl parity notes (isaaclab -> mjlab):
  - object mass 0.5-2.0x log_uniform + recompute_inertia
      -> dr.pseudo_inertia(alpha_range): mass & inertia scale e^{2a} jointly
         (exact), and uniform-in-alpha IS log-uniform-in-mass.
  - material static/dynamic friction + restitution
      -> dr.geom_friction (MuJoCo: one tangential friction; restitution is
         solref territory — dropped).
  - add_joint_default_pos -> dr.encoder_bias: the honest model (biased obs
    via joint_pos_rel(biased=True) + PD target shifted -bias, both halves of
    a mis-zeroed encoder). mjlab's qpos0 term never reaches obs/action
    offsets (they snapshot default_joint_pos at entity init).
"""

from __future__ import annotations

import math

from mjlab.envs import ManagerBasedRlEnvCfg
from mjlab.envs.mdp import dr
from mjlab.managers.event_manager import EventTermCfg
from mjlab.managers.scene_entity_config import SceneEntityCfg

from sortr.uolm import mdp
from sortr.uolm.mdp.commands_omni_object import OmniObjectMotionCommandCfg

__all__ = ["STATE_VARIATION", "PARAM_VARIATION", "apply_robustness"]


# ---------------------------------------------------------------------------
# Ranges (fcrl parity unless noted)
# ---------------------------------------------------------------------------

_ROOT_VELOCITY_RANGE = {  # m/s, rad/s — same as the low-level WBC MDP
    "x": (-0.3, 0.3),
    "y": (-0.3, 0.3),
    "z": (-0.15, 0.15),
    "roll": (-0.3, 0.3),
    "pitch": (-0.3, 0.3),
    "yaw": (-0.5, 0.5),
}

STATE_VARIATION = {
    # ── RSI (consumed by OmniObjectMotionCommand._resample_command) ──
    "robot_pose_range": {},  # zeros — RSI pose is exact, like the WBC's MDP
    "robot_velocity_range": _ROOT_VELOCITY_RANGE,
    "robot_joint_position_range": (-0.05, 0.05),  # rad
    # clip-start only (non-contact by demo construction) — mirrors sim2real
    # initial-placement uncertainty on hardware.
    "object_init_pose_range": {
        "x": (-0.1, 0.1),        # m
        "y": (-0.1, 0.1),
        "z": (0.0, 0.0),
        "roll": (-0.1, 0.1),     # rad
        "pitch": (-0.1, 0.1),
        "yaw": (-0.174533, 0.174533),
    },
    # mid-clip, ref-contact only (robot has control-authority).
    "object_in_contact_velocity_range": {
        "x": (-0.5, 0.5),        # m/s
        "y": (-0.5, 0.5),
        "z": (-0.5, 0.5),
        "roll": (-1.0, 1.0),     # rad/s
        "pitch": (-1.0, 1.0),
        "yaw": (-1.0, 1.0),
    },
    # ── runtime pushes ──
    "perturb_robot_velocity_range": _ROOT_VELOCITY_RANGE,
    "perturb_object_velocity_range": {
        "x": (-0.5, 0.5),
        "y": (-0.5, 0.5),
        "z": (-0.5, 0.5),
        "roll": (-1.0, 1.0),
        "pitch": (-1.0, 1.0),
        "yaw": (-1.0, 1.0),
    },
}

# mass scale [0.5, 2.0]: pseudo-inertia scales mass+inertia by e^{2a}
# -> a = 0.5*ln(scale); uniform-in-a == log_uniform-in-mass (fcrl parity).
_OBJECT_MASS_SCALE = (0.5, 2.0)

PARAM_VARIATION = {
    "object_inertia_alpha_range": tuple(
        0.5 * math.log(s) for s in _OBJECT_MASS_SCALE),
    "object_friction_range": (0.2, 0.8),
    "robot_friction_range": (0.3, 1.6),
    "robot_com_range": {  # torso_link, m
        0: (-0.025, 0.025),
        1: (-0.05, 0.05),
        2: (-0.05, 0.05),
    },
    "encoder_bias_range": (-0.01, 0.01),  # rad
}


# ---------------------------------------------------------------------------
# apply_robustness — THE entry point (mutates cfg in place)
# ---------------------------------------------------------------------------

def apply_robustness(
    cfg: ManagerBasedRlEnvCfg,
    *,
    state: bool = True,
    param: bool = True,
    object_name: str = "object",
    sensor_name: str = "object_contact_graph",
    hand_body_names: tuple[str, ...] = ("left_wrist_yaw_link", "right_wrist_yaw_link"),
) -> None:
    """Wire the robustness domain into an omni-object/repose cfg.

    state: RSI randomization (command cfg fields) + perturb_* interval events.
    param: rand_* startup events + biased joint_pos on the actor stream
           (encoder bias must reach obs AND action, or it's a half-model).

    Degrades per-cfg: RSI rand needs a motion command (goal-command envs
    scatter the object via their own command); perturb_object needs the
    contact-graph sensor. Absent -> skipped, rest still applies.
    """
    sv, pv = STATE_VARIATION, PARAM_VARIATION

    if state:
        mc = cfg.commands.get("motion")
        if isinstance(mc, OmniObjectMotionCommandCfg):
            mc.pose_range = dict(sv["robot_pose_range"])
            mc.velocity_range = dict(sv["robot_velocity_range"])
            mc.joint_position_range = sv["robot_joint_position_range"]
            mc.object_init_pose_range = dict(sv["object_init_pose_range"])
            mc.object_in_contact_velocity_range = dict(
                sv["object_in_contact_velocity_range"])

        cfg.events["perturb_robot"] = EventTermCfg(
            func=mdp.push_by_setting_velocity,
            mode="interval",
            interval_range_s=(0.0, cfg.episode_length_s),
            params={
                "velocity_range": sv["perturb_robot_velocity_range"],
                "asset_cfg": SceneEntityCfg("robot"),
            },
        )
        if any(s.name == sensor_name
               for s in (cfg.scene.sensors or ())):
            cfg.events["perturb_object"] = EventTermCfg(
                func=mdp.PerturbObjectInRobotContact,
                mode="interval",
                interval_range_s=(0.0, 0.0),  # tick every step; class latches
                params={
                    "velocity_range": sv["perturb_object_velocity_range"],
                    "object_cfg": SceneEntityCfg(object_name),
                    "sensor_name": sensor_name,
                    "contact_body_names": hand_body_names,
                    "contact_force_threshold": 0.1,
                },
            )

    if param:
        cfg.events["rand_object_inertia"] = EventTermCfg(
            func=dr.pseudo_inertia,
            mode="startup",
            params={
                "asset_cfg": SceneEntityCfg(object_name),
                "alpha_range": pv["object_inertia_alpha_range"],
            },
        )
        cfg.events["rand_object_friction"] = EventTermCfg(
            func=dr.geom_friction,
            mode="startup",
            params={
                "asset_cfg": SceneEntityCfg(object_name),
                "operation": "abs",
                "ranges": pv["object_friction_range"],
            },
        )
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
