"""Robustness domain, OBJECT half — the terms that need something to manipulate.

The robot half is :mod:`orcs.core.robustness` and is every task's; what is left
here is what only exists when a scene carries an object: its inertia, its
friction, its RSI placement, and the in-contact push. Same two families and the
same key prefixes (`perturb_*` state, `rand_*` param), so play-time stripping
stays a prefix match across both halves.

fcrl parity notes (isaaclab -> mjlab):
  - object mass 0.5-2.0x log_uniform + recompute_inertia
      -> dr.pseudo_inertia(alpha_range): mass & inertia scale e^{2a} jointly
         (exact), and uniform-in-alpha IS log-uniform-in-mass.
  - material static/dynamic friction -> dr.geom_friction (one tangential
    friction; restitution is solref territory — dropped).
"""

from __future__ import annotations

import math

from mjlab.envs import ManagerBasedRlEnvCfg
from mjlab.envs.mdp import dr
from mjlab.managers.event_manager import EventTermCfg
from mjlab.managers.scene_entity_config import SceneEntityCfg

from orcs.core.robustness import (
    ROBOT_PARAM_VARIATION,
    ROBOT_STATE_VARIATION,
    apply_robot_robustness,
)
from orcs.tasks.uolm import mdp
from orcs.tasks.uolm.mdp.commands import ObjectMotionCommandCfg

__all__ = [
    "STATE_VARIATION", "PARAM_VARIATION", "OBJECT_STATE_VARIATION",
    "OBJECT_PARAM_VARIATION", "apply_object_robustness", "apply_robustness",
]


# ---------------------------------------------------------------------------
# Ranges (fcrl parity unless noted)
# ---------------------------------------------------------------------------

OBJECT_STATE_VARIATION = {
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

OBJECT_PARAM_VARIATION = {
    "object_inertia_alpha_range": tuple(
        0.5 * math.log(s) for s in _OBJECT_MASS_SCALE),
    "object_friction_range": (0.2, 0.8),
}

# The whole domain under one name, for a reader who wants the uolm view of it.
STATE_VARIATION = {**ROBOT_STATE_VARIATION, **OBJECT_STATE_VARIATION}
PARAM_VARIATION = {**ROBOT_PARAM_VARIATION, **OBJECT_PARAM_VARIATION}


# ---------------------------------------------------------------------------
# apply_* — THE entry points (mutate cfg in place)
# ---------------------------------------------------------------------------

def apply_object_robustness(
    cfg: ManagerBasedRlEnvCfg,
    *,
    state: bool = True,
    param: bool = True,
    object_name: str = "object",
    sensor_name: str = "object_contact_graph",
    hand_body_names: tuple[str, ...] = ("left_wrist_yaw_link", "right_wrist_yaw_link"),
) -> None:
    """Wire the object half. Degrades per-cfg: RSI rand needs an object motion
    command, perturb_object needs the contact-graph sensor; absent -> skipped."""
    sv, pv = OBJECT_STATE_VARIATION, OBJECT_PARAM_VARIATION

    if state:
        mc = cfg.commands.get("motion")
        if isinstance(mc, ObjectMotionCommandCfg):
            mc.object_init_pose_range = dict(sv["object_init_pose_range"])
            mc.object_in_contact_velocity_range = dict(
                sv["object_in_contact_velocity_range"])

        if any(s.name == sensor_name for s in (cfg.scene.sensors or ())):
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


def apply_robustness(
    cfg: ManagerBasedRlEnvCfg,
    *,
    state: bool = True,
    param: bool = True,
    object_name: str = "object",
    sensor_name: str = "object_contact_graph",
    hand_body_names: tuple[str, ...] = ("left_wrist_yaw_link", "right_wrist_yaw_link"),
) -> None:
    """Both halves — what an object task means by "the training domain"."""
    apply_robot_robustness(cfg, state=state, param=param)
    apply_object_robustness(
        cfg, state=state, param=param, object_name=object_name,
        sensor_name=sensor_name, hand_body_names=hand_body_names)
