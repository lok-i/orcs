"""UOLM env config — frozen SONIC WBC adapter over per-world object variants.

Uni-Object Loco-Manipulation: each env simulates ONE object from `object_names`
(mjlab VariantEntityCfg, round-robin world->variant) and tracks demo clips of
THAT object (OmniObjectMotionCommand in omni mode: env->object from
sim.world_to_variant, per-env clip masking). Registered as Orcs-Uolm (robot
command space) and Orcs-Uolm-Smpl (human SMPL command space).

SONIC-only, ObjKin-only: the policy stream + tokenizer stream come from
mocke.sonic.profile (frozen base I/O contract); the augmentation stream is
base-frame object kinematics + sys1 feedforward commands; the critic is
privileged. Single factory:

  uolm_env_cfg(play=False, ...)
"""

from __future__ import annotations

from functools import lru_cache
from typing import Mapping

import numpy as np
from mjlab.envs import ManagerBasedRlEnvCfg
from mjlab.managers.event_manager import EventTermCfg
from mjlab.managers.observation_manager import ObservationGroupCfg, ObservationTermCfg
from mjlab.managers.reward_manager import RewardTermCfg
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.managers.termination_manager import TerminationTermCfg
from mjlab.scene import SceneCfg
from mjlab.sensor import ContactSensorCfg
from mjlab.sensor.contact_sensor import ContactMatch
from mjlab.sim import MujocoCfg, SimulationCfg
from mjlab.tasks.manipulation.mdp.terminations import illegal_contact
from mjlab.tasks.tracking.mdp import rewards as tracking_rewards
from mjlab.terrains import TerrainEntityCfg
from mjlab.viewer import ViewerConfig
from mocke.sonic import profile

from orcs.assets import (
    OBJECT_BODY_NAME,
    Collision,
    get_g1_flat_hand_cfg,
    omni_object_entity_cfg,
)
from orcs.core.paths import DATA_ROOT
from orcs.tasks.uolm import mdp
from orcs.tasks.uolm.mdp.commands_omni_object import OmniObjectMotionCommandCfg
from orcs.tasks.uolm.mdp.demo_loader import get_motion_files_for_objects
from orcs.tasks.uolm.robustness import apply_robustness

_G1_DATASETS_ROOT = str(DATA_ROOT / "retargeted_motions/data/unitree_g1")
# SMPL command-space dataset (flat <root>/<clip>/<sampleN>/*.npz), built by
# scripts/build_smpl_dataset.py. Absent until the contributor builds it —
# registration degrades gracefully (see _resolve_smpl_motions).
_SMPL_DATASETS_ROOT = str(DATA_ROOT / "smpl_motions")

# fcrl's default roster (assets + motions verified locally). Order matters:
# it is the variant order, i.e. the object-id space.
_DEFAULT_OBJECT_NAMES = (
    "suitcase",
    # "trashcan",
    # "largetable",
    # "plasticbox",
    # "tire",
    # "woodchair2",
)
_EXCLUDE_MOTIONS = ("sub5_suitcase_015", "woodchair2_sit", "custom/tire_flip")

# Collision budget: cvx_dcmp everywhere. Whole-object hulls looked like the
# cheap option but are a narrowphase trap (measured 2026-07-15, 4096 envs):
# container-shaped hulls (trashcan, plasticbox) run ~11x SLOWER than their
# 33-part decompositions at identical contact counts — the hull fills the
# cavity, demo RSI spawns robot parts inside it, and deep penetration on
# smooth convex shapes drives GJK/EPA to worst-case iterations every step.
# Boxy hulls (suitcase) are fine (60k fps) — hull remains a per-object
# override option, not a default.
_DEFAULT_COLLISION: dict[str, Collision] = {}  # all -> "cvx_dcmp"

# Post-motion hold padding (episode — not the motion — owns resets).
_MOTION_PAD_EPS_SEC = 2.0

# --- contact-graph nodes (single source of truth; 1:1 with the demo
# contact_matrix.npz legend names) ---
_CONTACT_GRAPH_BODY_NAMES = (
    "pelvis", "torso_link",
    "left_shoulder_roll_link", "left_elbow_link", "left_wrist_yaw_link",
    "right_shoulder_roll_link", "right_elbow_link", "right_wrist_yaw_link",
    "left_knee_link", "left_ankle_roll_link",
    "right_knee_link", "right_ankle_roll_link",
)
_CONTACT_GRAPH_SENSOR_NAME = "object_contact_graph"

# hand subset of the graph nodes — gates the contact-conditional object
# perturbation (hand contact == control-authority over the object).
_HAND_BODY_NAMES = ("left_wrist_yaw_link", "right_wrist_yaw_link")


@lru_cache(maxsize=None)
def _resolve_motions(
    names: tuple[str, ...], excludes: tuple[str, ...]
) -> tuple[tuple[str, ...], int]:
    """(ordered motion files, longest clip in frames) — cached per roster."""
    _, files = get_motion_files_for_objects(
        list(names), _G1_DATASETS_ROOT, exclude_motions=list(excludes)
    )
    max_len = max(int(np.load(f)["joint_pos"].shape[0]) for f in files)
    return tuple(files), max_len


def _resolve_smpl_motions() -> tuple[str | None, int]:
    """(first clip's motion.npz | None, longest clip frames) for the flat SMPL
    dataset. Graceful when data/smpl_motions is absent/empty — registration must
    not require the (contributor-built) dataset; env build then errors clearly."""
    from orcs.tasks.uolm.mdp.commands_omni_object import _scan_flat_dataset
    try:
        files = _scan_flat_dataset(_SMPL_DATASETS_ROOT)
    except (FileNotFoundError, NotADirectoryError, OSError):
        return None, 500  # ~10 s @ 50 fps placeholder episode length
    max_len = max(int(np.load(f)["joint_pos"].shape[0]) for f in files)
    return files[0], max_len


def _object_contact_graph_sensor(object_entity: str) -> ContactSensorCfg:
    """One object-filtered multi-primary contact sensor for all graph nodes.

    mjlab expands `primary` to P=K primaries in a single sensor, so
    `data.force` is already the batched (B, K, 3) per-body vector. Column
    order is MODEL order (find_bodies), not tuple order — consumers must
    reorder by name via `sensor.primary_names`. Secondary matches by BODY,
    so it is collision-representation agnostic.
    """
    return ContactSensorCfg(
        name=_CONTACT_GRAPH_SENSOR_NAME,
        primary=ContactMatch(
            mode="body", pattern=_CONTACT_GRAPH_BODY_NAMES, entity="robot"),
        secondary=ContactMatch(
            mode="body", pattern=object_entity, entity=object_entity),
        fields=("found", "force"),
        reduce="netforce",
    )


# ---------------------------------------------------------------------------
# Obs groups: policy (frozen SONIC) / augmentation (task) / critic (privileged)
# ---------------------------------------------------------------------------

def _aug_obs_group(obj: SceneEntityCfg, _p: dict) -> ObservationGroupCfg:
    """Augmentation stream (adapter injects into this) — ObjKin feedback +
    sys1 feedforward commands.

    feedback:    base-frame object kin + robot root lin vel (world frame
                 needs odometry — unavailable on hw).
    feedforward: task goal (object_goal_*, fixed per episode, from above
                 sys1) + sys1 command stream {l, v, w}_cmd_t (SUGAR c_t
                 parity, robot-state language).
    """
    return ObservationGroupCfg(
        terms={
            "object_pose_b": ObservationTermCfg(
                func=mdp.object_pose_b, params={"object_cfg": obj}),
            "object_twist_b": ObservationTermCfg(
                func=mdp.object_twist_b, params={"object_cfg": obj}),
            "base_lin_vel": ObservationTermCfg(func=mdp.base_lin_vel),
            "object_goal_ori": ObservationTermCfg(
                func=mdp.object_goal_ori_mat6d, params=_p),
            "object_goal_pos": ObservationTermCfg(
                func=mdp.object_goal_pos_env, params=_p),
            "bodywise_contact_cmd": ObservationTermCfg(
                func=mdp.bodywise_contact_cmd, params=_p),
            "robot_root_lin_vel_cmd": ObservationTermCfg(
                func=mdp.robot_root_lin_vel_cmd, params=_p),
            "robot_root_ang_vel_cmd": ObservationTermCfg(
                func=mdp.robot_root_ang_vel_cmd, params=_p),
        },
        concatenate_terms=True,
        enable_corruption=False,
        nan_policy="sanitize",
        nan_check_per_term=True,
    )


def _critic_obs_group(obj: SceneEntityCfg, _p: dict) -> ObservationGroupCfg:
    """Privileged critic obs: full proprio + object + goal + ref (un-zeroed)."""
    return ObservationGroupCfg(
        terms={
            # tracking command + anchor (privileged: not zeroed)
            "command": ObservationTermCfg(
                func=mdp.generated_commands, params=_p),
            "motion_anchor_pos_b": ObservationTermCfg(
                func=mdp.motion_anchor_pos_b_future, params=_p),
            "motion_anchor_ori_b": ObservationTermCfg(
                func=mdp.motion_anchor_ori_b_future, params=_p),
            # proprio
            "projected_gravity": ObservationTermCfg(func=mdp.projected_gravity),
            "base_lin_vel": ObservationTermCfg(func=mdp.base_lin_vel),
            "base_ang_vel": ObservationTermCfg(func=mdp.base_ang_vel),
            "joint_pos": ObservationTermCfg(func=mdp.joint_pos_rel),
            "joint_vel": ObservationTermCfg(func=mdp.joint_vel_rel),
            "actions": ObservationTermCfg(func=mdp.last_action),
            # object state (env-frame)
            "object_pos_w": ObservationTermCfg(
                func=mdp.object_pos_w_obs, params={"object_cfg": obj}),
            "object_ori_mat6d_w": ObservationTermCfg(
                func=mdp.object_ori_mat6d_w, params={"object_cfg": obj}),
            "object_lin_vel_w": ObservationTermCfg(
                func=mdp.object_lin_vel_w_obs, params={"object_cfg": obj}),
            "object_ang_vel_w": ObservationTermCfg(
                func=mdp.object_ang_vel_w_obs, params={"object_cfg": obj}),
            # object goal (full pose)
            "object_goal_ori": ObservationTermCfg(
                func=mdp.object_goal_ori_mat6d, params=_p),
            "object_goal_pos": ObservationTermCfg(
                func=mdp.object_goal_pos_env, params=_p),
            # object reference trajectory (N-step future, body frame)
            "object_ref_pos_b": ObservationTermCfg(
                func=mdp.motion_object_pos_b_future, params=_p),
            "object_ref_ori_b": ObservationTermCfg(
                func=mdp.motion_object_ori_b_future, params=_p),
            # sys1 command stream (reward-relevant: contact_consistency)
            "bodywise_contact_cmd": ObservationTermCfg(
                func=mdp.bodywise_contact_cmd, params=_p),
            "robot_root_lin_vel_cmd": ObservationTermCfg(
                func=mdp.robot_root_lin_vel_cmd, params=_p),
            "robot_root_ang_vel_cmd": ObservationTermCfg(
                func=mdp.robot_root_ang_vel_cmd, params=_p),
            # reward conditioning: V(concat(s, r_vec))
            "reward_vec": ObservationTermCfg(
                func=mdp.unweighted_reward_vector, params={"enabled": True}),
        },
        concatenate_terms=True,
        enable_corruption=False,
        nan_policy="sanitize",
        nan_check_per_term=True,
    )


def _sonic_obs(
    cfg: ManagerBasedRlEnvCfg, obj: SceneEntityCfg, _p: dict, mode: str = "g1"
) -> None:
    """3-stream obs layout: frozen SONIC streams + augmentation + critic.

    policy = the frozen decoder's proprio stream (history-10, no odometry by
    construction); extra groups = the g1-encoder tokenizer stream (10 future
    ref frames @ 0.1 s) — both from mocke.sonic.profile.
    """
    cfg.observations = {
        "policy": ObservationGroupCfg(
            terms=profile.policy_obs_terms(),
            concatenate_terms=True,
            enable_corruption=True,
            nan_policy="sanitize",
            nan_check_per_term=True,
        ),
        **profile.extra_obs_groups("motion", mode=mode),
        "augmentation": _aug_obs_group(obj, _p),
        "critic": _critic_obs_group(obj, _p),
    }


# ---------------------------------------------------------------------------
# THE factory
# ---------------------------------------------------------------------------

def uolm_env_cfg(
    *,
    command_space: str = "robot",
    play: bool = False,
    object_names: tuple[str, ...] | None = None,
    collision: Collision | Mapping[str, Collision] | None = None,
    num_steps_per_env: int = 24,
) -> ManagerBasedRlEnvCfg:
    """THE Orcs-Uolm env config factory (SONIC augment layout, MoTr rewards).

    command_space="robot": object-keyed omni dataset (retargeted G1 clips).
    command_space="smpl":  flat SMPL dataset (data/smpl_motions, single object);
                           rollout-only (rewards/RSI unsupported, PR pending).
    """
    names = tuple(object_names or _DEFAULT_OBJECT_NAMES)
    obj = SceneEntityCfg(OBJECT_BODY_NAME)
    _p = {"command_name": "motion"}

    if command_space == "smpl":
        print("[orcs.tasks.uolm] Orcs-Uolm-Smpl: rewards + RSI unsupported "
              "(rollout only — PR pending).")
        motion_file, max_clip_len = _resolve_smpl_motions()
        dataset_dir, cmd_object_names, cmd_excludes = _SMPL_DATASETS_ROOT, None, None
        if motion_file is None:  # dataset not built yet — harmless placeholder
            motion_file = _resolve_motions(names, _EXCLUDE_MOTIONS)[0][0]
    else:
        files, max_clip_len = _resolve_motions(names, _EXCLUDE_MOTIONS)
        motion_file, dataset_dir = files[0], _G1_DATASETS_ROOT
        cmd_object_names, cmd_excludes = names, _EXCLUDE_MOTIONS

    cfg = ManagerBasedRlEnvCfg(
        scene=SceneCfg(
            terrain=TerrainEntityCfg(terrain_type="plane"),
            entities={
                OBJECT_BODY_NAME: omni_object_entity_cfg(
                    names, collision or _DEFAULT_COLLISION),
            },
            num_envs=1,
            env_spacing=6.0,  # largetable-sized objects need clearance
        ),
        observations={},  # set by _sonic_obs below
        actions={},       # set below (SONIC profile)
        commands={},
        events={
            "reset_default": EventTermCfg(
                func=mdp.reset_scene_to_default, mode="reset"),
            "policy_update_counter": EventTermCfg(
                func=mdp.PolicyUpdateCounter,
                mode="interval",
                interval_range_s=(0.0, 0.0),
                is_global_time=True,
                params={"num_steps_per_env": num_steps_per_env},
            ),
            # VOF: decaying PD wrench pushing the object toward the demo ref.
            # Per-world mass/inertia scaling comes from the variant model
            # (build_variant_model populates body_mass/body_inertia per world).
            "virtual_object_force": EventTermCfg(
                func=mdp.VirtualObjectForceCurriculum,
                mode="interval",
                interval_range_s=(0.0, 0.0),
                params={
                    "natural_frequency": 12.0,
                    "terminal_scale": 1e-4,
                    "decay_mode": "exponential",
                    "decay_by_policy_iterations": 10_000,
                    "object_cfg": obj,
                    **_p,
                },
            ),
        },
        rewards={},       # filled below
        terminations={
            "time_out": TerminationTermCfg(func=mdp.time_out, time_out=True),
            "illegal_contact": TerminationTermCfg(func=illegal_contact),
        },
        viewer=ViewerConfig(
            origin_type=ViewerConfig.OriginType.WORLD,
            distance=4.0,
            elevation=-20.0,
            azimuth=135.0,
        ),
        # VRAM budget: contact/efc/CCD arenas scale with nconmax*nworld —
        # CCD-row cap lives in _mjlab_compat._patch_put_data_nccdmax.
        # Undershoot is loud: mjwarp printf's contact/CCD overflow to stderr.
        sim=SimulationCfg(
            nconmax=150,
            njmax=450,
            mujoco=MujocoCfg(timestep=0.005, iterations=10, ls_iterations=20,
                             ccd_iterations=24),  # EPA scratch ~ 6+5*it rows
        ),
        decimation=4,
        episode_length_s=10.0,  # overwritten below from the dataset
    )

    # ── SONIC robot + action (flat-hand G1, hip_pitch regroup, MJ-order) ──
    robot = profile.robot_cfg(base=get_g1_flat_hand_cfg())
    cfg.scene.entities["robot"] = robot
    cfg.actions["joint_pos"] = profile.action_cfg(robot)

    # Loco-manip legitimately kneels/braces (knees, thighs, forearms on the
    # ground while lifting) — terrain kill-switch covers the upper-body core
    # only, not "everything but feet".
    cfg.scene.sensors = (cfg.scene.sensors or ()) + (
        ContactSensorCfg(
            name="torso_terrain_contact",
            primary=ContactMatch(
                mode="geom",
                pattern=("pelvis_collision", "torso_collision",
                         ".*shoulder.*_collision"),
                entity="robot",
            ),
            secondary=ContactMatch(mode="geom", pattern="terrain"),
            fields=("found",),
        ),
    )
    cfg.terminations["illegal_contact"].params["sensor_name"] = "torso_terrain_contact"

    # ── motion command (omni mode) + object contact-graph sensor ──
    cfg.commands["motion"] = OmniObjectMotionCommandCfg(
        motion_file=motion_file,
        dataset_dir=dataset_dir,
        ordered_object_names=cmd_object_names,
        exclude_motions=cmd_excludes,
        object_entity_name=OBJECT_BODY_NAME,
        command_space=command_space,
        future_steps=5,
        resampling_time_range=(1e9, 1e9),
        debug_vis=True,
        pose_range={},
        velocity_range={},
        joint_position_range=(0.0, 0.0),
        contact_graph_body_names=_CONTACT_GRAPH_BODY_NAMES,
        contact_graph_sensor_name=_CONTACT_GRAPH_SENSOR_NAME,
    )
    cfg.scene.sensors = cfg.scene.sensors + (
        _object_contact_graph_sensor(OBJECT_BODY_NAME),
    )

    # episode = longest clip + ε hold padding (episode owns resets)
    step_dt = cfg.sim.mujoco.timestep * cfg.decimation
    cfg.episode_length_s = max_clip_len * step_dt + _MOTION_PAD_EPS_SEC
    cfg.terminations.update({
        "bad_anchor_pos": TerminationTermCfg(
            func=mdp.bad_anchor_pos, params={**_p, "threshold": 0.3}),
        "bad_anchor_ori": TerminationTermCfg(
            func=mdp.bad_anchor_ori, params={**_p, "threshold": 0.8}),
        "bad_object_pos": TerminationTermCfg(
            func=mdp.bad_object_pos, params={**_p, "threshold": 0.3}),
        "bad_object_ori": TerminationTermCfg(
            func=mdp.bad_object_ori, params={**_p, "threshold": 0.8}),
        "exceeded_motion": TerminationTermCfg(
            func=mdp.exceeded_motion_by_eps, time_out=True,
            params={**_p, "epsilon_steps": int(_MOTION_PAD_EPS_SEC / step_dt)}),
    })

    # ── rewards: fcrl parity — robot tracking + object tracking + pose goal ──
    cfg.rewards = {
        "object_goal": RewardTermCfg(
            func=mdp.object_goal_pose_reward,
            weight=0.5, params={"object_cfg": obj, **_p,
                                "std_pos": 0.3, "std_quat": 0.4}),
        "object_pos": RewardTermCfg(
            func=mdp.object_pos_tracking_reward,
            weight=2.0, params={**_p, "std": 0.3}),
        "object_ori": RewardTermCfg(
            func=mdp.object_ori_tracking_reward,
            weight=1.0, params={**_p, "std": 0.4}),
        "root_pos": RewardTermCfg(
            func=tracking_rewards.motion_global_anchor_position_error_exp,
            weight=0.5, params={**_p, "std": 0.3}),
        "root_ori": RewardTermCfg(
            func=tracking_rewards.motion_global_anchor_orientation_error_exp,
            weight=0.5, params={**_p, "std": 0.4}),
        "body_pos": RewardTermCfg(
            func=tracking_rewards.motion_relative_body_position_error_exp,
            weight=1.0, params={**_p, "std": 0.3}),
        "body_ori": RewardTermCfg(
            func=tracking_rewards.motion_relative_body_orientation_error_exp,
            weight=1.0, params={**_p, "std": 0.4}),
        "body_lin_vel": RewardTermCfg(
            func=tracking_rewards.motion_global_body_linear_velocity_error_exp,
            weight=1.0, params={**_p, "std": 1.0}),
        "body_ang_vel": RewardTermCfg(
            func=tracking_rewards.motion_global_body_angular_velocity_error_exp,
            weight=1.0, params={**_p, "std": 3.14}),
        "contact_consistency": RewardTermCfg(
            func=mdp.object_contact_consistency,
            weight=1.0, params={**_p,
                                "sensor_name": _CONTACT_GRAPH_SENSOR_NAME,
                                "contact_force_threshold": 0.1}),
        "action_rate_l2": RewardTermCfg(func=mdp.action_rate_l2, weight=-0.1),
        "joint_pos_limits": RewardTermCfg(func=mdp.joint_pos_limits, weight=-1.0),
    }

    # ── obs: 3-stream layout (policy + tokenizer / augmentation / critic) ──
    _sonic_obs(cfg, obj, _p, mode="smpl" if command_space == "smpl" else "g1")

    if command_space == "smpl":
        # Rollout-only for now: rewards + RSI design deferred (frozen base,
        # zero-adapt). No tracking kills / robustness / VOF — just play.
        cfg.rewards = {}
        cfg.observations["critic"].terms.pop("reward_vec")
        cfg.events.pop("virtual_object_force")
        for k in ("bad_anchor_pos", "bad_anchor_ori",
                  "bad_object_pos", "bad_object_ori"):
            cfg.terminations.pop(k)
    else:
        # ── robustness domain: state (isr + pushes) + param (physical DR) ──
        apply_robustness(
            cfg, object_name=OBJECT_BODY_NAME,
            sensor_name=_CONTACT_GRAPH_SENSOR_NAME,
            hand_body_names=_HAND_BODY_NAMES,
        )

    if play:
        _play_overrides(cfg)

    return cfg


def _play_overrides(cfg: ManagerBasedRlEnvCfg) -> None:
    """Play-mode overrides: no corruption, no anneal/VOF, no tracking kills."""
    cfg.observations["policy"].enable_corruption = False
    for event in ("policy_update_counter", "virtual_object_force"):
        cfg.events.pop(event, None)
    for k in ("bad_anchor_pos", "bad_anchor_ori", "bad_object_pos", "bad_object_ori"):
        cfg.terminations.pop(k, None)
    cfg.commands["motion"].start_from_zero = True
