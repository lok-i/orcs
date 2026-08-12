"""UOLM env config — frozen SONIC WBC adapter over per-world object variants.

Uni-Object Loco-Manipulation: each env simulates ONE object from `object_names`
(mjlab VariantEntityCfg, round-robin world->variant) and tracks demo clips of
THAT object (ObjectMotionCommand in omni mode: env->object from
sim.world_to_variant, per-env clip masking). Registered as Orcs-Uolm-AdaptSonic (robot
command space) and Orcs-Uolm-AdaptSonic-Smpl (human SMPL command space).

SONIC-only, ObjKin-only: the policy stream + tokenizer stream come from
mocke.sonic.profile (frozen base I/O contract); the augmentation stream is
base-frame object kinematics + sys1 feedforward commands; the critic is
privileged. Single factory:

  uolm_env_cfg(play=False, ...)
"""

from __future__ import annotations

from functools import lru_cache
from typing import Callable, Mapping

import numpy as np
from mjlab.entity import EntityCfg
from mjlab.envs import ManagerBasedRlEnvCfg
from mjlab.managers.event_manager import EventTermCfg
from mjlab.managers.reward_manager import RewardTermCfg
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.managers.termination_manager import TerminationTermCfg
from mjlab.scene import SceneCfg
from mjlab.sim import MujocoCfg, SimulationCfg
from mjlab.tasks.manipulation.mdp.terminations import illegal_contact
from mjlab.tasks.tracking.mdp import rewards as tracking_rewards
from mjlab.terrains import TerrainEntityCfg
from mjlab.viewer import ViewerConfig
from mocke.sonic import profile

from orcs.assets import (
    OBJECT_BODY_NAME,
    TABLE_CENTER_HEIGHT,
    Collision,
    get_g1_flat_hand_cfg,
    omni_object_entity_cfg,
    reconstructed_object_entity_cfg,
    reconstructed_object_variants_entity_cfg,
    table_entity_cfg,
)
from orcs.core.data.seeds import SeedMotion
from orcs.core.obs import apply_obs_noise
from orcs.core.paths import DATA_ROOT
from orcs.core.robustness import strip_domain
from orcs.tasks.uolm import mdp
from orcs.tasks.uolm.mdp.commands import (
    ObjectMotionCommandCfg,
    SmplSeedObjectMotionCommandCfg,
)
from orcs.tasks.uolm.mdp.demo_loader import get_motion_files_for_objects
from orcs.tasks.uolm.observation_cfgs import ObsCtx, sonic_obs, tara_obs
from orcs.tasks.uolm.robustness import apply_robustness
from orcs.tasks.uolm.sensors import (
    CONTACT_GRAPH_BODY_NAMES,
    CONTACT_GRAPH_SENSOR_NAME,
    GROUND_CONTACT_SENSOR_NAME,
    HAND_BODY_NAMES,
    UOLM_KILL_BODIES,
    ground_contact_sensor,
    object_contact_graph_sensor,
)
from orcs.tasks.uolm.sources.reconstructed import MOTION_SETS, cache_root

_G1_DATASETS_ROOT = str(DATA_ROOT / "retargeted_motions/data/unitree_g1")
# SMPL command-space dataset (flat <root>/<clip>/<sampleN>/*.npz), built by
# scripts/build_smpl_dataset.py. Absent until the contributor builds it —
# registration degrades gracefully (see _resolve_smpl_motions).
_SMPL_DATASETS_ROOT = str(DATA_ROOT / "smpl_motions")
_RECONSTRUCTED_SMPL_ROOT = str(cache_root())

# fcrl's default roster (assets + motions verified locally). Order matters:
# it is the variant order, i.e. the object-id space.
_DEFAULT_OBJECT_NAMES = (
    "suitcase",
    "trashcan",
    "largetable",
    "plasticbox",
    "tire",
    "woodchair2",
)
_EXCLUDE_MOTIONS = ("sub5_suitcase_015", 
                    "woodchair2_sit", 

                    "tire_flip",
                    "custom/tire_roll",
                    "sugar/tire_roll/sample1",
                    "sugar/tire_roll/sample3",

                    "custom/woodchair2_flip/sample1",
                    "custom/woodchair2_flip/sample3",
                    )

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
    from orcs.tasks.uolm.mdp.commands import _scan_flat_dataset
    try:
        files = _scan_flat_dataset(_SMPL_DATASETS_ROOT)
    except (FileNotFoundError, NotADirectoryError, OSError):
        return None, 500  # ~10 s @ 50 fps placeholder episode length
    max_len = max(int(np.load(f)["joint_pos"].shape[0]) for f in files)
    return files[0], max_len


@lru_cache(maxsize=None)
def _resolve_reconstructed_smpl_seeds(
    motion_sets: tuple[str, ...],
) -> tuple[str, int, int]:
    """Return first locator, complete clip count, and longest seed length."""
    unknown = tuple(name for name in motion_sets if name not in MOTION_SETS)
    if unknown:
        raise ValueError(
            f"unknown reconstructed motion sets {unknown}; choose from "
            f"{tuple(MOTION_SETS)}"
        )
    if not motion_sets or len(set(motion_sets)) != len(motion_sets):
        raise ValueError("motion_sets must be non-empty and contain no duplicates")

    ready_samples: list[tuple[str, int]] = []
    for motion_set in motion_sets:
        root = cache_root() / motion_set
        ready = 0
        for smpl_file in sorted(root.rglob("smpl_motion.npz")):
            sample = smpl_file.parent
            if not (sample / "object_motion.npz").exists():
                continue
            seed_file = sample / "seed_state.npz"
            if not seed_file.exists():
                continue
            seed = SeedMotion.load(seed_file)
            if not seed.valid.all() or seed.object_pos_w is None:
                continue
            ready_samples.append((str(smpl_file), seed.num_frames))
            ready += 1
        if ready == 0:
            raise FileNotFoundError(
                f"no complete kinematic retargets under {root}; run "
                "orcs-pseudo-retarget --scene uolm "
                f"--motion-set {motion_set} --all"
            )
    return ready_samples[0][0], len(ready_samples), max(
        length for _, length in ready_samples
    )


# ---------------------------------------------------------------------------
# THE factory
# ---------------------------------------------------------------------------

def uolm_env_cfg(
    *,
    command_space: str = "robot",
    agent: str = "sonic",
    play: bool = False,
    object_names: tuple[str, ...] | None = None,
    collision: Collision | Mapping[str, Collision] | None = None,
    num_steps_per_env: int = 24,
    robot_cfg: Callable[[], EntityCfg] | None = None,
    kill_bodies: tuple[str, ...] = UOLM_KILL_BODIES,
    kill_exclude: tuple[str, ...] = (),
    _bootstrap_motion: tuple[str, str, int] | None = None,
    _object_entity: EntityCfg | None = None,
) -> ManagerBasedRlEnvCfg:
    """THE Orcs-Uolm-AdaptSonic env config factory (SONIC augment layout, MoTr rewards).

    command_space="robot": object-keyed omni dataset (retargeted G1 clips).
    command_space="smpl":  flat SMPL dataset (data/smpl_motions, single object);
                           rollout-only (rewards/RSI unsupported, PR pending).

    agent="sonic": frozen SONIC base + LoRA adapter (3-stream obs).
    agent="tara":  tabula rasa, from-scratch MLP (2-stream obs) — the
                   no-frozen-base baseline. smpl needs the SONIC smpl encoder,
                   so that pairing is rejected.

    Injection points for a downstream consumer (§ethos: adapt, don't fork):
      robot_cfg   the G1 variant to build on — physics is identical across
                  variants, so this only picks the visual set (a consumer with
                  a camera wants out-of-frame meshes demoted).
      kill_bodies which robot geoms ending up on the terrain terminate the
                  episode. Default is the root link alone (UOLM_KILL_BODIES,
                  fcrl parity); pass LOCOMANIP_KILL_BODIES for the upper-body
                  core, or STRICT_KILL_BODIES + exclude for a task where
                  nothing but the feet should touch down.
    """
    assert agent in ("sonic", "tara"), f"unknown agent {agent!r}"
    assert not (agent == "tara" and command_space == "smpl"), (
        "the smpl command space rides the SONIC smpl encoder — no tabula-rasa variant")

    names = tuple(object_names or _DEFAULT_OBJECT_NAMES)
    obj = SceneEntityCfg(OBJECT_BODY_NAME)
    _p = {"command_name": "motion"}

    if _bootstrap_motion is not None:
        # Internal composition seam for source-only pipelines. The command and
        # scene entity are replaced by their specialized factory before env
        # build; supplying the bootstrap explicitly prevents an accidental
        # dependency on the legacy retargeted-motion corpus.
        motion_file, dataset_dir, max_clip_len = _bootstrap_motion
        cmd_object_names, cmd_excludes = None, None
    elif command_space == "smpl":
        # rollout-only (rewards + RSI nullified below); documented in
        # tasks/uolm/__init__ rather than printed at every import.
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
                OBJECT_BODY_NAME: _object_entity or omni_object_entity_cfg(
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
    robot = profile.robot_cfg(base=(robot_cfg or get_g1_flat_hand_cfg)())
    cfg.scene.entities["robot"] = robot
    cfg.actions["joint_pos"] = profile.action_cfg(robot)

    cfg.scene.sensors = (cfg.scene.sensors or ()) + (
        ground_contact_sensor(kill_bodies, kill_exclude),
    )
    cfg.terminations["illegal_contact"].params["sensor_name"] = (
        GROUND_CONTACT_SENSOR_NAME)

    # ── motion command (omni mode) + object contact-graph sensor ──
    cfg.commands["motion"] = ObjectMotionCommandCfg(
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
        contact_graph_body_names=CONTACT_GRAPH_BODY_NAMES,
        contact_graph_sensor_name=CONTACT_GRAPH_SENSOR_NAME,
    )
    cfg.scene.sensors = cfg.scene.sensors + (
        object_contact_graph_sensor(OBJECT_BODY_NAME),
    )

    # episode = longest clip + ε hold padding (episode owns resets)
    step_dt = cfg.sim.mujoco.timestep * cfg.decimation
    cfg.episode_length_s = max_clip_len * step_dt + _MOTION_PAD_EPS_SEC
    # OBJECT tracking tubes only — fcrl parity (2026-08-01). The robot anchor
    # kills (`bad_anchor_{pos,ori}`) are the pure-tracking layer's and fcrl's
    # uolm dropped them: under loco-manip the object legitimately drags the
    # root off the reference, so an anchor tube kills recoverable states and
    # truncates every episode before the goal earns credit. `bad_object_ori`
    # is back to fcrl's 0.6 (0.8 let the object tumble past recovery).
    cfg.terminations.update({
        "bad_object_pos": TerminationTermCfg(
            func=mdp.bad_object_pos, params={**_p, "threshold": 0.3}),
        "bad_object_ori": TerminationTermCfg(
            func=mdp.bad_object_ori, params={**_p, "threshold": 0.6}),
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
                                "sensor_name": CONTACT_GRAPH_SENSOR_NAME,
                                "contact_force_threshold": 0.1}),
        "action_rate_l2": RewardTermCfg(func=mdp.action_rate_l2, weight=-0.1),
        "joint_pos_limits": RewardTermCfg(func=mdp.joint_pos_limits, weight=-1.0),
    }

    # ── obs: 3-stream layout (policy + tokenizer / augmentation / critic) ──
    ctx = ObsCtx(obj=obj, p=_p)
    cfg.observations = (
        tara_obs(ctx) if agent == "tara"
        else sonic_obs(ctx, mode="smpl" if command_space == "smpl" else "g1")
    )

    if command_space == "smpl":
        # Rollout-only for now: rewards + RSI design deferred (frozen base,
        # zero-adapt). No tracking kills / robustness / VOF — just play.
        cfg.rewards = {}
        cfg.observations["critic"].terms.pop("reward_vec")
        cfg.events.pop("virtual_object_force")
        for k in ("bad_object_pos", "bad_object_ori"):
            cfg.terminations.pop(k)
    else:
        # ── robustness domain: state (isr + pushes) + param (physical DR),
        #    both halves — this is the task that has an object ──
        apply_robustness(
            cfg, object_name=OBJECT_BODY_NAME,
            sensor_name=CONTACT_GRAPH_SENSOR_NAME,
            hand_body_names=HAND_BODY_NAMES,
        )
    apply_obs_noise(cfg)

    # INVARIANT: play overrides are LAST — they SUBTRACT from the assembled
    # domain, so anything wired below this line leaks into play/eval.
    if play:
        _play_overrides(cfg)

    return cfg


def uolm_smpl_env_cfg(
    *,
    play: bool = False,
    motion_sets: tuple[str, ...] = (
        "small-cube-table",
        "big-cube-floor",
    ),
    num_steps_per_env: int = 24,
    robot_cfg: Callable[[], EntityCfg] | None = None,
    kill_bodies: tuple[str, ...] = UOLM_KILL_BODIES,
    kill_exclude: tuple[str, ...] = (),
    point_anchor_threshold: float = 0.75,
) -> ManagerBasedRlEnvCfg:
    """Reconstructed SMPL/object tracking with kinematic-retarget RSI.

    ``motion_sets`` is the only dataset selector: one name produces a
    specialized homogeneous batch; multiple names produce matched per-world
    object variants and clip masks.  Seed robot/object states initialize the
    simulator, while rewards refer only to source SMPL points and source
    object motion.
    """
    motion_sets = tuple(motion_sets)
    bootstrap_file, _, max_clip_len = _resolve_reconstructed_smpl_seeds(motion_sets)

    if len(motion_sets) == 1:
        object_entity = reconstructed_object_entity_cfg(motion_sets[0])
        ordered_motion_sets = None
    else:
        object_entity = reconstructed_object_variants_entity_cfg(motion_sets)
        ordered_motion_sets = motion_sets

    # Reuse the proven UOLM physics/action/event shell. Everything specific to
    # its retargeted robot demonstrations is replaced below before env build.
    cfg = uolm_env_cfg(
        command_space="robot",
        agent="sonic",
        play=False,
        num_steps_per_env=num_steps_per_env,
        robot_cfg=robot_cfg,
        kill_bodies=kill_bodies,
        kill_exclude=kill_exclude,
        _bootstrap_motion=(
            bootstrap_file,
            _RECONSTRUCTED_SMPL_ROOT,
            max_clip_len,
        ),
        _object_entity=object_entity,
    )
    # One movable fixed support is cheaper and more exact than a scene switch:
    # command reset puts it below the world for floor clips, or beneath the
    # selected table clip's authored final object XY.
    cfg.scene.entities["table"] = table_entity_cfg()

    bootstrap_motion = cfg.commands["motion"].motion_file
    cfg.commands["motion"] = SmplSeedObjectMotionCommandCfg(
        motion_file=bootstrap_motion,
        dataset_dir=_RECONSTRUCTED_SMPL_ROOT,
        motion_set_names=motion_sets,
        ordered_object_names=ordered_motion_sets,
        exclude_motions=None,
        object_entity_name=OBJECT_BODY_NAME,
        support_entity_name="table",
        table_center_height=TABLE_CENTER_HEIGHT,
        command_space="smpl",
        future_steps=5,
        resampling_time_range=(1e9, 1e9),
        debug_vis=True,
        pose_range={},
        velocity_range={},
        joint_position_range=(0.0, 0.0),
        # Reconstructed clips carry no authored robot contact schedule. The
        # live sensor remains for perturbations, but no fake contact labels are
        # introduced into observations or rewards.
        contact_graph_body_names=None,
        contact_graph_sensor_name=None,
    )

    p = {"command_name": "motion"}
    obj = SceneEntityCfg(OBJECT_BODY_NAME)
    cfg.rewards = {
        "object_goal": RewardTermCfg(
            func=mdp.object_goal_pose_reward,
            weight=0.5,
            params={"object_cfg": obj, **p, "std_pos": 0.3, "std_quat": 0.4},
        ),
        "object_pos": RewardTermCfg(
            func=mdp.object_pos_tracking_reward,
            weight=2.0,
            params={**p, "std": 0.3},
        ),
        "object_ori": RewardTermCfg(
            func=mdp.object_ori_tracking_reward,
            weight=1.0,
            params={**p, "std": 0.4},
        ),
        "point_pos": RewardTermCfg(
            func=mdp.point_position_error_exp,
            weight=4.0,
            params={**p, "std": 0.3},
        ),
        "point_vel": RewardTermCfg(
            func=mdp.point_velocity_error_exp,
            weight=1.0,
            params={**p, "std": 1.0},
        ),
        "action_rate_l2": RewardTermCfg(func=mdp.action_rate_l2, weight=-0.1),
        "joint_pos_limits": RewardTermCfg(func=mdp.joint_pos_limits, weight=-1.0),
    }
    cfg.terminations["bad_point_anchor_pos"] = TerminationTermCfg(
        func=mdp.bad_point_anchor_pos,
        params={**p, "threshold": point_anchor_threshold},
    )
    step_dt = cfg.sim.mujoco.timestep * cfg.decimation
    cfg.episode_length_s = max_clip_len * step_dt + _MOTION_PAD_EPS_SEC
    cfg.terminations["exceeded_motion"].params["epsilon_steps"] = int(
        _MOTION_PAD_EPS_SEC / step_dt
    )

    ctx = ObsCtx(obj=obj, p=p)
    cfg.observations = sonic_obs(ctx, mode="smpl", include_contact=False)

    # Re-apply after replacing the motion command and observation groups. The
    # SMPL command uses clip-start object pose noise but deliberately has no
    # invented mid-clip reference-contact gate.
    apply_robustness(
        cfg,
        object_name=OBJECT_BODY_NAME,
        sensor_name=CONTACT_GRAPH_SENSOR_NAME,
        hand_body_names=HAND_BODY_NAMES,
    )
    cfg.commands["motion"].object_in_contact_velocity_range = None
    apply_obs_noise(cfg)

    if play:
        _play_overrides(cfg)
        cfg.terminations.pop("bad_point_anchor_pos", None)

    return cfg


def _play_overrides(cfg: ManagerBasedRlEnvCfg) -> None:
    """Play-mode overrides: no domain, no anneal/VOF, no tracking kills.

    `strip_domain` is a prefix match over `perturb_*`/`rand_*`, so it now takes
    the PARAM half too. Those events used to survive play (the list named only
    the anneal terms), which handed every eval rollout a randomized mass,
    friction, torso COM and a biased joint encoder.
    """
    strip_domain(cfg)
    for event in ("policy_update_counter", "virtual_object_force"):
        cfg.events.pop(event, None)
    for k in ("bad_object_pos", "bad_object_ori"):
        cfg.terminations.pop(k, None)
    # cfg.commands["motion"].start_from_zero = True
    # remove the intial statn randomization in motion
    cfg.commands["motion"].pose_range = {}
    cfg.commands["motion"].velocity_range = {}
    cfg.commands["motion"].joint_position_range = (0.0, 0.0)
    cfg.commands["motion"].object_init_pose_range = {}
    cfg.commands["motion"].object_in_contact_velocity_range = {}
