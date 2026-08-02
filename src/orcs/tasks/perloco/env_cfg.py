"""PerLoco env config — frozen SONIC WBC adapter over a staged terrain grid.

Perceptive Locomotion: the scene is ONE sub-terrain grid built from staged
(terrain, motion) pairs, each env stands on one tile, and the clips it tracks
are the ones staged against that tile. The adapter's conditioning stream is a
terrain height scan where uolm's is object kinematics — that one swap is the
task.

Single factory:

  perloco_env_cfg(agent="sonic"|"tara", play=False, ...)

**Memory.** mjlab replicates the whole model `nworld = num_envs` times, so
robots never share a world and terrain contact is always 1 robot x terrain,
never n x n. What every world DOES carry is the full grid — 145 tiles is ~415
static box geoms, and static pairs are culled in broadphase, so the cost is
model size (once) rather than contacts (per step). Boxes are what keep that
true: the same terrain as meshes or heightfields would be the whole budget.
"""

from __future__ import annotations

from functools import lru_cache
from typing import Callable

import numpy as np
from mjlab.entity import EntityCfg
from mjlab.envs import ManagerBasedRlEnvCfg
from mjlab.managers.event_manager import EventTermCfg
from mjlab.managers.reward_manager import RewardTermCfg
from mjlab.managers.termination_manager import TerminationTermCfg
from mjlab.scene import SceneCfg
from mjlab.sim import MujocoCfg, SimulationCfg
from mjlab.tasks.manipulation.mdp.terminations import illegal_contact
from mjlab.tasks.tracking.mdp import rewards as tracking_rewards
from mjlab.terrains import TerrainEntityCfg
from mjlab.viewer import ViewerConfig
from mocke.sonic import profile

from orcs.assets import get_g1_flat_hand_cfg
from orcs.core.paths import DATA_ROOT
from orcs.tasks.perloco import mdp
from orcs.tasks.perloco.mdp.commands import TerrainMotionCommandCfg
from orcs.tasks.perloco.observation_cfgs import ObsCtx, sonic_obs, tara_obs
from orcs.tasks.perloco.sensors import (
    PERLOCO_KILL_BODIES,
    TERRAIN_CONTACT_SENSOR_NAME,
    terrain_contact_sensor,
    terrain_scan_sensor,
)
from orcs.tasks.perloco.terrain import (
    TILE_SIZE,
    staged_roster,
    terrain_generator_cfg,
)

DEFAULT_SOURCE = "omni"
"""Staged source under `data/terrain_motions/`. A second source is a second
registration, not a second code path — see `sources/`."""

_MOTION_PAD_EPS_SEC = 2.0
"""Post-motion hold padding (episode — not the motion — owns resets)."""


def staged_root(source: str = DEFAULT_SOURCE):
    return DATA_ROOT / "terrain_motions" / source


@lru_cache(maxsize=None)
def _resolve(
    source: str, families: tuple[str, ...] | None, levels: tuple[float, ...] | None,
    excludes: tuple[str, ...],
) -> tuple[tuple[str, ...], tuple[float, ...], str, int]:
    """(families, levels, first clip, longest clip in frames) — cached per roster.

    The roster is resolved ONCE and handed to both the terrain builder and the
    command, so the grid axes and the clip index space cannot disagree.
    """
    from orcs.core.data.scan import scan_flat

    root = staged_root(source)
    fam, lvl = staged_roster(root, families, levels)
    files = scan_flat(str(root), excludes or None)
    max_len = max(int(np.load(f)["joint_pos"].shape[0]) for f in files)
    return fam, lvl, files[0], max_len


# ---------------------------------------------------------------------------
# THE factory
# ---------------------------------------------------------------------------

def perloco_env_cfg(
    *,
    agent: str = "sonic",
    play: bool = False,
    source: str = DEFAULT_SOURCE,
    families: tuple[str, ...] | None = None,
    levels: tuple[float, ...] | None = None,
    exclude_motions: tuple[str, ...] = (),
    scan_frame: str = "pelvis",
    tile_size: tuple[float, float] = TILE_SIZE,
    num_steps_per_env: int = 24,
    robot_cfg: Callable[[], EntityCfg] | None = None,
    kill_bodies: tuple[str, ...] = PERLOCO_KILL_BODIES,
    kill_exclude: tuple[str, ...] = (),
) -> ManagerBasedRlEnvCfg:
    """THE Orcs-PerLoco env config factory.

    agent="sonic": frozen SONIC base + LoRA adapter (3-stream obs). THE task.
    agent="tara":  tabula rasa, from-scratch MLP (2-stream obs) — the
                   no-frozen-base floor to measure the adapter against.

    Injection points for a downstream consumer (§ethos: adapt, don't fork):
      families/levels  restrict the grid to a sub-roster (must stay
                       rectangular — see `terrain.staged_roster`).
      scan_frame       which body the height scan hangs off. An open design
                       question, not a settled default — see `sensors.py`.
      robot_cfg        the G1 variant to build on (physics is identical across
                       variants; this picks the visual set).
      kill_bodies      which robot geoms on the terrain end the episode.
                       Default is the ROOT ALONE: climbing loads hands, knees
                       and forearms onto terrain by design.
    """
    assert agent in ("sonic", "tara"), f"unknown agent {agent!r}"

    fam, lvl, motion_file, max_clip_len = _resolve(
        source, families, levels, tuple(exclude_motions))
    root = staged_root(source)
    _p = {"command_name": "motion"}

    cfg = ManagerBasedRlEnvCfg(
        scene=SceneCfg(
            terrain=TerrainEntityCfg(
                terrain_type="generator",
                terrain_generator=terrain_generator_cfg(
                    root, fam, lvl, size=tile_size),
            ),
            num_envs=1,
        ),
        observations={},  # set below
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
        },
        rewards={},       # filled below
        terminations={
            "time_out": TerminationTermCfg(func=mdp.time_out, time_out=True),
            "illegal_contact": TerminationTermCfg(
                func=illegal_contact,
                params={"sensor_name": TERRAIN_CONTACT_SENSOR_NAME}),
        },
        viewer=ViewerConfig(
            origin_type=ViewerConfig.OriginType.WORLD,
            distance=6.0,
            elevation=-20.0,
            azimuth=135.0,
        ),
        # Terrain is boxes, so contacts are box-box: cheap, few, and bounded by
        # the robot's own geom count rather than by the grid's.
        sim=SimulationCfg(
            nconmax=100,
            njmax=300,
            mujoco=MujocoCfg(timestep=0.005, iterations=10, ls_iterations=20),
        ),
        decimation=4,
        episode_length_s=10.0,  # overwritten below from the dataset
    )

    # ── SONIC robot + action (flat-hand G1, hip_pitch regroup, MJ-order) ──
    robot = profile.robot_cfg(base=(robot_cfg or get_g1_flat_hand_cfg)())
    cfg.scene.entities["robot"] = robot
    cfg.actions["joint_pos"] = profile.action_cfg(robot)

    cfg.scene.sensors = (cfg.scene.sensors or ()) + (
        terrain_contact_sensor(kill_bodies, kill_exclude),
        terrain_scan_sensor(scan_frame, debug_vis=play),
    )

    # ── motion command: clips masked by the tile the env stands on ──
    cfg.commands["motion"] = TerrainMotionCommandCfg(
        motion_file=motion_file,
        dataset_dir=str(root),
        exclude_motions=exclude_motions or None,
        families=fam,
        levels=lvl,
        future_steps=5,
        resampling_time_range=(1e9, 1e9),
        debug_vis=True,
        pose_range={},
        velocity_range={},
        joint_position_range=(0.0, 0.0),
    )

    # episode = longest clip + ε hold padding (episode owns resets)
    step_dt = cfg.sim.mujoco.timestep * cfg.decimation
    cfg.episode_length_s = max_clip_len * step_dt + _MOTION_PAD_EPS_SEC
    cfg.terminations.update({
        # Anchor tubes are BACK (uolm drops them): with no object dragging the
        # root off the reference, a robot far from its anchor is simply failing
        # to track, and letting it run wastes the episode. Loose enough for the
        # large vertical excursions climbing produces.
        "bad_anchor_pos": TerminationTermCfg(
            func=mdp.bad_anchor_pos, params={**_p, "threshold": 0.4}),
        "bad_anchor_ori": TerminationTermCfg(
            func=mdp.bad_anchor_ori, params={**_p, "threshold": 0.8}),
        "exceeded_motion": TerminationTermCfg(
            func=mdp.exceeded_motion_by_eps, time_out=True,
            params={**_p, "epsilon_steps": int(_MOTION_PAD_EPS_SEC / step_dt)}),
    })

    # ── rewards: pure motion tracking (no object, so no object terms) ──
    cfg.rewards = {
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
        "action_rate_l2": RewardTermCfg(func=mdp.action_rate_l2, weight=-0.1),
        "joint_pos_limits": RewardTermCfg(func=mdp.joint_pos_limits, weight=-1.0),
    }

    # ── obs: 3-stream (frozen base) or 2-stream (tabula rasa) ──
    ctx = ObsCtx(p=_p)
    cfg.observations = tara_obs(ctx) if agent == "tara" else sonic_obs(ctx)

    if play:
        _play_overrides(cfg)

    return cfg


def _play_overrides(cfg: ManagerBasedRlEnvCfg) -> None:
    """Play-mode overrides: no corruption, no anneal, no tracking kills.

    The tracking kills go so a `--agent initial` rollout survives long enough
    to SEE where it fails; the terrain kill (`illegal_contact`) stays, because
    a pelvis on the ground is exactly what you want to notice.
    """
    cfg.observations["policy"].enable_corruption = False
    cfg.events.pop("policy_update_counter", None)
    for k in ("bad_anchor_pos", "bad_anchor_ori"):
        cfg.terminations.pop(k, None)
    cfg.commands["motion"].start_from_zero = True
