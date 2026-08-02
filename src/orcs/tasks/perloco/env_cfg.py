"""PerLoco envs — frozen SONIC WBC adapter over a staged terrain grid.

One factory per source, side by side, sharing `_core`. Not one factory with a
`source=` switch: the sources differ in the things a switch hides (grid axes,
tile size, episode length), and a branch per difference is how a factory turns
into a maze.

  omni_env_cfg()    OmniRetarget robot-terrain — climb, z_scale = difficulty
  grail_env_cfg()   GRAIL curb — no difficulty axis, one row

What they SHARE is the point: robot, action, sensors, obs, rewards,
terminations, agent. A policy trained on one loads into the other unchanged,
because the height scan is 187 rays either way.

Memory: mjlab replicates the model `nworld = num_envs`, so robots never share a
world and terrain contact is always 1 robot x terrain. Every world does carry
the whole grid, but statics are culled in broadphase — cost is model size once,
not contacts per step. Boxes are what keep that true.
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
from orcs.tasks.perloco.roster import Roster, load_roster
from orcs.tasks.perloco.sensors import (
    PERLOCO_KILL_BODIES,
    TERRAIN_CONTACT_SENSOR_NAME,
    terrain_contact_sensor,
    terrain_scan_sensor,
)
from orcs.tasks.perloco.terrain import TILE_SIZE, terrain_generator_cfg

GRAIL_TILE_SIZE = (12.0, 12.0)
"""GRAIL curbs are not centred on their own origin — they run x: 0 -> 5.6 m,
so placing one at the tile centre needs 2x that to stay off the neighbour
(measured: box half-footprint 5.61 m, reference pelvis travel 4.25 m).
Recentring geometry AND motion by the same offset at staging would let this
drop to ~7 m; it is not done because shifting staged data is a change that
has to be re-verified, and static geoms are not what costs walltime."""

__all__ = ["omni_env_cfg", "grail_env_cfg", "staged_root"]

_MOTION_PAD_EPS_SEC = 2.0
"""Post-motion hold padding — the episode, not the motion, owns resets."""

_P = {"command_name": "motion"}


def staged_root(source: str):
    return DATA_ROOT / "terrain_motions" / source


@lru_cache(maxsize=None)
def _resolve(source: str, roster_path: str | None) -> tuple[Roster, str, int]:
    """(roster, first clip, longest clip in frames), cached per roster."""
    root = staged_root(source)
    roster = load_roster(source, root, roster_path)
    files = [f for k in roster.tile_keys
             for f in sorted((root / k).glob("sample*/motion.npz"))
             if not roster.clips.get(k) or f.parent.name in roster.clips[k]]
    if not files:
        raise FileNotFoundError(f"roster selected no clips under {root}")
    max_len = max(int(np.load(f)["joint_pos"].shape[0]) for f in files)
    return roster, str(files[0]), max_len


def _core(
    source: str,
    roster: Roster,
    motion_file: str,
    max_clip_len: int,
    *,
    agent: str,
    play: bool,
    scan_frame: str,
    tile_size: tuple[float, float],
    num_steps_per_env: int,
    robot_cfg: Callable[[], EntityCfg] | None,
    kill_bodies: tuple[str, ...],
    kill_exclude: tuple[str, ...],
) -> ManagerBasedRlEnvCfg:
    """Everything both sources agree on."""
    assert agent in ("sonic", "tara"), f"unknown agent {agent!r}"
    root = staged_root(source)

    cfg = ManagerBasedRlEnvCfg(
        scene=SceneCfg(
            terrain=TerrainEntityCfg(
                terrain_type="generator",
                terrain_generator=terrain_generator_cfg(
                    root, roster, size=tile_size),
            ),
            num_envs=1,
        ),
        observations={},
        actions={},
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
        rewards={},
        terminations={
            "time_out": TerminationTermCfg(func=mdp.time_out, time_out=True),
            "illegal_contact": TerminationTermCfg(
                func=illegal_contact,
                params={"sensor_name": TERRAIN_CONTACT_SENSOR_NAME}),
        },
        viewer=ViewerConfig(
            origin_type=ViewerConfig.OriginType.WORLD,
            distance=6.0, elevation=-20.0, azimuth=135.0,
        ),
        # Terrain is boxes, so contacts are box-box: few, cheap, and bounded by
        # the robot's geom count rather than the grid's.
        sim=SimulationCfg(
            nconmax=100, njmax=300,
            mujoco=MujocoCfg(timestep=0.005, iterations=10, ls_iterations=20),
        ),
        decimation=4,
        episode_length_s=10.0,  # set below from the dataset
    )

    robot = profile.robot_cfg(base=(robot_cfg or get_g1_flat_hand_cfg)())
    cfg.scene.entities["robot"] = robot
    cfg.actions["joint_pos"] = profile.action_cfg(robot)
    cfg.scene.sensors = (
        terrain_contact_sensor(kill_bodies, kill_exclude),
        terrain_scan_sensor(scan_frame, debug_vis=play),
    )

    cfg.commands["motion"] = TerrainMotionCommandCfg(
        motion_file=motion_file,
        dataset_dir=str(root),
        tile_keys=roster.tile_keys,
        n_rows=roster.n_rows,
        clips=roster.clips,
        future_steps=5,
        resampling_time_range=(1e9, 1e9),
        debug_vis=True,
        pose_range={},
        velocity_range={},
        joint_position_range=(0.0, 0.0),
    )

    step_dt = cfg.sim.mujoco.timestep * cfg.decimation
    cfg.episode_length_s = max_clip_len * step_dt + _MOTION_PAD_EPS_SEC
    cfg.terminations.update({
        # Anchor tubes are ON here (uolm drops them): with no object dragging
        # the root off the reference, a robot far from its anchor is simply
        # failing to track. Loose enough for climbing's vertical excursions.
        "bad_anchor_pos": TerminationTermCfg(
            func=mdp.bad_anchor_pos, params={**_P, "threshold": 0.4}),
        "bad_anchor_ori": TerminationTermCfg(
            func=mdp.bad_anchor_ori, params={**_P, "threshold": 0.8}),
        "exceeded_motion": TerminationTermCfg(
            func=mdp.exceeded_motion_by_eps, time_out=True,
            params={**_P, "epsilon_steps": int(_MOTION_PAD_EPS_SEC / step_dt)}),
    })

    # Pure motion tracking — no object, so no object terms.
    cfg.rewards = {
        "root_pos": RewardTermCfg(
            func=tracking_rewards.motion_global_anchor_position_error_exp,
            weight=0.5, params={**_P, "std": 0.3}),
        "root_ori": RewardTermCfg(
            func=tracking_rewards.motion_global_anchor_orientation_error_exp,
            weight=0.5, params={**_P, "std": 0.4}),
        "body_pos": RewardTermCfg(
            func=tracking_rewards.motion_relative_body_position_error_exp,
            weight=1.0, params={**_P, "std": 0.3}),
        "body_ori": RewardTermCfg(
            func=tracking_rewards.motion_relative_body_orientation_error_exp,
            weight=1.0, params={**_P, "std": 0.4}),
        "body_lin_vel": RewardTermCfg(
            func=tracking_rewards.motion_global_body_linear_velocity_error_exp,
            weight=1.0, params={**_P, "std": 1.0}),
        "body_ang_vel": RewardTermCfg(
            func=tracking_rewards.motion_global_body_angular_velocity_error_exp,
            weight=1.0, params={**_P, "std": 3.14}),
        "action_rate_l2": RewardTermCfg(func=mdp.action_rate_l2, weight=-0.1),
        "joint_pos_limits": RewardTermCfg(func=mdp.joint_pos_limits, weight=-1.0),
    }

    ctx = ObsCtx(p=_P)
    cfg.observations = tara_obs(ctx) if agent == "tara" else sonic_obs(ctx)

    if play:
        _play_overrides(cfg)
    return cfg


def omni_env_cfg(
    *,
    agent: str = "sonic",
    play: bool = False,
    roster: str | None = None,
    scan_frame: str = "pelvis",
    tile_size: tuple[float, float] = TILE_SIZE,
    num_steps_per_env: int = 24,
    robot_cfg: Callable[[], EntityCfg] | None = None,
    kill_bodies: tuple[str, ...] = PERLOCO_KILL_BODIES,
    kill_exclude: tuple[str, ...] = (),
) -> ManagerBasedRlEnvCfg:
    """OmniRetarget robot-terrain: climb families x z_scale levels.

    `roster` swaps `rosters/omni.toml` for another file — the only supported way
    to change which tiles a run sees, and how an eval isolates a subset on
    byte-identical infrastructure.
    """
    r, motion_file, max_len = _resolve("omni", roster)
    return _core(
        "omni", r, motion_file, max_len,
        agent=agent, play=play, scan_frame=scan_frame, tile_size=tile_size,
        num_steps_per_env=num_steps_per_env, robot_cfg=robot_cfg,
        kill_bodies=kill_bodies, kill_exclude=kill_exclude,
    )


def grail_env_cfg(
    *,
    agent: str = "sonic",
    play: bool = False,
    roster: str | None = None,
    scan_frame: str = "pelvis",
    tile_size: tuple[float, float] = GRAIL_TILE_SIZE,
    num_steps_per_env: int = 24,
    robot_cfg: Callable[[], EntityCfg] | None = None,
    kill_bodies: tuple[str, ...] = PERLOCO_KILL_BODIES,
    kill_exclude: tuple[str, ...] = (),
) -> ManagerBasedRlEnvCfg:
    """GRAIL curb: one terrain per column, ONE row — there is no difficulty axis.

    Staged under a single `level_0.00` rather than a faked difficulty: a row
    axis that does not mean height is a curriculum that promotes nothing.
    """
    r, motion_file, max_len = _resolve("grail", roster)
    return _core(
        "grail", r, motion_file, max_len,
        agent=agent, play=play, scan_frame=scan_frame, tile_size=tile_size,
        num_steps_per_env=num_steps_per_env, robot_cfg=robot_cfg,
        kill_bodies=kill_bodies, kill_exclude=kill_exclude,
    )


def _play_overrides(cfg: ManagerBasedRlEnvCfg) -> None:
    """No corruption, no anneal, no tracking kills — so a rollout survives long
    enough to SHOW where it fails. `illegal_contact` stays: a pelvis on the
    ground is exactly what you want to notice."""
    cfg.observations["policy"].enable_corruption = False
    cfg.events.pop("policy_update_counter", None)
    for k in ("bad_anchor_pos", "bad_anchor_ori"):
        cfg.terminations.pop(k, None)
    cfg.commands["motion"].start_from_zero = True
