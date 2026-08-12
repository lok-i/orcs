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

from dataclasses import fields
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
from orcs.core.data.seeds import SeedMotion
from orcs.core.obs import apply_obs_noise
from orcs.core.paths import DATA_ROOT
from orcs.core.robustness import apply_robot_robustness, strip_domain
from orcs.tasks.perloco import mdp
from orcs.tasks.perloco.mdp.commands import (
    SmplSeedTerrainMotionCommandCfg,
    TerrainMotionCommandCfg,
)
from orcs.tasks.perloco.observation_cfgs import ObsCtx, sonic_obs, tara_obs
from orcs.tasks.perloco.roster import Roster, load_roster
from orcs.tasks.perloco.sensors import (
    GROUND_CONTACT_SENSOR_NAME,
    PERLOCO_KILL_BODIES,
    ground_contact_sensor,
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

__all__ = [
    "OMNI_RENDER_Z_SCALE",
    "grail_env_cfg",
    "grail_smpl_env_cfg",
    "omni_env_cfg",
    "staged_root",
]

_MOTION_PAD_EPS_SEC = 2.0
"""Post-motion hold padding — the episode, not the motion, owns resets."""

_P = {"command_name": "motion"}

# The robustness domain is UNCONDITIONAL (the `SIM2REAL` switch is gone,
# 2026-08-06): the robot half of `orcs.core.robustness` plus `apply_obs_noise`,
# the same set repose transfers to hardware with. Run 1 asked a behavior
# question and could afford a nominal domain; a policy anyone intends to deploy
# cannot, and a switch that is always on in every task is not an axis.
OMNI_RENDER_Z_SCALE: float | None = 0.75
"""OmniRetarget geometry height, decoupled from the clip's own z_scale.

`None` = staged pairing (each row's obstacle matches the motion retargeted
against it). A float renders EVERY row at that absolute z_scale while the
motions stay per-row — so the grid becomes N motions over ONE obstacle, and the
policy must generalize across the gap instead of memorizing a height.

**0.75 is measured, not chosen.** A checkpoint trained on levels 1.0/1.1/1.2
still climbs at 0.75 (and 0.8) without ever having seen it, so the gap is
inside the frozen base's competence rather than past it. It survives training,
not just play, because the reference floats only `(level - 0.75) * 0.475` =
0.119 / 0.166 / 0.214 m above the box — all well inside `anchor_pos_thresh`
0.4, so `bad_anchor_pos` sees a tracking error and not a termination. Drop this
much below ~0.5 and that stops being true: the float reaches the tube and the
episode dies on the reference rather than on the robot.

The source ships URDFs for z_scale 0.8-1.2 only, which is why this is a render
knob and not a roster row — there is no 0.75 tile to stage.
"""


def staged_root(source: str):
    return DATA_ROOT / "terrain_motions" / source


@lru_cache(maxsize=None)
def _resolve(
    source: str, roster_path: str | None, need_smpl: bool = False
) -> tuple[Roster, str, int]:
    """(roster, first clip, longest clip in frames), cached per roster."""
    root = staged_root(source)
    roster = load_roster(source, root, roster_path)
    files = [f for k in roster.tile_keys
             for f in sorted((root / k).glob("sample*/motion.npz"))
             if not roster.clips.get(k) or f.parent.name in roster.clips[k]]
    if not files:
        raise FileNotFoundError(f"roster selected no clips under {root}")
    # Checked HERE, not in the loader: an unstaged SMPL half loads as zeros,
    # which the encoder consumes without complaint and the reward never sees.
    if need_smpl and (bad := [f for f in files
                              if not (f.parent / "smpl_motion.npz").exists()]):
        raise FileNotFoundError(
            f"{len(bad)}/{len(files)} selected clips have no smpl_motion.npz "
            f"(e.g. {bad[0].parent}) — restage with "
            f"`stage_terrain_motions.py --source {source} --smpl`")
    max_len = max(int(np.load(f)["joint_pos"].shape[0]) for f in files)
    return roster, str(files[0]), max_len


@lru_cache(maxsize=None)
def _resolve_grail_smpl_seeds(roster_path: str | None) -> tuple[Roster, str, int]:
    """The rectangular subset with complete SMPL + kinematic retarget data.

    Phase-1 generation is intentionally resumable, so a partial seed dataset
    is a valid development state.  A family enters this task only when every
    selected level has at least one complete clip.  GRAIL has one level today;
    spelling the rectangular rule keeps terrain columns and clip masks honest.
    """
    root = staged_root("grail")
    requested = load_roster("grail", root, roster_path)
    ready_by_key: dict[str, tuple[str, ...]] = {}
    lengths: dict[tuple[str, str], int] = {}

    for key in requested.tile_keys:
        keep = requested.clips.get(key)
        ready: list[str] = []
        for motion_file in sorted((root / key).glob("sample*/motion.npz")):
            sample = motion_file.parent
            if keep and sample.name not in keep:
                continue
            if not (sample / "smpl_motion.npz").exists() \
                    or not (sample / "seed_state.npz").exists():
                continue
            seed = SeedMotion.load(sample / "seed_state.npz")
            if not seed.valid.all():
                continue
            ready.append(sample.name)
            lengths[(key, sample.name)] = seed.num_frames
        if ready:
            ready_by_key[key] = tuple(ready)

    families = tuple(
        family
        for family in requested.families
        if all(f"{family}/level_{level:.2f}" in ready_by_key
               for level in requested.levels)
    )
    if not families:
        raise FileNotFoundError(
            f"no complete GRAIL SMPL seed clips under {root}; run "
            "orcs-pseudo-retarget --scene perloco-grail --source <sample-dir>"
        )
    clips = {
        key: ready_by_key[key]
        for family in families
        for level in requested.levels
        if (key := f"{family}/level_{level:.2f}") in ready_by_key
    }
    selected = [(key, sample) for key, samples in clips.items() for sample in samples]
    first_key, first_sample = selected[0]
    first_motion = str(root / first_key / first_sample / "motion.npz")
    max_len = max(lengths[item] for item in selected)
    return Roster(families, requested.levels, clips), first_motion, max_len


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
    anchor_pos_thresh: float,
    anchor_ori_thresh: float,
    command_space: str = "robot",
    render_z_scale: float | None = None,
) -> ManagerBasedRlEnvCfg:
    """Everything both sources agree on."""
    assert agent in ("sonic", "tara"), f"unknown agent {agent!r}"
    assert command_space in ("robot", "smpl"), f"unknown space {command_space!r}"
    assert not (agent == "tara" and command_space == "smpl"), (
        "the smpl command space IS the SONIC smpl encoder — no tabula-rasa variant")
    root = staged_root(source)

    cfg = ManagerBasedRlEnvCfg(
        scene=SceneCfg(
            terrain=TerrainEntityCfg(
                terrain_type="generator",
                terrain_generator=terrain_generator_cfg(
                    root, roster, size=tile_size,
                    render_z_scale=render_z_scale),
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
                params={"sensor_name": GROUND_CONTACT_SENSOR_NAME}),
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
        ground_contact_sensor(kill_bodies, kill_exclude),
        terrain_scan_sensor(scan_frame, debug_vis=play),
    )

    cfg.commands["motion"] = TerrainMotionCommandCfg(
        motion_file=motion_file,
        dataset_dir=str(root),
        tile_keys=roster.tile_keys,
        n_rows=roster.n_rows,
        clips=roster.clips,
        command_space=command_space,
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
        # failing to track. Width is PER SOURCE — see the two factories.
        "bad_anchor_pos": TerminationTermCfg(
            func=mdp.bad_anchor_pos, params={**_P, "threshold": anchor_pos_thresh}),
        "bad_anchor_ori": TerminationTermCfg(
            func=mdp.bad_anchor_ori, params={**_P, "threshold": anchor_ori_thresh}),
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
    # command_space "robot"/"smpl" -> mocke's encoder mode "g1"/"smpl"
    cfg.observations = (
        tara_obs(ctx) if agent == "tara"
        else sonic_obs(ctx, "smpl" if command_space == "smpl" else "g1"))

    # ── the training domain: robot half only (there is no object here) ──
    apply_robot_robustness(cfg)
    apply_obs_noise(cfg)

    # INVARIANT: play overrides are LAST — they SUBTRACT from the assembled
    # domain, so anything wired below this line leaks into play/eval.
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
    anchor_pos_thresh: float = 0.4,
    anchor_ori_thresh: float = 0.8,
    render_z_scale: float | None = OMNI_RENDER_Z_SCALE,
) -> ManagerBasedRlEnvCfg:
    """OmniRetarget robot-terrain: climb families x z_scale levels.

    `roster` swaps `rosters/omni.toml` for another file — the only supported way
    to change which tiles a run sees, and how an eval isolates a subset on
    byte-identical infrastructure.

    `render_z_scale` pins the OBSTACLE height while the roster keeps choosing
    the MOTION — see `OMNI_RENDER_Z_SCALE` for why it defaults to 0.75 and what
    bounds it. Pass `None` for the staged pairing.

    Wide anchor tubes, unlike GRAIL's: climbing MEANS large pelvis-z excursions,
    and the frozen base spends a third of its steps past 0.2 m of |dz| (p90
    0.562 m, vs GRAIL's 0.281) doing the task correctly. Tightening here would
    terminate the behaviour instead of the failure.
    """
    r, motion_file, max_len = _resolve("omni", roster)
    return _core(
        "omni", r, motion_file, max_len,
        agent=agent, play=play, scan_frame=scan_frame, tile_size=tile_size,
        num_steps_per_env=num_steps_per_env, robot_cfg=robot_cfg,
        kill_bodies=kill_bodies, kill_exclude=kill_exclude,
        anchor_pos_thresh=anchor_pos_thresh, anchor_ori_thresh=anchor_ori_thresh,
        render_z_scale=render_z_scale,
    )


def grail_env_cfg(
    *,
    agent: str = "sonic",
    command_space: str = "robot",
    play: bool = False,
    roster: str | None = None,
    scan_frame: str = "pelvis",
    tile_size: tuple[float, float] = GRAIL_TILE_SIZE,
    num_steps_per_env: int = 24,
    robot_cfg: Callable[[], EntityCfg] | None = None,
    kill_bodies: tuple[str, ...] = PERLOCO_KILL_BODIES,
    kill_exclude: tuple[str, ...] = (),
    anchor_pos_thresh: float = 0.2,
    anchor_ori_thresh: float = 0.3,
) -> ManagerBasedRlEnvCfg:
    """GRAIL curb: one terrain per column, ONE row — there is no difficulty axis.

    Staged under a single `level_0.00` rather than a faked difficulty: a row
    axis that does not mean height is a curriculum that promotes nothing.

    `command_space="smpl"` is retained as the phase-1 rollout mode: the frozen
    encoder reads the human while the existing GRAIL robot motion seeds the
    assisted rollout.  The registered `-Smpl` training task does not use this
    path; :func:`grail_smpl_env_cfg` supplies point rewards and seed-only RSI.

    Tighter anchor tubes than OmniRetarget's (0.2 m / 0.3 rad vs 0.4 / 0.8).
    Curb-walking has no legitimate vertical excursion, and the frozen base
    tracks it to |dz| p50 0.018 m / ori p50 0.101 rad — so 0.2/0.3 sits at
    roughly its p90 and kills divergence, not the behaviour. (`bad_anchor_pos`
    is pelvis-z drift, not a 3-D tube.)
    """
    r, motion_file, max_len = _resolve("grail", roster, command_space == "smpl")
    return _core(
        "grail", r, motion_file, max_len,
        agent=agent, play=play, scan_frame=scan_frame, tile_size=tile_size,
        num_steps_per_env=num_steps_per_env, robot_cfg=robot_cfg,
        kill_bodies=kill_bodies, kill_exclude=kill_exclude,
        anchor_pos_thresh=anchor_pos_thresh, anchor_ori_thresh=anchor_ori_thresh,
        command_space=command_space,
    )


def _seed_command_cfg(
    cfg: TerrainMotionCommandCfg,
) -> SmplSeedTerrainMotionCommandCfg:
    """Change command implementation without re-declaring its many base fields."""
    values = {
        field.name: getattr(cfg, field.name)
        for field in fields(SmplSeedTerrainMotionCommandCfg)
        if field.init and hasattr(cfg, field.name)
    }
    return SmplSeedTerrainMotionCommandCfg(**values)


def grail_smpl_env_cfg(
    *,
    play: bool = False,
    roster: str | None = None,
    scan_frame: str = "pelvis",
    tile_size: tuple[float, float] = GRAIL_TILE_SIZE,
    num_steps_per_env: int = 24,
    robot_cfg: Callable[[], EntityCfg] | None = None,
    kill_bodies: tuple[str, ...] = PERLOCO_KILL_BODIES,
    kill_exclude: tuple[str, ...] = (),
    point_anchor_threshold: float = 0.75,
) -> ManagerBasedRlEnvCfg:
    """GRAIL SMPL point tracking with one-to-one seed-backed RSI.

    Only the paired SMPL source defines rewards, commands, and divergence.
    ``seed_state.npz`` supplies the reset state and initial action history; the
    upstream GRAIL robot retarget is not a teacher trajectory.
    """
    r, motion_file, max_len = _resolve_grail_smpl_seeds(roster)
    cfg = _core(
        "grail",
        r,
        motion_file,
        max_len,
        agent="sonic",
        play=play,
        scan_frame=scan_frame,
        tile_size=tile_size,
        num_steps_per_env=num_steps_per_env,
        robot_cfg=robot_cfg,
        kill_bodies=kill_bodies,
        kill_exclude=kill_exclude,
        anchor_pos_thresh=0.2,
        anchor_ori_thresh=0.3,
        command_space="smpl",
    )
    cfg.commands["motion"] = _seed_command_cfg(cfg.commands["motion"])
    cfg.rewards = {
        "point_pos": RewardTermCfg(
            func=mdp.point_position_error_exp,
            weight=4.0,
            params={**_P, "std": 0.3},
        ),
        "point_vel": RewardTermCfg(
            func=mdp.point_velocity_error_exp,
            weight=1.0,
            params={**_P, "std": 1.0},
        ),
        "action_rate_l2": RewardTermCfg(func=mdp.action_rate_l2, weight=-0.1),
        "joint_pos_limits": RewardTermCfg(func=mdp.joint_pos_limits, weight=-1.0),
    }
    cfg.terminations.pop("bad_anchor_pos", None)
    cfg.terminations.pop("bad_anchor_ori", None)
    if not play:
        cfg.terminations["bad_point_anchor_pos"] = TerminationTermCfg(
            func=mdp.bad_point_anchor_pos,
            params={**_P, "threshold": point_anchor_threshold},
        )
    return cfg


def _play_overrides(cfg: ManagerBasedRlEnvCfg) -> None:
    """No domain, no anneal, no tracking kills — so a rollout survives long
    enough to SHOW where it fails. `illegal_contact` stays: a pelvis on the
    ground is exactly what you want to notice."""
    strip_domain(cfg)
    cfg.events.pop("policy_update_counter", None)
    for k in ("bad_anchor_pos", "bad_anchor_ori"):
        cfg.terminations.pop(k, None)
    # cfg.commands["motion"].start_from_zero = True
