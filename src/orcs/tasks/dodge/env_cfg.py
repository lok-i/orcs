"""Dodge env — frozen SONIC WBC adapter over a NOMINAL STAND reference.

A ball is thrown at a standing humanoid, which must avoid contact on every body
link while staying upright. One factory:

    dodge_env_cfg(play=False, ...)

**What makes this task different from uolm and perloco: the reference is
constant.** Both of those track a demo clip, so the adapter's job is to correct
a reference that already describes the motion. Here the reference says "stand
there" for every frame of every episode, and the evasion exists ONLY as the
adapter's departure from it. The reference pose is the robot entity's own
`init_state`, which is also the offset the action term works from — so a zero
adapter action reproduces the reference exactly, and the whole dodge is
measurable as action magnitude (`orcs.cli.make_nominal_motion`).

**Reward stack: one task term plus a station hold.** Posture, balance, foot
placement, naturalness and smoothness are what the frozen base is for; spelling
them as rewards would be a second, weaker copy of a prior that is already
structural. That is the claim, and the term count is how it gets tested.

Terminations are a hit and a fall, both from iteration zero and both without a
penalty — the only cost of being hit is the reward foregone by an early episode
end. Deferring the hit termination behind a curriculum is a known failure mode
(it converges to a never-dodge stander, then collapses when the switch flips),
so it is not offered as an option here.
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

from orcs.assets import BALL_BODY_NAME, ball_entity_cfg, get_g1_flat_hand_cfg
from orcs.core.data.scan import scan_flat
from orcs.core.mdp.commands import MultiClipMotionCommandCfg
from orcs.core.paths import DATA_ROOT
from orcs.tasks.dodge import mdp
from orcs.tasks.dodge.observation_cfgs import ObsCtx, sonic_obs
from orcs.tasks.dodge.sensors import (
    BALL_CONTACT_SENSOR_NAME,
    DODGE_KILL_BODIES,
    GROUND_CONTACT_SENSOR_NAME,
    ball_contact_sensor,
    ground_contact_sensor,
)

__all__ = ["dodge_env_cfg", "nominal_root", "SIM2REAL"]

_P = {"command_name": "motion"}

SIM2REAL = False
"""THE robustness switch, and the one place it lives.

Off for run 1, matching perloco. The question is behavioural — can a frozen WBC
be adapted into a whole-body evasion at all — and a domain that makes it harder
answers a different one. Flipping this on turns on proprio noise; a push / mass
/ friction domain does not exist for dodge yet and belongs behind this same
switch when it lands.
"""

EPISODE_LENGTH_S = 10.0
"""Long enough for several throws at `interval_range_s`, short enough that a
policy which learns to survive is rewarded for it repeatedly."""


def nominal_root() -> str:
    return str(DATA_ROOT / "nominal_motions")


@lru_cache(maxsize=None)
def _resolve() -> tuple[str, int]:
    """(first clip, its length in frames) — cached; raises if not generated.

    The raise becomes a `SKIP_REASON` entry, and that entry is the only place a
    user learns why the task vanished — `list_tasks()` just omits it. So the
    message has to carry the fix, not merely the path: unlike perloco's staged
    terrain, this reference is ONE orcs command away, and "no motion.npz under
    /some/path" does not tell anyone that command exists.
    """
    try:
        files = scan_flat(nominal_root())
    except FileNotFoundError as e:
        raise FileNotFoundError(
            f"{e} — generate it with `orcs-make-nominal`. If that ran and wrote "
            f"somewhere else, orcs resolved a different DATA_ROOT than it does "
            f"here; check `python -c 'from orcs.core.paths import DATA_ROOT; "
            f"print(DATA_ROOT)'` and ORCS_DATA_ROOT."
        ) from e
    max_len = max(int(np.load(f)["joint_pos"].shape[0]) for f in files)
    return files[0], max_len


def dodge_env_cfg(
    *,
    play: bool = False,
    num_steps_per_env: int = 24,
    robot_cfg: Callable[[], EntityCfg] | None = None,
    ball_radius: float = 0.12,
    ball_mass: float = 0.62,
    kill_bodies: tuple[str, ...] = DODGE_KILL_BODIES,
    kill_exclude: tuple[str, ...] = (),
    station_std: float = 1.0,
    **throw_kw,
) -> ManagerBasedRlEnvCfg:
    """Frozen SONIC + LoRA, standing, dodging a thrown ball.

    `**throw_kw` forwards to :class:`~orcs.tasks.dodge.mdp.events.ThrowBall`, so
    the whole threat model (interval, cone, distance, flight time, threat mix,
    anchor fraction) is tunable without re-declaring a single default here.

    `station_std` widens or tightens the pull back to the reference position.
    Wide by default: at 1.0 m a 0.5 m sidestep keeps ~78% of the term, so it
    corrects cumulative drift without charging for the evasion itself.
    """
    motion_file, max_clip_len = _resolve()

    cfg = ManagerBasedRlEnvCfg(
        scene=SceneCfg(
            terrain=TerrainEntityCfg(terrain_type="plane"),
            entities={
                BALL_BODY_NAME: ball_entity_cfg(
                    radius=ball_radius, mass=ball_mass),
            },
            num_envs=1,
            env_spacing=2.5,
        ),
        observations={},  # set by sonic_obs below
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
            "throw_ball": EventTermCfg(
                func=mdp.ThrowBall,
                mode="interval",
                interval_range_s=(0.0, 0.0),  # every env step
                params={"ball_name": BALL_BODY_NAME, **throw_kw},
            ),
        },
        rewards={},       # filled below
        terminations={},  # filled below
        viewer=ViewerConfig(
            origin_type=ViewerConfig.OriginType.WORLD,
            distance=4.0, elevation=-15.0, azimuth=135.0,
        ),
        # One robot, one sphere, a plane: the cheapest contact set orcs has.
        sim=SimulationCfg(
            nconmax=100, njmax=300,
            mujoco=MujocoCfg(timestep=0.005, iterations=10, ls_iterations=20),
        ),
        decimation=4,
        episode_length_s=EPISODE_LENGTH_S,
    )

    # ── SONIC robot + action (flat-hand G1, hip_pitch regroup, MJ-order) ──
    robot = profile.robot_cfg(base=(robot_cfg or get_g1_flat_hand_cfg)())
    cfg.scene.entities["robot"] = robot
    cfg.actions["joint_pos"] = profile.action_cfg(robot)
    cfg.scene.sensors = (
        ground_contact_sensor(kill_bodies, kill_exclude),
        ball_contact_sensor(),
    )

    # ── the nominal-stand reference ──
    # `start_from_zero`: every frame of the clip is the same pose, so phase
    # carries no information and a random init would only risk running an env
    # off the end of the timeline mid-episode. Pose/velocity/joint ranges stay
    # empty — RSI onto ONE pose is the point (the robot always starts standing);
    # the variation this task needs is in the THROW, not the initial condition.
    cfg.commands["motion"] = MultiClipMotionCommandCfg(
        motion_file=motion_file,
        dataset_dir=nominal_root(),
        future_steps=5,
        resampling_time_range=(1e9, 1e9),
        start_from_zero=True,
        debug_vis=True,
        pose_range={},
        velocity_range={},
        joint_position_range=(0.0, 0.0),
    )
    step_dt = cfg.sim.mujoco.timestep * cfg.decimation
    assert max_clip_len * step_dt >= cfg.episode_length_s, (
        f"nominal clip is {max_clip_len * step_dt:.1f} s but the episode is "
        f"{cfg.episode_length_s:.1f} s — regenerate it with "
        f"`orcs-make-nominal --seconds {cfg.episode_length_s + 5:.0f}`, or an "
        "env will run off the end of its reference mid-episode")

    # ── terminations: a hit and a fall. No anchor tubes — the whole task is a
    # deliberate departure from the reference, so a tube would terminate the
    # behaviour instead of the failure. Neither carries a reward penalty; the
    # cost of either is the reward foregone by ending early.
    cfg.terminations = {
        "time_out": TerminationTermCfg(func=mdp.time_out, time_out=True),
        "ball_hit": TerminationTermCfg(
            func=illegal_contact,
            params={"sensor_name": BALL_CONTACT_SENSOR_NAME}),
        "fell": TerminationTermCfg(
            func=illegal_contact,
            params={"sensor_name": GROUND_CONTACT_SENSOR_NAME}),
    }

    # ── rewards ──
    cfg.rewards = {
        # THE task: far from the ball, still when it is not there.
        "ball_clearance": RewardTermCfg(
            func=mdp.ball_clearance, weight=1.0,
            params={"ball_name": BALL_BODY_NAME}),
        # Station hold. `ball_clearance` penalizes SPEED, not position, so a
        # policy that sidesteps every throw and never comes back accumulates
        # drift unpenalized. This is the return-to-reference pull, and it is
        # spelled as tracking the nominal anchor because that is what it is.
        "station": RewardTermCfg(
            func=tracking_rewards.motion_global_anchor_position_error_exp,
            weight=0.3, params={**_P, "std": station_std}),
        # Physical sanity, orcs's standard pair — not posture, not style.
        "action_rate_l2": RewardTermCfg(func=mdp.action_rate_l2, weight=-0.1),
        "joint_pos_limits": RewardTermCfg(func=mdp.joint_pos_limits, weight=-1.0),
    }

    cfg.observations = sonic_obs(ObsCtx(p=_P, noisy=SIM2REAL))

    if play:
        _play_overrides(cfg)
    return cfg


def _play_overrides(cfg: ManagerBasedRlEnvCfg) -> None:
    """Eval build: no domain, fewer envs, and every env gets thrown at.

    `stand_fraction=0` because the ball-free anchors exist to keep a standing
    skill in the TRAINING batch; scoring them would dilute the dodge rate with
    episodes that were never at risk.
    """
    cfg.scene.num_envs = 16
    cfg.events["throw_ball"].params["stand_fraction"] = 0.0
    for group in cfg.observations.values():
        group.enable_corruption = False
