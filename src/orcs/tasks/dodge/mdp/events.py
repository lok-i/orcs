"""Event terms — the projectile threat model.

RSI is the motion command's (nominal stand); generic resets come from mjlab
stock mdp. What is here is the one thing dodge owns: when a ball is thrown, from
where, and how fast.
"""

from __future__ import annotations

import math

import torch
from mjlab.envs import ManagerBasedRlEnv
from mjlab.managers.manager_base import ManagerTermBase
from mjlab.utils.lab_api.math import quat_apply, yaw_quat

from orcs.core.mdp.events import PolicyUpdateCounter  # noqa: F401 — re-export

__all__ = ["PolicyUpdateCounter", "ThrowBall"]


class ThrowBall(ManagerTermBase):
    """Throw a ball at the robot on a per-env timer; park it the rest of the time.

    ``mode="interval"``, ``interval_range_s=(0.0, 0.0)`` -> ticks every env step.

    **Trajectory.** Pure ballistic under the model's own gravity, parameterised
    by the REACTION WINDOW rather than by a launch speed: the ball is released in
    the robot's frontal cone at ``dist_range`` and must arrive at the aim point
    after ``flight_time_range``, which fixes the horizontal speed. Two threat
    types are mixed so the evasion vocabulary is not one move:

    | type | release | vz0 | arrives at | evasion |
    |---|---|---|---|---|
    | descending | ``launch_height_range`` (~2 m) | 0 | wherever it has fallen to | sidestep / step over |
    | low arc | ``low_launch_height_range`` (~waist) | >0 | ``low_target_z_range`` (torso/head) | duck / lean |

    The aim point is the robot's xy, LED by its own velocity and then jittered,
    so a robot that walks in a straight line does not evade for free and no two
    throws share a geometry. Descending flight times are capped so the ball
    cannot land short of the robot.

    **Parking is a pin, not a spawn.** A ball is only ever teleported: for
    ``flight_window_s`` after a throw it flies free, and every other step its
    pose and velocity are written to the park point overhead. That makes a
    between-throws ball unconditionally inert — no rolling ball on the ground in
    front of the camera, which would be a second, static, non-looming threat the
    policy could learn to dodge. Pinning costs one indexed write per step.

    ``stand_fraction`` of envs are ANCHORS: never thrown at, so the batch always
    carries the ball-free standing behaviour and episodes do not all end in a
    hit. Re-rolled per reset, not fixed per env, so no env is permanently easy.
    """

    def __init__(self, cfg, env: ManagerBasedRlEnv) -> None:
        super().__init__(env)
        n, dev = env.num_envs, env.device
        self._countdown = torch.zeros(n, dtype=torch.long, device=dev)
        self._flight = torch.zeros(n, dtype=torch.long, device=dev)
        self._anchor = torch.zeros(n, dtype=torch.bool, device=dev)
        self._g = -float(env.sim.mj_model.opt.gravity[2])
        p = cfg.params
        self._interval_range = p.get("interval_range_s", (1.0, 4.0))
        self._stand_fraction = float(p.get("stand_fraction", 0.2))
        self.reset(None)

    # ── state ──

    def _sample_countdown(self, n: int) -> torch.Tensor:
        lo, hi = self._interval_range
        secs = torch.rand(n, device=self.device) * (hi - lo) + lo
        return (secs / self._env.step_dt).long().clamp(min=1)

    def reset(self, env_ids=None) -> None:
        if env_ids is None:
            env_ids = torch.arange(self.num_envs, device=self.device)
        n = len(env_ids)
        self._countdown[env_ids] = self._sample_countdown(n)
        self._flight[env_ids] = 0
        self._anchor[env_ids] = (
            torch.rand(n, device=self.device) < self._stand_fraction)

    # ── per step ──

    def __call__(
        self,
        env: ManagerBasedRlEnv,
        env_ids,
        ball_name: str = "ball",
        robot_name: str = "robot",
        interval_range_s: tuple[float, float] = (1.0, 4.0),
        stand_fraction: float = 0.2,
        flight_window_s: float = 1.0,
        park_z: float = 8.0,
        dist_range: tuple[float, float] = (2.0, 3.0),
        angle_deg: float = 25.0,
        flight_time_range: tuple[float, float] = (0.55, 0.62),
        launch_height_range: tuple[float, float] = (1.8, 2.0),
        low_arc_fraction: float = 0.5,
        low_launch_height_range: tuple[float, float] = (0.4, 0.9),
        low_target_z_range: tuple[float, float] = (0.9, 1.3),
        lead_target: bool = True,
        aim_noise: float = 0.1,
    ) -> None:
        del env_ids, interval_range_s, stand_fraction  # read at construction
        ball, robot = env.scene[ball_name], env.scene[robot_name]
        dev = self.device

        # 1. Pin every ball that is not mid-flight. Unconditional, so a ball that
        #    landed, bounced or was never thrown is in exactly one place.
        self._flight = (self._flight - 1).clamp(min=0)
        parked = (self._flight == 0).nonzero(as_tuple=False).squeeze(-1)
        if parked.numel():
            pose = torch.zeros(len(parked), 7, device=dev)
            pose[:, 0:3] = env.scene.env_origins[parked]
            pose[:, 2] += park_z
            pose[:, 3] = 1.0
            ball.write_root_link_pose_to_sim(pose, env_ids=parked)
            ball.write_root_link_velocity_to_sim(
                torch.zeros(len(parked), 6, device=dev), env_ids=parked)

        # 2. Tick the throw timer; anchors never fire.
        self._countdown = self._countdown - 1
        fire = (self._countdown <= 0) & ~self._anchor
        throw_ids = fire.nonzero(as_tuple=False).squeeze(-1)
        if throw_ids.numel() == 0:
            return
        self._countdown[fire] = self._sample_countdown(int(fire.sum()))
        self._flight[fire] = max(int(flight_window_s / env.step_dt), 1)

        # 3. Launch.
        n = len(throw_ids)
        root_pos = robot.data.root_link_pos_w[throw_ids]
        yq = yaw_quat(robot.data.root_link_quat_w[throw_ids])

        def u(lo: float, hi: float) -> torch.Tensor:
            return torch.rand(n, device=dev) * (hi - lo) + lo

        # Release point: in the frontal cone, so the throw is camera-visible.
        dist = u(*dist_range)
        bearing = u(-angle_deg, angle_deg) * (math.pi / 180.0)
        offset_b = torch.stack(
            [dist, dist * torch.tan(bearing), torch.zeros_like(dist)], dim=-1)
        low = torch.rand(n, device=dev) < low_arc_fraction
        start = torch.empty(n, 3, device=dev)
        start[:, 0:2] = root_pos[:, 0:2] + quat_apply(yq, offset_b)[:, 0:2]
        start[:, 2] = torch.where(
            low, u(*low_launch_height_range), u(*launch_height_range))

        # Reaction window. A DESCENDING ball (vz0=0) falls the whole flight, so
        # cap its time at the ground: past that it lands short and is no threat.
        # A LOW-ARC ball is still climbing at arrival and needs no cap.
        t_req = u(*flight_time_range)
        t_max = torch.sqrt((2.0 * (start[:, 2] - 0.05).clamp(min=1e-3)) / self._g)
        t = torch.where(low, t_req, torch.minimum(t_req, t_max))

        aim = root_pos[:, 0:2].clone()
        if lead_target:
            aim = aim + robot.data.root_link_lin_vel_w[throw_ids, :2] * t[:, None]
        if aim_noise > 0.0:
            aim = aim + aim_noise * torch.randn_like(aim)

        vel = torch.zeros(n, 3, device=dev)
        vel[:, 0:2] = (aim - start[:, 0:2]) / t[:, None]
        # z_target = z0 + vz0 t - g t^2 / 2  ->  vz0 = (z_tgt - z0)/t + g t / 2.
        vel[:, 2] = torch.where(
            low,
            (u(*low_target_z_range) - start[:, 2]) / t + 0.5 * self._g * t,
            torch.zeros(n, device=dev),
        )

        pose = torch.zeros(n, 7, device=dev)
        pose[:, 0:3] = start
        pose[:, 3] = 1.0
        ball.write_root_link_pose_to_sim(pose, env_ids=throw_ids)
        twist = torch.zeros(n, 6, device=dev)
        twist[:, 0:3] = vel
        ball.write_root_link_velocity_to_sim(twist, env_ids=throw_ids)
