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

    **Trajectory.** Pure ballistic under the model's own gravity. The default
    threat model is parameterised by the REACTION WINDOW rather than by a launch
    speed: the ball is released in the robot's frontal cone at ``dist_range``
    and must arrive after ``flight_time_range``. Two threat types are mixed:

    | type | release | vz0 | arrives at | evasion |
    |---|---|---|---|---|
    | descending | ``launch_height_range`` (~2 m) | 0 | wherever it has fallen to | sidestep / step over |
    | low arc | ``low_launch_height_range`` (~waist) | >0 | ``low_target_z_range`` (torso/head) | duck / lean |

    The aim point is the robot's xy, LED by its own velocity and then jittered,
    so a robot that walks in a straight line does not evade for free and no two
    throws share a geometry. Descending flight times are capped so the ball
    cannot land short of the robot.

    A camera-aware task may instead pass ``launch_camera_name``. Its release
    point is sampled directly in the camera frustum's intersection with the
    low ``underarm_height_range`` plane, using the camera pose captured at the
    nominal reset. Horizontal speed is sampled first, then incompatible
    footprint points are redrawn. ``underarm_vertical_speed_max`` bounds the
    launch impulse; target height is sampled only from the reachable part of
    ``underarm_target_height_range``. This mode is always a rising underarm
    throw; the legacy cone and descending branch are deliberately bypassed.

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
        self._launch_camera_name = p.get("launch_camera_name")
        self._camera_pos_0: torch.Tensor | None = None
        self._camera_mat_0: torch.Tensor | None = None
        if self._launch_camera_name is not None:
            camera = env.scene.sensors[self._launch_camera_name]
            self._camera_id = env.sim.mj_model.camera(self._launch_camera_name).id
            if camera.cfg.fovy is None:
                raise ValueError("camera-footprint throws require a perspective fovy")
            half_v = math.radians(camera.cfg.fovy) / 2.0
            self._tan_v = math.tan(half_v)
            self._tan_h = (camera.cfg.width / camera.cfg.height) * self._tan_v
            height_range = p.get("underarm_height_range", (0.4, 0.7))
            target_range = p.get("underarm_target_height_range", (0.9, 1.3))
            speed_range = p.get("horizontal_speed_range", (3.0, 5.0))
            vertical_speed_max = float(p.get("underarm_vertical_speed_max", 5.0))
            ndc_margin = p.get("launch_ndc_margin", 0.15)
            if not 0.0 <= ndc_margin < 1.0:
                raise ValueError(
                    f"launch_ndc_margin must be in [0, 1), got {ndc_margin}"
                )
            if height_range[0] > height_range[1] or target_range[0] > target_range[1]:
                raise ValueError("underarm height ranges must be ordered")
            if height_range[1] >= target_range[0]:
                raise ValueError(
                    "underarm release heights must be below target heights"
                )
            if speed_range[0] <= 0.0 or speed_range[0] > speed_range[1]:
                raise ValueError(
                    f"invalid horizontal_speed_range: {speed_range}"
                )
            min_vertical_speed = math.sqrt(
                2.0 * self._g * (target_range[0] - height_range[0])
            )
            if vertical_speed_max < min_vertical_speed:
                raise ValueError(
                    f"underarm_vertical_speed_max={vertical_speed_max} cannot "
                    f"reach target height {target_range[0]} from {height_range[0]}"
                )
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

    def _capture_nominal_camera(self, env: ManagerBasedRlEnv) -> None:
        """Cache the post-DR camera pose relative to each scene origin once."""
        if self._launch_camera_name is None or self._camera_pos_0 is not None:
            return
        self._camera_pos_0 = (
            env.sim.data.cam_xpos[:, self._camera_id] - env.scene.env_origins
        ).clone()
        self._camera_mat_0 = env.sim.data.cam_xmat[:, self._camera_id].reshape(
            self.num_envs, 3, 3
        ).clone()

    def _sample_camera_launch(
        self,
        env: ManagerBasedRlEnv,
        throw_ids: torch.Tensor,
        heights: torch.Tensor,
        ndc_margin: float,
    ) -> torch.Tensor:
        """Sample the nominal camera frustum on one horizontal launch plane."""
        if not 0.0 <= ndc_margin < 1.0:
            raise ValueError(f"launch_ndc_margin must be in [0, 1), got {ndc_margin}")
        assert self._camera_pos_0 is not None and self._camera_mat_0 is not None

        n = len(throw_ids)
        ndc = (
            torch.rand(n, 2, device=self.device) * 2.0 - 1.0
        ) * (1.0 - ndc_margin)
        ray_c = torch.stack(
            [
                ndc[:, 0] * self._tan_h,
                ndc[:, 1] * self._tan_v,
                -torch.ones(n, device=self.device),
            ],
            dim=-1,
        )
        ray_w = torch.bmm(
            self._camera_mat_0[throw_ids], ray_c.unsqueeze(-1)
        ).squeeze(-1)
        camera_pos = (
            env.scene.env_origins[throw_ids] + self._camera_pos_0[throw_ids]
        )
        plane_z = env.scene.env_origins[throw_ids, 2] + heights
        scale = (plane_z - camera_pos[:, 2]) / ray_w[:, 2]
        if bool((scale <= 0.0).any()):
            raise RuntimeError(
                f"camera {self._launch_camera_name!r} does not face the underarm "
                "launch plane; its nominal frustum has no forward intersection"
            )
        start = camera_pos + scale[:, None] * ray_w
        start[:, 2] = plane_z
        return start

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
        launch_camera_name: str | None = None,
        launch_ndc_margin: float = 0.15,
        underarm_height_range: tuple[float, float] = (0.4, 0.7),
        underarm_target_height_range: tuple[float, float] = (0.9, 1.3),
        horizontal_speed_range: tuple[float, float] = (3.0, 5.0),
        underarm_vertical_speed_max: float = 5.0,
        lead_target: bool = True,
        aim_noise: float = 0.1,
    ) -> None:
        del env_ids, interval_range_s, stand_fraction  # read at construction
        ball, robot = env.scene[ball_name], env.scene[robot_name]
        dev = self.device
        self._capture_nominal_camera(env)

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

        def u(lo: float, hi: float) -> torch.Tensor:
            return torch.rand(n, device=dev) * (hi - lo) + lo

        if launch_camera_name is not None:
            if launch_camera_name != self._launch_camera_name:
                raise ValueError(
                    "launch_camera_name cannot change after ThrowBall construction"
                )
            # Sample speed first so enforcing the vertical cap cannot silently
            # delete slow throws. A footprint point that is too far away for
            # that speed is redrawn; a robot which has left the entire reachable
            # footprint simply gets this throw deferred instead of killing the
            # training run.
            speed = u(*horizontal_speed_range)
            target_lo = (
                env.scene.env_origins[throw_ids, 2]
                + underarm_target_height_range[0]
            )
            start = torch.empty(n, 3, device=dev)
            aim = torch.empty(n, 2, device=dev)
            distance = torch.empty(n, device=dev)
            feasible = torch.zeros(n, dtype=torch.bool, device=dev)
            for _ in range(8):
                rows = (~feasible).nonzero(as_tuple=False).squeeze(-1)
                if rows.numel() == 0:
                    break
                ids = throw_ids[rows]
                m = len(ids)
                heights = torch.rand(m, device=dev) * (
                    underarm_height_range[1] - underarm_height_range[0]
                ) + underarm_height_range[0]
                candidate = self._sample_camera_launch(
                    env, ids, heights, launch_ndc_margin
                )
                candidate_aim = root_pos[rows, :2].clone()
                if lead_target:
                    lead_time = torch.linalg.vector_norm(
                        candidate_aim - candidate[:, :2], dim=-1
                    ) / speed[rows]
                    candidate_aim += (
                        robot.data.root_link_lin_vel_w[ids, :2]
                        * lead_time[:, None]
                    )
                if aim_noise > 0.0:
                    candidate_aim += aim_noise * torch.randn_like(candidate_aim)
                candidate_distance = torch.linalg.vector_norm(
                    candidate_aim - candidate[:, :2], dim=-1
                ).clamp(min=1e-3)
                candidate_t = candidate_distance / speed[rows]
                reachable_z = (
                    candidate[:, 2]
                    + underarm_vertical_speed_max * candidate_t
                    - 0.5 * self._g * candidate_t.square()
                )
                start[rows] = candidate
                aim[rows] = candidate_aim
                distance[rows] = candidate_distance
                feasible[rows] = reachable_z >= target_lo[rows]

            if not bool(feasible.all()):
                self._flight[throw_ids[~feasible]] = 0
                throw_ids = throw_ids[feasible]
                start = start[feasible]
                aim = aim[feasible]
                distance = distance[feasible]
                speed = speed[feasible]
                target_lo = target_lo[feasible]
                n = len(throw_ids)
                if n == 0:
                    return

            delta = aim - start[:, :2]
            t = distance / speed
            vel = torch.zeros(n, 3, device=dev)
            vel[:, 0:2] = delta / t[:, None]
            target_hi = torch.minimum(
                env.scene.env_origins[throw_ids, 2]
                + underarm_target_height_range[1],
                start[:, 2]
                + underarm_vertical_speed_max * t
                - 0.5 * self._g * t.square(),
            )
            target_hi = torch.maximum(target_hi, target_lo)
            target_z = target_lo + torch.rand(n, device=dev) * (target_hi - target_lo)
            vel[:, 2] = (target_z - start[:, 2]) / t + 0.5 * self._g * t
        else:
            # Legacy release point: robot-relative frontal cone.
            dist = u(*dist_range)
            bearing = u(-angle_deg, angle_deg) * (math.pi / 180.0)
            offset_b = torch.stack(
                [dist, dist * torch.tan(bearing), torch.zeros_like(dist)], dim=-1)
            yq = yaw_quat(robot.data.root_link_quat_w[throw_ids])
            low = torch.rand(n, device=dev) < low_arc_fraction
            start = torch.empty(n, 3, device=dev)
            start[:, 0:2] = root_pos[:, 0:2] + quat_apply(yq, offset_b)[:, 0:2]
            start[:, 2] = torch.where(
                low, u(*low_launch_height_range), u(*launch_height_range))

            # A descending ball falls the whole flight, so cap its time at the
            # ground: past that it lands short and is no threat.
            t_req = u(*flight_time_range)
            t_max = torch.sqrt(
                (2.0 * (start[:, 2] - 0.05).clamp(min=1e-3)) / self._g)
            t = torch.where(low, t_req, torch.minimum(t_req, t_max))

            aim = root_pos[:, 0:2].clone()
            if lead_target:
                aim = aim + robot.data.root_link_lin_vel_w[throw_ids, :2] * t[:, None]
            if aim_noise > 0.0:
                aim = aim + aim_noise * torch.randn_like(aim)

            vel = torch.zeros(n, 3, device=dev)
            vel[:, 0:2] = (aim - start[:, 0:2]) / t[:, None]
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
