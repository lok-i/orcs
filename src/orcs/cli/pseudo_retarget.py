"""Bake a kinematic G1 retarget trajectory from an SMPL motion.

The frozen SONIC base produces every action.  A task-agnostic controller adds
temporary torso/body/object wrenches while the trajectory is recorded.  The
result is ``seed_state.npz`` beside (or explicitly separate from) the source;
the SMPL file remains the only tracking reference.

Examples:
  orcs-pseudo-retarget --scene uolm --source clip.pkl --output /tmp/seed_state.npz
  orcs-pseudo-retarget --scene uolm --source path/to/sample0
  orcs-pseudo-retarget --scene perloco-grail --source \
      data/terrain_motions/grail/curb_000/level_0.00/sample0
"""

from __future__ import annotations

import argparse
import tempfile
from contextlib import ExitStack, contextmanager
from dataclasses import asdict
from pathlib import Path
from typing import Iterator

import numpy as np
import torch

from orcs.core.assisted_retarget import AssistanceSnapshot, AssistedMotionController
from orcs.core.data.seeds import SeedMotion
from orcs.core.paths import DATA_ROOT

_SETTLE_STEPS = 100
_CAPTURE_STABLE_STEPS = 5
_CAPTURE_MAX_ERROR_M = 0.20
_CAPTURE_ERROR_SPREAD_M = 0.005


def _sample_dir(path: Path) -> Path:
    return path.parent if path.name == "smpl_motion.npz" else path


@contextmanager
def _resolve_source(
    source: str | None,
    *,
    object_path: str | None,
    z_up: bool,
) -> Iterator[tuple[Path, bool]]:
    """Yield ``(sample_dir, temporary)`` for a staged directory or raw clip."""
    if source is not None:
        path = Path(source).expanduser().resolve()
        sample = _sample_dir(path)
        if sample.is_dir():
            for name in ("motion.npz", "smpl_motion.npz"):
                if not (sample / name).exists():
                    raise FileNotFoundError(f"{sample} has no {name}")
            yield sample, False
            return

    from orcs.tasks.uolm.smpl_data import load_smpl_clip, stage_clip

    with tempfile.TemporaryDirectory(prefix="orcs_pseudo_source_") as tmp:
        joints, root_quat, joints_viz = load_smpl_clip(source, z_up)
        sample = Path(tmp) / "clip" / "sample0"
        stage_clip(sample, joints, root_quat, joints_viz, object_path)
        yield sample, True


@contextmanager
def _isolated_flat_dataset(sample: Path) -> Iterator[tuple[Path, Path]]:
    """Expose exactly one UOLM sample through the normal flat dataset loader."""
    with tempfile.TemporaryDirectory(prefix="orcs_pseudo_dataset_") as tmp:
        root = Path(tmp)
        mirror = root / "clip" / "sample0"
        mirror.mkdir(parents=True)
        for source_file in sample.iterdir():
            if source_file.is_file():
                (mirror / source_file.name).symlink_to(source_file.resolve())
        yield root, mirror


def _grail_roster(sample: Path, stack: ExitStack) -> str:
    root = DATA_ROOT / "terrain_motions" / "grail"
    try:
        relative = sample.resolve().relative_to(root.resolve())
    except ValueError as exc:
        raise ValueError(
            f"PerLoco-GRAIL source must be staged under {root}, got {sample}"
        ) from exc
    if len(relative.parts) != 3 or not relative.parts[1].startswith("level_"):
        raise ValueError(
            "expected <family>/level_<L>/sampleN below the staged GRAIL root, "
            f"got {relative}"
        )
    family, level_dir, sample_name = relative.parts
    level = float(level_dir.removeprefix("level_"))
    tmp = Path(stack.enter_context(tempfile.TemporaryDirectory(prefix="orcs_roster_")))
    roster = tmp / "single.toml"
    roster.write_text(
        f'families = ["{family}"]\n'
        f"levels = [{level}]\n\n"
        "[clips]\n"
        f'"{family}/{level_dir}" = ["{sample_name}"]\n'
    )
    return str(roster)


def _build_cfg(scene: str, sample: Path, flat_root: Path | None, stack: ExitStack):
    if scene == "uolm":
        from orcs.tasks.uolm.env_cfg import uolm_env_cfg

        assert flat_root is not None
        cfg = uolm_env_cfg(command_space="smpl", play=True)
        command = cfg.commands["motion"]
        command.dataset_dir = str(flat_root)
        command.ordered_object_names = None
        command.exclude_motions = None
        command.motion_file = str(flat_root / "clip/sample0/motion.npz")
        object_name = "object"
    else:
        from orcs.tasks.perloco.env_cfg import grail_env_cfg

        roster = _grail_roster(sample, stack)
        cfg = grail_env_cfg(
            agent="sonic", command_space="smpl", play=True, roster=roster
        )
        object_name = None

    cfg.scene.num_envs = 1
    cfg.episode_length_s = 1e6
    cfg.terminations = {}
    cfg.auto_reset = False
    cfg.commands["motion"].start_from_zero = True
    return cfg, object_name


def _set_frame(env, local_frame: int) -> None:
    command = env.command_manager.get_term("motion")
    clip_id = torch.zeros(env.num_envs, dtype=torch.long, device=env.device)
    command._clip_ids[:] = clip_id
    command.time_steps[:] = command.motion.clip_offsets[clip_id] + local_frame
    command._steps_past_end.zero_()
    command.update_relative_body_poses()


def _observations(env, local_frame: int):
    _set_frame(env, local_frame)
    env.sim.forward()
    env.sim.sense()
    return env.observation_manager.compute(update_history=True)


def _reset_robot_to_nominal(env) -> None:
    robot = env.scene["robot"]
    root = robot.data.default_root_state.clone()
    root[:, 0:3] += env.scene.env_origins
    robot.write_root_state_to_sim(root)
    robot.write_joint_state_to_sim(
        robot.data.default_joint_pos.clone(), robot.data.default_joint_vel.clone()
    )
    env.scene.write_data_to_sim()
    env.sim.forward()


def _align_robot_root_to_source(controller: AssistedMotionController) -> None:
    """Place the nominal robot root at the first source pelvis pose.

    Only the floating base is aligned.  Joint positions remain nominal, so
    this is a source-derived initial condition for force-assisted settling,
    not a frame-wise robot pose projection.
    """
    env = controller.env
    robot = controller.robot
    clip_ids = torch.zeros(env.num_envs, dtype=torch.long, device=env.device)
    frame = controller.command.motion.clip_offsets[clip_ids]
    pelvis_idx = controller.body_names.index("pelvis")

    root = robot.data.default_root_state.clone()
    root[:, 0:3] = controller.targets(frame)[:, pelvis_idx]
    root[:, 3:7] = controller.command.motion.smpl_root_quat[frame]
    root[:, 7:13] = 0.0
    robot.write_root_state_to_sim(root)
    env.scene.write_data_to_sim()
    env.sim.forward()


class _SettlingMonitor:
    """Detect a low-error plateau without task-specific tuning."""

    def __init__(self) -> None:
        self.errors: list[float] = []

    def update(self, error_m: float) -> None:
        self.errors.append(error_m)
        self.errors = self.errors[-_CAPTURE_STABLE_STEPS:]

    @property
    def converged(self) -> bool:
        if len(self.errors) < _CAPTURE_STABLE_STEPS:
            return False
        if not np.isfinite(self.errors).all():
            return False
        return (
            max(self.errors) <= _CAPTURE_MAX_ERROR_M
            and max(self.errors) - min(self.errors) <= _CAPTURE_ERROR_SPREAD_M
        )


def _cpu(value: torch.Tensor) -> np.ndarray:
    return value.detach().cpu().numpy().copy()


class _Recorder:
    def __init__(self, controller: AssistedMotionController, fps: float) -> None:
        self.controller = controller
        self.fps = fps
        self.rows: list[dict[str, np.ndarray | bool | float]] = []

    def append(
        self,
        frame: int,
        action: torch.Tensor,
        assist: AssistanceSnapshot,
    ) -> None:
        robot = self.controller.robot
        body_ids = self.controller.body_ids
        body_pos = robot.data.body_link_pos_w[:, body_ids]
        body_error = torch.linalg.vector_norm(assist.body_targets_w - body_pos, dim=-1)
        finite = torch.isfinite(
            torch.cat(
                (
                    robot.data.root_link_pos_w,
                    robot.data.root_link_quat_w,
                    robot.data.root_link_lin_vel_w,
                    robot.data.root_link_ang_vel_w,
                    robot.data.joint_pos,
                    robot.data.joint_vel,
                ),
                dim=-1,
            )
        ).all(-1)
        row: dict[str, np.ndarray | bool | float] = {
            "source_frame_idx": np.array(frame, dtype=np.int64),
            "robot_root_pos_w": _cpu(robot.data.root_link_pos_w[0]),
            "robot_root_quat_w": _cpu(robot.data.root_link_quat_w[0]),
            "robot_root_lin_vel_w": _cpu(robot.data.root_link_lin_vel_w[0]),
            "robot_root_ang_vel_w": _cpu(robot.data.root_link_ang_vel_w[0]),
            "joint_pos": _cpu(robot.data.joint_pos[0]),
            "joint_vel": _cpu(robot.data.joint_vel[0]),
            "last_action": _cpu(action[0]),
            "body_pos_w": _cpu(body_pos[0]),
            "body_quat_w": _cpu(robot.data.body_link_quat_w[0, body_ids]),
            "assist_force_w": _cpu(assist.body_forces_w[0]),
            "assist_torque_w": _cpu(assist.body_torques_w[0]),
            "body_tracking_error": _cpu(body_error[0]),
            "assist_saturation": float(assist.saturation[0].item()),
            # This state was realized by MuJoCo, whose hard constraints already
            # bound the joints.  Soft-limit proximity is a training cost, not a
            # reason to break the source/seed coverage contract.
            "valid": bool(finite[0].item()),
        }
        if self.controller.object is not None:
            obj = self.controller.object
            assert assist.object_target_pos_w is not None
            assert assist.object_force_w is not None
            assert assist.object_torque_w is not None
            row.update(
                object_pos_w=_cpu(obj.data.root_link_pos_w[0]),
                object_quat_w=_cpu(obj.data.root_link_quat_w[0]),
                object_lin_vel_w=_cpu(obj.data.root_link_lin_vel_w[0]),
                object_ang_vel_w=_cpu(obj.data.root_link_ang_vel_w[0]),
                object_target_pos_w=_cpu(assist.object_target_pos_w[0]),
                object_assist_force_w=_cpu(assist.object_force_w[0]),
                object_assist_torque_w=_cpu(assist.object_torque_w[0]),
            )
        self.rows.append(row)

    def finish(
        self,
        *,
        source_path: str,
        capture_steps: int,
    ) -> SeedMotion:
        keys = self.rows[0].keys()
        arrays = {
            key: np.stack([np.asarray(row[key]) for row in self.rows])
            for key in keys
        }
        return SeedMotion(
            fps=self.fps,
            joint_names=tuple(self.controller.robot.joint_names),
            body_names=self.controller.body_names,
            source_path=source_path,
            capture_steps=capture_steps,
            morphology_scale=self.controller.morphology_scale,
            **arrays,
        )


def _run(
    cfg,
    *,
    object_name: str | None,
    sample: Path,
    output: Path,
    device: str,
) -> SeedMotion:
    from mjlab.envs import ManagerBasedRlEnv
    from mjlab.rl import RslRlVecEnvWrapper
    from mjlab.rl.runner import MjlabOnPolicyRunner

    from orcs.core.rl import SMPL_CKPT, adapt_sonic_agent_cfg

    raw_env = ManagerBasedRlEnv(cfg=cfg, device=device, render_mode=None)
    agent_cfg = adapt_sonic_agent_cfg(
        "orcs_pseudo_retarget", base_checkpoint=SMPL_CKPT
    )
    env = RslRlVecEnvWrapper(raw_env, clip_actions=agent_cfg.clip_actions)
    runner = MjlabOnPolicyRunner(env, asdict(agent_cfg), device=device)
    policy = runner.get_inference_policy(device=device)

    try:
        _reset_robot_to_nominal(raw_env)
        controller = AssistedMotionController(raw_env, object_name=object_name)
        _align_robot_root_to_source(controller)
        command = raw_env.command_manager.get_term("motion")
        num_frames = int(command.motion.clip_lengths[0].item())
        if num_frames < 1:
            raise ValueError("source clip contains no frames")

        action = torch.zeros(
            raw_env.num_envs, raw_env.action_manager.total_action_dim,
            device=raw_env.device,
        )
        monitor = _SettlingMonitor()
        capture_steps = _SETTLE_STEPS
        final_error = float("inf")
        for _ in range(_SETTLE_STEPS):
            obs = _observations(raw_env, 0)
            assist = controller.apply()
            with torch.no_grad():
                action = policy(obs)
            env.step(action)
            final_error = float(assist.body_error.mean().item())
            monitor.update(final_error)
        if not monitor.converged:
            raise RuntimeError(
                "kinematic retarget settling did not converge after "
                f"{_SETTLE_STEPS} steps (final mean point error "
                f"{final_error:.3f} m)"
            )

        recorder = _Recorder(controller, fps=1.0 / raw_env.step_dt)
        obs = _observations(raw_env, 0)
        assist = controller.apply()
        recorder.append(0, action, assist)

        for frame in range(1, num_frames):
            obs = _observations(raw_env, frame)
            assist = controller.apply()
            with torch.no_grad():
                action = policy(obs)
            env.step(action)
            recorder.append(frame, action, assist)

        seed = recorder.finish(
            source_path=str(sample / "smpl_motion.npz"),
            capture_steps=capture_steps,
        )
        seed.save(output)
        return seed
    finally:
        raw_env.close()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument(
        "--scene", choices=("uolm", "perloco-grail"), default="uolm"
    )
    parser.add_argument(
        "--source",
        default=None,
        help="staged sample dir, smpl_motion.npz, SONIC pkl, or prepared npz",
    )
    parser.add_argument("--object", default=None, help="object_motion npz for a raw clip")
    parser.add_argument(
        "--z-up", action="store_true", help="raw SONIC pose/transl are already z-up"
    )
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--device", default=None)
    args = parser.parse_args()

    if args.scene == "perloco-grail" and args.source is None:
        parser.error("--scene perloco-grail requires a staged --source sample")

    device = args.device or ("cuda:0" if torch.cuda.is_available() else "cpu")
    with ExitStack() as stack:
        sample, temporary = stack.enter_context(
            _resolve_source(args.source, object_path=args.object, z_up=args.z_up)
        )
        flat_root = None
        if args.scene == "uolm":
            flat_root, _ = stack.enter_context(_isolated_flat_dataset(sample))
        cfg, object_name = _build_cfg(args.scene, sample, flat_root, stack)
        if args.output is not None:
            output = args.output.expanduser().resolve()
        elif not temporary:
            output = sample / "seed_state.npz"
        else:
            output = Path.cwd() / "seed_state.npz"
        seed = _run(
            cfg,
            object_name=object_name,
            sample=sample,
            output=output,
            device=device,
        )
    valid = int(seed.valid.sum())
    print(
        f"[kinematic-retarget] wrote {seed.num_frames} frames -> {output}\n"
        f"  valid={valid}/{seed.num_frames}, capture_steps={seed.capture_steps}, "
        f"morphology_scale={seed.morphology_scale:.3f}"
    )


if __name__ == "__main__":
    main()
