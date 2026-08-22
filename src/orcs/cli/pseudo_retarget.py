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
  orcs-pseudo-retarget --scene perloco-grail --all
  orcs-pseudo-retarget --scene uolm --all
  orcs-pseudo-retarget --scene uolm --motion-set small-cube-table --all
"""

from __future__ import annotations

import argparse
import subprocess
import sys
import tempfile
from collections.abc import Callable
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
_SETTLE_MAX_STEPS = 500
_CAPTURE_STABLE_STEPS = 5
_CAPTURE_MAX_ERROR_M = 0.20
_CAPTURE_ERROR_SPREAD_M = 0.005


def _sample_dir(path: Path) -> Path:
    return path.parent if path.name == "smpl_motion.npz" else path


def _selected_grail_samples() -> list[Path]:
    """Return every staged sample selected by the packaged GRAIL roster."""
    from orcs.tasks.perloco.roster import load_roster

    root = DATA_ROOT / "terrain_motions" / "grail"
    roster = load_roster("grail", root)
    samples = [
        motion.parent
        for key in roster.tile_keys
        for motion in sorted((root / key).glob("sample*/motion.npz"))
        if not roster.clips.get(key) or motion.parent.name in roster.clips[key]
    ]
    if not samples:
        raise FileNotFoundError(f"GRAIL roster selected no samples under {root}")
    missing = [sample for sample in samples if not (sample / "smpl_motion.npz").exists()]
    if missing:
        raise FileNotFoundError(
            f"{len(missing)}/{len(samples)} selected GRAIL samples have no "
            f"smpl_motion.npz (e.g. {missing[0]}); restage with "
            "scripts/setup/perceptive_locomotion.sh"
        )
    return samples


def _seed_is_complete(sample: Path) -> bool:
    """Whether a resumable batch may safely skip this sample."""
    try:
        seed = SeedMotion.load(sample / "seed_state.npz")
        with np.load(sample / "smpl_motion.npz") as source:
            num_source_frames = int(source["smpl_joints"].shape[0])
    except (FileNotFoundError, KeyError, OSError, ValueError):
        return False
    return seed.num_frames == num_source_frames and bool(seed.valid.all())


def _object_seed_is_complete(sample: Path) -> bool:
    if not _seed_is_complete(sample):
        return False
    try:
        seed = SeedMotion.load(sample / "seed_state.npz")
        with np.load(sample / "seed_state.npz") as raw:
            # Experimental hand springs predate the shipped seed contract.
            # Treat those files as resumable-incomplete so a normal batch run
            # rewrites them without the dead channels.
            if any(name.startswith("hand_object") or name.startswith("object_hand")
                   for name in raw.files):
                return False
    except (FileNotFoundError, KeyError, OSError, ValueError):
        return False
    # A source-adapter schema upgrade may rewrite the staged human/object
    # channels while leaving an older generated seed beside them.  Timestamp
    # ordering keeps resumable batches honest without coupling the generic
    # seed schema to one source adapter.
    seed_path = sample / "seed_state.npz"
    source_paths = (
        sample / "smpl_motion.npz",
        sample / "object_motion.npz",
        sample / "metadata.json",
    )
    source_is_newer = any(
        path.exists() and path.stat().st_mtime_ns > seed_path.stat().st_mtime_ns
        for path in source_paths
    )
    return seed.object_pos_w is not None and not source_is_newer


def _run_grail_batch(*, device: str | None, overwrite: bool) -> None:
    """Retarget the full roster in one vectorized mjlab rollout."""
    samples = _selected_grail_samples()
    pending = [
        sample for sample in samples if overwrite or not _seed_is_complete(sample)
    ]
    skipped = len(samples) - len(pending)
    print(
        f"[kinematic-retarget] GRAIL roster: {len(samples)} clips "
        f"({skipped} complete, {len(pending)} pending)"
    )

    if not pending:
        return

    run_device = device or ("cuda:0" if torch.cuda.is_available() else "cpu")
    failures = _run_grail_vectorized(pending, device=run_device)

    # A bad reconstruction should not throw away a successful vectorized
    # corpus.  Retry only the exceptional worlds in isolation so their full
    # error remains easy to diagnose.
    retry_failures: list[Path] = []
    for index, sample in enumerate(failures, start=1):
        print(f"\n[retry {index}/{len(failures)}] {sample}", flush=True)
        command = [
            sys.executable,
            "-m",
            "orcs.cli.pseudo_retarget",
            "--scene",
            "perloco-grail",
            "--source",
            str(sample),
        ]
        command.extend(("--device", run_device))
        if subprocess.run(command, check=False).returncode != 0:
            retry_failures.append(sample)

    complete = sum(_seed_is_complete(sample) for sample in samples)
    print(
        f"\n[kinematic-retarget] complete={complete}/{len(samples)}, "
        f"failed={len(retry_failures)}"
    )
    if retry_failures:
        print("failed samples:")
        for sample in retry_failures:
            print(f"  {sample}")
        raise SystemExit(1)


def _run_grail_vectorized(samples: list[Path], *, device: str) -> list[Path]:
    """Run one pending GRAIL sample per mjlab environment.

    Clip lengths are ragged: completed worlds hold their last source frame
    while longer worlds finish, and their recorders stop at their own length.
    """
    from orcs.tasks.perloco.env_cfg import grail_env_cfg

    cfg = grail_env_cfg(
        agent="sonic", command_space="smpl", play=True, roster=None
    )
    cfg.scene.num_envs = len(samples)
    cfg.episode_length_s = 1e6
    cfg.terminations = {}
    cfg.auto_reset = False
    cfg.commands["motion"].start_from_zero = True

    def prepare_scene(raw_env, command, clip_ids: torch.Tensor) -> None:
        # Pin each world to the terrain tile paired with its clip.  The full
        # terrain grid already exists in every mjlab world; this only selects
        # the correct origin within that grid.
        terrain = raw_env.scene.terrain
        assert terrain is not None
        tile_ids = command._clip_tile[clip_ids]
        terrain.terrain_types[:] = torch.div(
            tile_ids, command.cfg.n_rows, rounding_mode="floor"
        )
        terrain.terrain_levels[:] = tile_ids % command.cfg.n_rows
        terrain.env_origins[:] = terrain.terrain_origins[
            terrain.terrain_levels, terrain.terrain_types
        ]

    return _run_vectorized_corpus(
        cfg,
        samples=samples,
        loader_samples=samples,
        device=device,
        prepare_scene=prepare_scene,
    )


def _run_vectorized_corpus(
    cfg,
    *,
    samples: list[Path],
    loader_samples: list[Path],
    device: str,
    prepare_scene: Callable | None = None,
    object_name: str | None = None,
) -> list[Path]:
    """Shared one-world-per-clip rollout engine used by every ``--all`` scene."""
    from mjlab.envs import ManagerBasedRlEnv
    from mjlab.rl import RslRlVecEnvWrapper
    from mjlab.rl.runner import MjlabOnPolicyRunner

    from orcs.core.rl import SMPL_CKPT, adapt_sonic_agent_cfg

    if len(samples) != len(loader_samples):
        raise ValueError("output and loader sample counts differ")
    raw_env = ManagerBasedRlEnv(cfg=cfg, device=device, render_mode=None)
    agent_cfg = adapt_sonic_agent_cfg(
        "orcs_pseudo_retarget", base_checkpoint=SMPL_CKPT
    )
    env = RslRlVecEnvWrapper(raw_env, clip_actions=agent_cfg.clip_actions)
    runner = MjlabOnPolicyRunner(env, asdict(agent_cfg), device=device)
    policy = runner.get_inference_policy(device=device)

    try:
        command = raw_env.command_manager.get_term("motion")
        clip_by_sample = {
            Path(locator).parent.resolve(): clip_id
            for clip_id, locator in enumerate(command.motion.motion_files)
        }
        try:
            clip_ids = torch.tensor(
                [clip_by_sample[sample.resolve()] for sample in loader_samples],
                dtype=torch.long,
                device=raw_env.device,
            )
        except KeyError as exc:
            raise ValueError(
                f"batch sample is absent from the configured loader: {exc.args[0]}"
            ) from exc
        if prepare_scene is not None:
            prepare_scene(raw_env, command, clip_ids)

        _reset_robot_to_nominal(raw_env)
        controller = AssistedMotionController(
            raw_env, object_name=object_name, clip_ids=clip_ids
        )
        _align_robot_root_to_source(controller, clip_ids)
        clip_lengths = command.motion.clip_lengths[clip_ids]
        if (clip_lengths < 1).any():
            raise ValueError("source corpus contains an empty clip")

        action = torch.zeros(
            raw_env.num_envs,
            raw_env.action_manager.total_action_dim,
            device=raw_env.device,
        )
        monitors = [_SettlingMonitor() for _ in samples]
        capture_steps = _SETTLE_MAX_STEPS
        zero_frames = torch.zeros_like(clip_ids)
        for step in range(_SETTLE_MAX_STEPS):
            obs = _observations(raw_env, zero_frames, clip_ids)
            assist = controller.apply()
            with torch.no_grad():
                action = policy(obs)
            env.step(action)
            errors = assist.body_error.mean(-1).detach().cpu().tolist()
            for monitor, error in zip(monitors, errors, strict=True):
                monitor.update(float(error))
            if step + 1 >= _SETTLE_STEPS and all(
                monitor.converged for monitor in monitors
            ):
                capture_steps = step + 1
                break

        converged = [monitor.converged for monitor in monitors]
        failed = [
            sample for sample, ok in zip(samples, converged, strict=True) if not ok
        ]
        if failed:
            print(
                f"[kinematic-retarget] batched settle: "
                f"{len(samples) - len(failed)}/{len(samples)} converged; "
                f"{len(failed)} queued for isolated retry"
            )

        recorders = [
            _Recorder(controller, fps=1.0 / raw_env.step_dt) for _ in samples
        ]
        assist = controller.apply()
        for env_id, recorder in enumerate(recorders):
            if converged[env_id]:
                recorder.append(0, action, assist, env_id=env_id)

        max_frames = int(clip_lengths.max().item())
        for frame in range(1, max_frames):
            local_frames = torch.full_like(clip_ids, frame)
            obs = _observations(raw_env, local_frames, clip_ids)
            assist = controller.apply()
            with torch.no_grad():
                action = policy(obs)
            env.step(action)
            active = frame < clip_lengths
            for env_id in active.nonzero().flatten().tolist():
                if converged[env_id]:
                    recorders[env_id].append(
                        frame, action, assist, env_id=env_id
                    )

        for env_id, (sample, recorder) in enumerate(
            zip(samples, recorders, strict=True)
        ):
            if not converged[env_id]:
                continue
            seed = recorder.finish(
                source_path=str(sample / "smpl_motion.npz"),
                capture_steps=capture_steps,
                env_id=env_id,
            )
            seed.save(sample / "seed_state.npz")
            if not seed.valid.all():
                failed.append(sample)

        print(
            f"[kinematic-retarget] one batched rollout: {len(samples)} worlds, "
            f"{max_frames} max frames"
        )
        return list(dict.fromkeys(failed))
    finally:
        raw_env.close()


def _run_uolm_vectorized(samples: list[Path], *, device: str) -> list[Path]:
    """Run reconstructed robot/object clips as one heterogeneous mjlab batch."""
    from orcs.assets import (
        TABLE_CENTER_HEIGHT,
        reconstructed_object_entity_cfg,
        reconstructed_object_variants_entity_cfg,
        table_entity_cfg,
    )
    from orcs.tasks.uolm.env_cfg import uolm_env_cfg
    from orcs.tasks.uolm.sources.reconstructed import MOTION_SETS, cache_root

    sample_sets = [
        sample.resolve().relative_to(cache_root().resolve()).parts[0]
        for sample in samples
    ]
    motion_sets = tuple(dict.fromkeys(sample_sets))
    if len(motion_sets) == 1:
        object_entity = reconstructed_object_entity_cfg(motion_sets[0])
    else:
        set_to_variant = {name: index for index, name in enumerate(motion_sets)}
        assignment = tuple(set_to_variant[name] for name in sample_sets)

        def assign_worlds(num_envs: int) -> tuple[int, ...]:
            if num_envs != len(assignment):
                raise ValueError(
                    f"object assignment has {len(assignment)} worlds, "
                    f"simulation requested {num_envs}"
                )
            return assignment

        object_entity = reconstructed_object_variants_entity_cfg(
            motion_sets, assignment=assign_worlds
        )

    with _isolated_flat_datasets(samples) as (flat_root, mirrors):
        lengths = []
        for sample in samples:
            with np.load(sample / "smpl_motion.npz") as source:
                lengths.append(int(source["smpl_joints"].shape[0]))
        cfg = uolm_env_cfg(
            command_space="smpl",
            play=True,
            _bootstrap_motion=(
                str(mirrors[0] / "motion.npz"),
                str(flat_root),
                max(lengths),
            ),
            _object_entity=object_entity,
        )
        command_cfg = cfg.commands["motion"]
        command_cfg.dataset_dir = str(flat_root)
        command_cfg.ordered_object_names = None
        command_cfg.exclude_motions = None
        command_cfg.motion_file = str(mirrors[0] / "motion.npz")
        command_cfg.start_from_zero = True
        table_motion_sets = tuple(
            name for name in motion_sets if MOTION_SETS[name].support == "table"
        )
        if table_motion_sets:
            cfg.scene.entities["table"] = table_entity_cfg()
        cfg.scene.num_envs = len(samples)
        cfg.episode_length_s = 1e6
        cfg.terminations = {}
        cfg.auto_reset = False

        def prepare_scene(raw_env, command, clip_ids: torch.Tensor) -> None:
            origins = raw_env.scene.env_origins
            first_frames = command.motion.clip_offsets[clip_ids]
            obj_state = torch.cat(
                (
                    command.motion.obj_pos[first_frames] + origins,
                    command.motion.obj_quat[first_frames],
                    command.motion.obj_lin_vel[first_frames],
                    command.motion.obj_ang_vel[first_frames],
                ),
                dim=-1,
            )
            raw_env.scene["object"].write_root_state_to_sim(obj_state)

            if table_motion_sets:
                table_pose = torch.zeros(
                    raw_env.num_envs, 7, device=raw_env.device
                )
                table_pose[:, :3] = origins
                table_pose[:, 2] = -10.0
                table_pose[:, 3] = 1.0
                table_ids = torch.tensor(
                    [name in table_motion_sets for name in sample_sets],
                    dtype=torch.bool,
                    device=raw_env.device,
                )
                final_frames = command.motion.clip_ends[clip_ids] - 1
                table_pose[table_ids, :2] = (
                    origins[table_ids, :2]
                    + command.motion.obj_pos[final_frames[table_ids], :2]
                )
                table_pose[table_ids, 2] = TABLE_CENTER_HEIGHT
                env_ids = torch.arange(raw_env.num_envs, device=raw_env.device)
                raw_env.scene["table"].write_mocap_pose_to_sim(
                    table_pose, env_ids=env_ids
                )

        return _run_vectorized_corpus(
            cfg,
            samples=samples,
            loader_samples=mirrors,
            device=device,
            prepare_scene=prepare_scene,
            object_name="object",
        )


def _run_reconstructed_batch(
    *, motion_sets: tuple[str, ...], device: str | None, overwrite: bool
) -> None:
    """Normalize and retarget named reconstructed UOLM collections."""
    from orcs.tasks.uolm.sources.reconstructed import stage_motion_set

    samples = [
        sample
        for motion_set in motion_sets
        for sample in stage_motion_set(motion_set, overwrite=overwrite)
    ]
    pending = [
        sample
        for sample in samples
        if overwrite or not _object_seed_is_complete(sample)
    ]
    skipped = len(samples) - len(pending)
    print(
        f"[kinematic-retarget] reconstructed UOLM: {len(samples)} clips "
        f"({skipped} complete, {len(pending)} pending)"
    )

    if not pending:
        return

    run_device = device or ("cuda:0" if torch.cuda.is_available() else "cpu")
    failures = _run_uolm_vectorized(pending, device=run_device)

    retry_failures: list[Path] = []
    for index, sample in enumerate(failures, start=1):
        motion_set = sample.relative_to(
            Path(DATA_ROOT) / "smpl_motions/uolm/reconstructed"
        ).parts[0]
        print(f"\n[retry {index}/{len(failures)}] {sample}", flush=True)
        command = [
            sys.executable,
            "-m",
            "orcs.cli.pseudo_retarget",
            "--scene",
            "uolm",
            "--motion-set",
            motion_set,
            "--source",
            str(sample),
        ]
        command.extend(("--device", run_device))
        if subprocess.run(command, check=False).returncode != 0:
            retry_failures.append(sample)

    complete = sum(_object_seed_is_complete(sample) for sample in samples)
    print(
        f"\n[kinematic-retarget] complete={complete}/{len(samples)}, "
        f"failed={len(retry_failures)}"
    )
    if retry_failures:
        print("failed samples:")
        for sample in retry_failures:
            print(f"  {sample}")
        raise SystemExit(1)


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
            if not (sample / "smpl_motion.npz").exists():
                raise FileNotFoundError(f"{sample} has no smpl_motion.npz")
            yield sample, False
            return

    from orcs.tasks.uolm.smpl_data import load_smpl_clip, stage_clip

    with tempfile.TemporaryDirectory(prefix="orcs_pseudo_source_") as tmp:
        joints, root_quat, joints_viz = load_smpl_clip(source, z_up)
        sample = Path(tmp) / "clip" / "sample0"
        stage_clip(sample, joints, root_quat, joints_viz, object_path)
        yield sample, True


def _populate_flat_mirror(sample: Path, mirror: Path) -> None:
    """Expose one source through ObjectMotionCommand's temporary file contract."""
    mirror.mkdir(parents=True)
    for source_file in sample.iterdir():
        if source_file.is_file():
            (mirror / source_file.name).symlink_to(source_file.resolve())
    with np.load(sample / "smpl_motion.npz") as source:
        num_frames = int(source["smpl_joints"].shape[0])
    # These placeholders satisfy the phase-1 loader only; neither becomes a
    # reference or is persisted in the normalized dataset.
    if not (mirror / "motion.npz").exists():
        body_pos = np.zeros((num_frames, 35, 3), dtype=np.float32)
        body_quat = np.zeros((num_frames, 35, 4), dtype=np.float32)
        body_quat[..., 0] = 1.0
        np.savez(
            mirror / "motion.npz",
            joint_pos=np.zeros((num_frames, 29), dtype=np.float32),
            joint_vel=np.zeros((num_frames, 29), dtype=np.float32),
            body_pos_w=body_pos,
            body_quat_w=body_quat,
            body_lin_vel_w=np.zeros_like(body_pos),
            body_ang_vel_w=np.zeros_like(body_pos),
        )
    if not (mirror / "contact_matrix.npz").exists():
        from orcs.tasks.uolm.sensors import CONTACT_GRAPH_BODY_NAMES

        names = list(CONTACT_GRAPH_BODY_NAMES) + ["object", "world"]
        np.savez(
            mirror / "contact_matrix.npz",
            body_names=np.asarray(names),
            matrix=np.zeros(
                (num_frames, len(names), len(names)), dtype=np.int8
            ),
        )


@contextmanager
def _isolated_flat_datasets(
    samples: list[Path],
) -> Iterator[tuple[Path, list[Path]]]:
    """Expose a corpus through one temporary flat ObjectMotionCommand dataset."""
    with tempfile.TemporaryDirectory(prefix="orcs_pseudo_dataset_") as tmp:
        root = Path(tmp)
        mirrors = [
            root / f"clip_{index:05d}" / "sample0"
            for index in range(len(samples))
        ]
        for sample, mirror in zip(samples, mirrors, strict=True):
            _populate_flat_mirror(sample, mirror)
        yield root, mirrors


@contextmanager
def _isolated_flat_dataset(sample: Path) -> Iterator[tuple[Path, Path]]:
    """Single-clip compatibility wrapper around the corpus staging path."""
    with _isolated_flat_datasets([sample]) as (root, mirrors):
        yield root, mirrors[0]


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


def _build_cfg(
    scene: str,
    sample: Path,
    flat_root: Path | None,
    stack: ExitStack,
    motion_set: str | None = None,
):
    if scene == "uolm":
        from orcs.assets import reconstructed_object_entity_cfg, table_entity_cfg
        from orcs.tasks.uolm.env_cfg import uolm_env_cfg
        from orcs.tasks.uolm.sources.reconstructed import MOTION_SETS

        assert flat_root is not None
        object_entity = (
            reconstructed_object_entity_cfg(motion_set)
            if motion_set is not None else None
        )
        with np.load(sample / "smpl_motion.npz") as source:
            max_clip_len = int(source["smpl_joints"].shape[0])
        cfg = uolm_env_cfg(
            command_space="smpl",
            play=True,
            _bootstrap_motion=(
                str(flat_root / "clip/sample0/motion.npz"),
                str(flat_root),
                max_clip_len,
            ),
            _object_entity=object_entity,
        )
        command = cfg.commands["motion"]
        command.dataset_dir = str(flat_root)
        command.ordered_object_names = None
        command.exclude_motions = None
        command.motion_file = str(flat_root / "clip/sample0/motion.npz")
        object_name = "object"
        if motion_set is not None and MOTION_SETS[motion_set].support == "table":
            with np.load(sample / "object_motion.npz") as obj:
                final_xy = tuple(float(x) for x in obj["obj_pos_w"][-1, :2])
            cfg.scene.entities["table"] = table_entity_cfg(xy=final_xy)
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


def _set_frames(
    env,
    local_frames: int | torch.Tensor,
    clip_ids: torch.Tensor | None = None,
) -> None:
    command = env.command_manager.get_term("motion")
    if clip_ids is None:
        clip_ids = torch.zeros(env.num_envs, dtype=torch.long, device=env.device)
    local_frames = torch.as_tensor(
        local_frames, dtype=torch.long, device=env.device
    ).expand(env.num_envs)
    clip_lengths = command.motion.clip_lengths[clip_ids]
    local_frames = torch.minimum(local_frames, clip_lengths - 1)
    command._clip_ids[:] = clip_ids
    command.time_steps[:] = command.motion.clip_offsets[clip_ids] + local_frames
    command._steps_past_end.zero_()
    command.update_relative_body_poses()


def _observations(
    env,
    local_frames: int | torch.Tensor,
    clip_ids: torch.Tensor | None = None,
):
    _set_frames(env, local_frames, clip_ids)
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


def _align_robot_root_to_source(
    controller: AssistedMotionController,
    clip_ids: torch.Tensor | None = None,
) -> None:
    """Place the nominal robot root at the first source pelvis pose.

    Only the floating base is aligned.  Joint positions remain nominal, so
    this is a source-derived initial condition for force-assisted settling,
    not a frame-wise robot pose projection.
    """
    env = controller.env
    robot = controller.robot
    if clip_ids is None:
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
        *,
        env_id: int = 0,
    ) -> None:
        robot = self.controller.robot
        body_ids = self.controller.body_ids
        body_pos = robot.data.body_link_pos_w[:, body_ids]
        origin = self.controller.env.scene.env_origins[env_id]
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
            # Seed positions are environment-local.  A vectorized bake gives
            # every simulated world a layout origin; persisting that origin
            # would make RSI add it a second time in the training scene.
            "robot_root_pos_w": _cpu(
                robot.data.root_link_pos_w[env_id] - origin
            ),
            "robot_root_quat_w": _cpu(robot.data.root_link_quat_w[env_id]),
            "robot_root_lin_vel_w": _cpu(robot.data.root_link_lin_vel_w[env_id]),
            "robot_root_ang_vel_w": _cpu(robot.data.root_link_ang_vel_w[env_id]),
            "joint_pos": _cpu(robot.data.joint_pos[env_id]),
            "joint_vel": _cpu(robot.data.joint_vel[env_id]),
            "last_action": _cpu(action[env_id]),
            "body_pos_w": _cpu(body_pos[env_id] - origin),
            "body_quat_w": _cpu(robot.data.body_link_quat_w[env_id, body_ids]),
            "assist_force_w": _cpu(assist.body_forces_w[env_id]),
            "assist_torque_w": _cpu(assist.body_torques_w[env_id]),
            "body_tracking_error": _cpu(body_error[env_id]),
            "assist_saturation": float(assist.saturation[env_id].item()),
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
                object_pos_w=_cpu(obj.data.root_link_pos_w[env_id] - origin),
                object_quat_w=_cpu(obj.data.root_link_quat_w[env_id]),
                object_lin_vel_w=_cpu(obj.data.root_link_lin_vel_w[env_id]),
                object_ang_vel_w=_cpu(obj.data.root_link_ang_vel_w[env_id]),
                object_target_pos_w=_cpu(
                    assist.object_target_pos_w[env_id] - origin
                ),
                object_assist_force_w=_cpu(assist.object_force_w[env_id]),
                object_assist_torque_w=_cpu(assist.object_torque_w[env_id]),
            )
            row["valid"] = bool(
                row["valid"]
                and torch.isfinite(
                    torch.cat(
                        (
                            obj.data.root_link_pos_w[env_id],
                            obj.data.root_link_quat_w[env_id],
                            obj.data.root_link_lin_vel_w[env_id],
                            obj.data.root_link_ang_vel_w[env_id],
                        ),
                        dim=-1,
                    )
                ).all()
            )
        self.rows.append(row)

    def finish(
        self,
        *,
        source_path: str,
        capture_steps: int,
        env_id: int = 0,
    ) -> SeedMotion:
        keys = self.rows[0].keys()
        arrays = {
            key: np.stack([np.asarray(row[key]) for row in self.rows])
            for key in keys
        }
        morphology_scale = self.controller.morphology_scale
        if isinstance(morphology_scale, torch.Tensor):
            morphology_scale = float(morphology_scale[env_id].item())
        return SeedMotion(
            fps=self.fps,
            joint_names=tuple(self.controller.robot.joint_names),
            body_names=self.controller.body_names,
            source_path=source_path,
            capture_steps=capture_steps,
            morphology_scale=morphology_scale,
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
        capture_steps = _SETTLE_MAX_STEPS
        final_error = float("inf")
        for step in range(_SETTLE_MAX_STEPS):
            obs = _observations(raw_env, 0)
            assist = controller.apply()
            with torch.no_grad():
                action = policy(obs)
            env.step(action)
            final_error = float(assist.body_error.mean().item())
            monitor.update(final_error)
            if step + 1 >= _SETTLE_STEPS and monitor.converged:
                capture_steps = step + 1
                break
        if not monitor.converged:
            raise RuntimeError(
                "kinematic retarget settling did not converge after "
                f"{_SETTLE_MAX_STEPS} steps (final mean point error "
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
    from orcs.tasks.uolm.sources.reconstructed import (
        DEFAULT_MOTION_SETS,
        MOTION_SETS,
    )

    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument(
        "--scene", choices=("uolm", "perloco-grail"), default="uolm"
    )
    parser.add_argument(
        "--source",
        default=None,
        help="staged sample dir, smpl_motion.npz, SONIC pkl, or prepared npz",
    )
    parser.add_argument(
        "--all",
        dest="all_data",
        action="store_true",
        help="retarget every selected GRAIL/UOLM sample; resumes by default",
    )
    parser.add_argument(
        "--motion-set",
        choices=tuple(MOTION_SETS),
        default=None,
        help="UOLM reconstructed collection; --all defaults to the base roster",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="with --all, regenerate complete seed_state.npz files",
    )
    parser.add_argument("--object", default=None, help="object_motion npz for a raw clip")
    parser.add_argument(
        "--z-up", action="store_true", help="raw SONIC pose/transl are already z-up"
    )
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--device", default=None)
    args = parser.parse_args()

    if args.all_data:
        if args.source is not None or args.output is not None or args.object is not None:
            parser.error("--all cannot be combined with --source, --output, or --object")
        if args.scene == "perloco-grail":
            if args.motion_set is not None:
                parser.error("--motion-set is only valid with --scene uolm")
            _run_grail_batch(device=args.device, overwrite=args.overwrite)
        else:
            motion_sets = (
                (args.motion_set,)
                if args.motion_set is not None
                else DEFAULT_MOTION_SETS
            )
            _run_reconstructed_batch(
                motion_sets=motion_sets,
                device=args.device,
                overwrite=args.overwrite,
            )
        return
    if args.overwrite:
        parser.error("--overwrite requires --all")
    if args.scene == "perloco-grail" and args.source is None:
        parser.error("--scene perloco-grail requires --source or --all")
    if args.scene == "perloco-grail" and args.motion_set is not None:
        parser.error("--motion-set is only valid with --scene uolm")

    if args.scene == "uolm" and args.motion_set is not None and args.source is not None:
        source_path = Path(args.source).expanduser().resolve()
        source_sample = _sample_dir(source_path)
        if source_sample.is_dir() and not (source_sample / "smpl_motion.npz").exists():
            from orcs.tasks.uolm.sources.reconstructed import stage_source_clip

            args.source = str(stage_source_clip(source_sample, args.motion_set))

    device = args.device or ("cuda:0" if torch.cuda.is_available() else "cpu")
    with ExitStack() as stack:
        sample, temporary = stack.enter_context(
            _resolve_source(args.source, object_path=args.object, z_up=args.z_up)
        )
        flat_root = None
        if args.scene == "uolm":
            flat_root, _ = stack.enter_context(_isolated_flat_dataset(sample))
        cfg, object_name = _build_cfg(
            args.scene, sample, flat_root, stack, motion_set=args.motion_set
        )
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
