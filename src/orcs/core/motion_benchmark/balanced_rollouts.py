"""Balanced first-episode batches using registered tasks and inference runners."""

from __future__ import annotations

import dataclasses
import gc
import hashlib
import importlib.metadata
import json
import subprocess
import time
from copy import deepcopy
from pathlib import Path

from .metrics import SCHEMA_VERSION, balanced_batches


def sha256(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def serializable(value):
    """Readable configuration receipt; never reload executable Python YAML."""
    if dataclasses.is_dataclass(value) and not isinstance(value, type):
        return {f.name: serializable(getattr(value, f.name)) for f in dataclasses.fields(value)}
    if isinstance(value, dict):
        return {str(k): serializable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [serializable(v) for v in value]
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    if callable(value):
        return f"{value.__module__}.{value.__qualname__}"
    return str(value)


def write_json(path: Path, value) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def load_agent_recipe(default: dict, checkpoint: Path | None, override: str | None):
    """Read primitive saved recipes, accepting only PyYAML's tuple extension."""
    import yaml

    class RecipeLoader(yaml.SafeLoader):
        pass

    RecipeLoader.add_constructor("tag:yaml.org,2002:python/tuple",
                                 lambda loader, node: tuple(loader.construct_sequence(node)))
    path = Path(override) if override else (checkpoint.parent / "params/agent.yaml" if checkpoint else None)
    if path and (override or path.is_file()):
        saved = yaml.load(path.read_text(), Loader=RecipeLoader)
        if not isinstance(saved, dict):
            raise ValueError("Agent recipe must be a mapping")
        # Restore the inference contract, not training state or logging settings.
        for key in ("actor", "critic", "obs_groups", "clip_actions"):
            if key in saved:
                default[key] = saved[key]
        return default, {"path": str(path.resolve()), "sha256": sha256(path)}
    return default, {"source": "registered task recipe"}


def collect(args) -> None:
    # MJLab must own discovery before an ORCS task module is imported.
    import mjlab  # noqa: F401
    import torch
    from mjlab.envs import ManagerBasedRlEnv
    from mjlab.managers.metrics_manager import MetricsTermCfg
    from mjlab.rl import MjlabOnPolicyRunner, RslRlVecEnvWrapper
    from mjlab.tasks.registry import load_env_cfg, load_rl_cfg, load_runner_cls
    from mjlab.utils.os import get_wandb_checkpoint_path
    from mjlab.utils.torch import configure_torch_backends
    from tensordict import TensorDict

    from orcs.core.paths import REPO_ROOT

    from .capture import TrialCapture, capture_step
    from .contract import load_spec

    started = time.monotonic()
    output = Path(args.output)
    configure_torch_backends()
    torch.set_num_threads(args.cpu_threads)
    device = args.device or ("cuda:0" if torch.cuda.is_available() else "cpu")
    spec = load_spec(args.task)
    cfg = load_env_cfg(args.task)
    agent = dataclasses.asdict(load_rl_cfg(args.task))
    checkpoint = Path(args.checkpoint_file).expanduser().resolve() if args.checkpoint_file else None
    if args.wandb_run_path:
        checkpoint, _ = get_wandb_checkpoint_path(
            REPO_ROOT / "logs/rsl_rl" / agent["experiment_name"],
            Path(args.wandb_run_path), args.wandb_checkpoint_name)
    elif args.agent == "release":
        from orcs.release import ensure_released_model
        checkpoint = ensure_released_model(args.task)
    if checkpoint and not checkpoint.is_file():
        raise FileNotFoundError(checkpoint)
    agent, recipe = load_agent_recipe(agent, checkpoint, args.agent_config)
    # Runner initialization is reused, but no training buffers proportional to
    # a training rollout, optimizer updates, or W&B writers are needed here.
    agent.update(num_steps_per_env=1, logger="tensorboard", upload_model=False)
    spec.prepare(cfg, args.nominal)
    cfg.commands[spec.command_name].start_from_zero = True
    cfg.commands[spec.command_name].init_phase_anneal_iterations = 0
    cfg.curriculum = {}  # fixed evaluation worlds, no training progression
    cfg.metrics["repertoire_capture"] = MetricsTermCfg(func=capture_step)
    cfg.scene.num_envs, cfg.seed = 1, args.seed
    probe = ManagerBasedRlEnv(cfg=deepcopy(cfg), device=device)
    try:
        command = probe.command_manager.get_term(spec.command_name)
        catalog = command.motion_catalog()
        if not catalog or len({c["id"] for c in catalog}) != len(catalog):
            raise ValueError("Repertoire must contain distinct playable clip identities")
        step_dt = probe.step_dt
        points = list(getattr(command, "point_body_names", command.cfg.body_names))
        target_type = "authored SMPL points" if hasattr(command, "point_target_pos_w") else "native robot bodies"
    finally:
        probe.close()
    del probe, command
    cfg.scene.num_envs = args.num_envs
    for clip in catalog:
        clip["sha256"] = sha256(Path(clip["path"]))
        clip["sample_hashes"] = {p.name: sha256(p) for p in sorted(Path(clip["path"]).parent.iterdir())
                                 if p.is_file() and p.suffix in (".npz", ".json", ".pkl", ".npy")}
    try:
        commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=REPO_ROOT, text=True,
                                         stderr=subprocess.DEVNULL).strip()
        diff = subprocess.check_output(["git", "diff", "HEAD", "--", "src", "pyproject.toml"], cwd=REPO_ROOT)
        (output / "source.patch").write_bytes(diff)
    except (OSError, subprocess.CalledProcessError):
        commit = None
    receipt = dict(
        schema_version=SCHEMA_VERSION, task=args.task, motions=catalog,
        checkpoint=str(checkpoint) if checkpoint else None,
        checkpoint_sha256=sha256(checkpoint) if checkpoint else None,
        wandb_run_path=args.wandb_run_path, agent=args.agent, agent_recipe=recipe,
        nominal=args.nominal, seed=args.seed, num_envs=args.num_envs,
        trials_per_motion=args.trials_per_motion, bins=args.bins, step_dt=step_dt,
        horizon=spec.horizon, completion_term=spec.completion_term,
        tracking_quality=spec.tracking_quality, target_type=target_type, point_names=points,
        start_phase=0, deterministic_actions=True, assistance=False,
        source_commit=commit, device=device,
        versions={name: importlib.metadata.version(name) for name in ("orcs", "mjlab", "torch", "rsl-rl-lib", "mujoco", "mujoco-warp")},
        lock_hashes={name: sha256(REPO_ROOT / name) for name in ("deps.lock", "data.lock")
                     if (REPO_ROOT / name).is_file()},
        protocol="balanced first-episode batches; fresh startup randomization per world; startup/reset seed = seed + batch index; runtime seed = batch seed + 1000000; fixed actor seed = seed",
        torch_math="MJLab configure_torch_backends defaults (TF32 enabled)",
        state="collecting", batches=[],
    )
    package_root = Path(__file__).resolve().parents[2]
    receipt["source_hashes"] = {str(p.relative_to(package_root)): sha256(p)
                                for p in sorted(package_root.rglob("*.py"))}
    write_json(output / "config.json", {"environment": serializable(cfg), "agent": serializable(agent)})
    write_json(output / "receipt.json", receipt)
    facts = output / "trials.jsonl"
    facts.touch(exist_ok=False)
    total = len(catalog) * args.trials_per_motion
    print(f"BENCH: {len(catalog)} motions × {args.trials_per_motion} trials = {total}; capacity {args.num_envs}", flush=True)
    try:
        for batch_index, assignment in enumerate(balanced_batches(len(catalog), args.trials_per_motion, args.num_envs)):
            batch_cfg = deepcopy(cfg)
            batch_cfg.scene.num_envs = len(assignment)
            batch_seed = args.seed + batch_index
            batch_cfg.seed = batch_seed
            clips = [catalog[c] for _, c, _ in assignment]
            spec.configure_batch(batch_cfg, clips)
            env = ManagerBasedRlEnv(cfg=batch_cfg, device=device)
            capture = None
            try:
                spec.bind_batch(env, clips)
                command = env.command_manager.get_term(spec.command_name)
                actual_catalog = command.motion_catalog()
                if actual_catalog != [{k: v for k, v in c.items() if k not in ("sha256", "sample_hashes")} for c in catalog]:
                    raise RuntimeError("Repertoire changed between batches")
                ids = torch.tensor([c["index"] for c in clips], device=device)
                env_ids = torch.arange(env.num_envs, device=device)
                command.queue_reset_clips(env_ids, ids)
                env.seed(batch_seed)
                # The standard wrapper performs its initial reset itself. Queue
                # assignments first so that this is the scored reset, without
                # an extra warm-up episode or stale reset-time observation state.
                wrapped = RslRlVecEnvWrapper(env, clip_actions=agent.get("clip_actions"))
                obs = TensorDict(env.obs_buf, batch_size=[env.num_envs])
                # Initial policies must be identical across batches, too.
                torch.manual_seed(args.seed)
                runner_cls = load_runner_cls(args.task) or MjlabOnPolicyRunner
                runner = runner_cls(wrapped, deepcopy(agent), device=device)
                if checkpoint:
                    runner.load(str(checkpoint), load_cfg={"actor": True}, strict=True, map_location=device)
                policy = runner.get_inference_policy(device=device)
                if not torch.equal(command.current_clip_ids, ids):
                    raise RuntimeError("Scored reset did not select the assigned motions")
                if (command.local_reference_frames > 1).any():
                    raise RuntimeError("Scored reset did not start at phase zero")
                policy.reset()
                capture = TrialCapture(env, spec, clips, args.bins)
                env._motion_repertoire_capture = capture
                # Keep inference-time task randomness independent of actor size.
                env.seed(batch_seed + 1_000_000)
                max_steps = (max(c["frames"] for c in clips) + capture.hold + 5
                             if spec.horizon == "clip" else env.max_episode_length + 1)
                with torch.inference_mode():
                    for step in range(max_steps):
                        actions = policy(obs, stochastic_output=False)
                        finite = torch.isfinite(actions).all(-1)
                        capture.invalidate(~finite)
                        actions = torch.where(finite[:, None], actions, 0.0)
                        obs, _, dones, _ = wrapped.step(actions)
                        policy.reset(dones)
                        if not capture.active.any():
                            break
                        if step % 200 == 0:
                            print(f"BENCH batch {batch_index+1}: step {step}/{max_steps}, active {int(capture.active.sum())}", flush=True)
            finally:
                if capture is not None:
                    with facts.open("a") as stream:
                        for record in capture.records(assignment, batch_index, batch_seed):
                            stream.write(json.dumps(record, allow_nan=False) + "\n")
                env.close()
            receipt["batches"].append(dict(index=batch_index, seed=batch_seed, trials=len(assignment)))
            write_json(output / "receipt.json", receipt)
            print(f"BENCH recorded {assignment[-1][0]+1}/{total}", flush=True)
            del env, wrapped, runner, policy, capture, command, obs, actions
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
    except BaseException:
        receipt["state"] = "interrupted"
        raise
    else:
        receipt["state"] = "complete"
    finally:
        receipt["wall_seconds"] = time.monotonic() - started
        write_json(output / "receipt.json", receipt)
