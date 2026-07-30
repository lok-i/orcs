"""Compat shim: let mjlab's train/play scripts tolerate orcs's multi-clip motion command.

Both scripts flag a task as "tracking" via ``isinstance(cmd, MotionCommandCfg)`` and then force
an *external single-file* motion resolution:

  - ``train.py``  -> passes ``registry_name`` to the runner (default runner rejects the kwarg).
  - ``play.py``   -> demands ``--motion-file`` / WandB registry, else raises before it can play
                     a local checkpoint.

A task's multi-clip cfg (e.g. UOLM's :class:`OmniObjectMotionCommandCfg`) is named ``"motion"``
(tracking rewards/obs key on it) and *is* a ``MotionCommandCfg`` subclass, but it loads its own
multi-clip dataset from ``dataset_dir`` — so that single-file path is both unnecessary and fatal
for local runs.

Rather than rename the command or fork the scripts, we swap the ``MotionCommandCfg`` symbol inside
each script for a metaclass sentinel whose ``isinstance`` reports those cfgs as *not* plain
tracking cfgs. Genuine mjlab single-file tracking tasks are untouched (still report True).

Task classes arrive as :func:`apply`'s ``multi_clip_cfgs`` argument — core never imports from
:mod:`orcs.tasks`; ``orcs/__init__.py`` wires the two together.

Idempotent, import-time. ``import orcs`` runs before either script's ``run_*``, since both call
``import mjlab.tasks`` (which imports this package via the entry point) before doing any work.
"""

from __future__ import annotations

import importlib
from typing import Literal

from mjlab.tasks.tracking.mdp.commands import MotionCommandCfg

# Scripts that gate on isinstance(cmd, MotionCommandCfg).
_PATCHED_SCRIPTS = ("mjlab.scripts.train", "mjlab.scripts.play")


def _single_file_sentinel(multi_clip_cfgs: tuple[type, ...]) -> type:
    """Build the scripts' ``MotionCommandCfg`` stand-in.

    ``isinstance`` is True only for *single-file* tracking cfgs — anything in
    ``multi_clip_cfgs`` reports False. An empty tuple is a no-op passthrough.
    """

    class _SingleFileMotionMeta(type):
        def __instancecheck__(cls, obj: object) -> bool:
            return isinstance(obj, MotionCommandCfg) and not isinstance(
                obj, multi_clip_cfgs
            )

    class _SingleFileMotionCfg(metaclass=_SingleFileMotionMeta):
        """Drop-in for the scripts' ``MotionCommandCfg`` isinstance target."""

    return _SingleFileMotionCfg


def apply(multi_clip_cfgs: tuple[type, ...] = ()) -> None:
    """Patch mjlab in place. ``multi_clip_cfgs``: task command cfgs that own
    their own dataset and must escape mjlab's single-file motion resolution."""
    sentinel = _single_file_sentinel(multi_clip_cfgs)
    for name in _PATCHED_SCRIPTS:
        mod = importlib.import_module(name)
        mod.MotionCommandCfg = sentinel  # type: ignore[attr-defined]
    _patch_play_init_agent()
    _muffle_mesh_support_warning()
    _patch_put_data_nccdmax()


def _patch_put_data_nccdmax() -> None:
    """Bound mujoco_warp's CCD workspace — the VRAM whale for mesh scenes.

    mjwarp sizes its GJK/EPA + multiccd scratch arrays by ``naccdmax`` rows
    (~6.8 KB/row at ccd_iterations=50), and ``naccdmax`` silently defaults to
    ``nconmax * nworld`` because mjlab's ``Simulation._finish_init`` never
    passes ``nccdmax``. Only mesh-pair candidate contacts consume CCD rows,
    so for the omni-object scenes (nconmax=150+, 12k worlds) the default
    allocates tens of GB of workspace that box/capsule contacts never touch.

    Wrap ``put_data`` to cap CCD rows per world at ``ORCS_NCCDMAX`` (default
    64, clamped to nconmax). Undershoot is loud, not silent: mjwarp printf's
    "CCD overflow - please increase naccdmax" to stderr and drops the
    contact — watch run logs and raise the env var if it appears.
    """
    import functools
    import os

    import mujoco_warp as mjwarp

    if getattr(mjwarp.put_data, "_orcs_nccdmax", False):
        return
    _orig = mjwarp.put_data

    @functools.wraps(_orig)
    def put_data(*args, **kwargs):
        nconmax = kwargs.get("nconmax")
        if (nconmax is not None
                and kwargs.get("nccdmax") is None
                and kwargs.get("naccdmax") is None):
            nccdmax = int(os.environ.get("ORCS_NCCDMAX", "64"))
            kwargs["nccdmax"] = min(nccdmax, nconmax)
        return _orig(*args, **kwargs)

    put_data._orcs_nccdmax = True  # type: ignore[attr-defined]
    mjwarp.put_data = put_data


def _muffle_mesh_support_warning() -> None:
    """Silence libmujoco's "mesh_support could not find support vertex" spam.

    Emitted by the HOST model's GJK (engine_collision_gjk) for the omni-object
    variant scenes — padded mesh slots / thin convex-decomposition slivers give
    degenerate support queries. Physics runs on mujoco_warp (GPU), so the host
    warning is cosmetic; rollouts are unaffected (verified 2026-07-12). Every
    OTHER MuJoCo warning still prints. Root-cause (mjlab variant padding) is
    parked upstream.
    """
    import mujoco

    def _warn(msg: str) -> None:
        if "mesh_support could not find support vertex" in msg:
            return
        print(f"WARNING: {msg}")

    mujoco.set_mju_user_warning(_warn)


def _patch_play_init_agent() -> None:
    """Add ``--agent initial`` to mjlab's play script (no core edits, no fork).

    "initial" = instantiate the task's ACTUAL agent (actor cfg, base_checkpoint
    and all) but load NO training checkpoint — the freshly constructed policy
    is rolled out. For adapter agents this is the frozen base bit-exact
    (zero-init LoRA / zero-init sidecar); for TaRa it's the untrained MLP.
    Unlike ``--agent zero`` (zero ACTIONS), this debugs/visualizes the real
    model stack: obs plumbing, base-ckpt load, normalizers, action heads.

    Mechanics: main() resolves ``PlayConfig`` and ``run_play`` as module
    globals at call time, so swapping both on the module is enough. tyro
    picks up the widened Literal; video/ckpt-hotswap are trained-only and
    stay untouched.
    """
    from dataclasses import asdict, dataclass

    import torch

    play = importlib.import_module("mjlab.scripts.play")
    if getattr(play, "_orcs_init_agent", False):
        return
    play._orcs_init_agent = True
    _orig_run_play = play.run_play
    _OrigPlayConfig = play.PlayConfig

    @dataclass(frozen=True)
    class PlayConfig(_OrigPlayConfig):  # type: ignore[misc, valid-type]
        agent: Literal["zero", "random", "trained", "initial"] = "trained"

    def run_play(task_id: str, cfg):
        if cfg.agent != "initial":
            return _orig_run_play(task_id, cfg)

        play.configure_torch_backends()
        device = cfg.device or ("cuda:0" if torch.cuda.is_available() else "cpu")
        env_cfg = play.load_env_cfg(task_id, play=True)
        agent_cfg = play.load_rl_cfg(task_id)
        if cfg.no_terminations:
            env_cfg.terminations = {}
            print("[INFO]: Terminations disabled")
        if cfg.num_envs is not None:
            env_cfg.scene.num_envs = cfg.num_envs

        env = play.ManagerBasedRlEnv(cfg=env_cfg, device=device, render_mode=None)
        env = play.RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)
        runner_cls = play.load_runner_cls(task_id) or play.MjlabOnPolicyRunner
        runner = runner_cls(env, asdict(agent_cfg), device=device)
        policy = runner.get_inference_policy(device=device)
        base = (agent_cfg.actor.get("base_checkpoint")
                if isinstance(agent_cfg.actor, dict) else None)
        print("[INFO]: agent=initial — freshly constructed policy, NO training "
              f"checkpoint{f' (frozen base: {base})' if base else ''}")

        import os as _os
        if cfg.viewer == "auto":
            has_display = bool(_os.environ.get("DISPLAY")
                               or _os.environ.get("WAYLAND_DISPLAY"))
            resolved_viewer = "native" if has_display else "viser"
        else:
            resolved_viewer = cfg.viewer
        if resolved_viewer == "native":
            play.NativeMujocoViewer(env, policy).run()
        elif resolved_viewer == "viser":
            play.ViserPlayViewer(env, policy, checkpoint_manager=None).run()
        else:
            raise RuntimeError(f"Unsupported viewer backend: {resolved_viewer}")
        env.close()

    play.PlayConfig = PlayConfig
    play.run_play = run_play
