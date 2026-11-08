"""Preserve the full terrain atlas, matching each trial to its authored tile."""

import torch

from orcs.core.motion_benchmark.contract import BenchmarkSpec
from orcs.core.robustness import strip_domain


def prepare(cfg, nominal: bool) -> None:
    cfg.events.pop("policy_update_counter", None)
    if nominal:
        strip_domain(cfg)


def bind_batch(env, clips: list[dict]) -> None:
    terrain = env.scene.terrain
    command = env.command_manager.get_term("motion")
    tiles = torch.tensor([clip["context"] for clip in clips], device=env.device)
    terrain.terrain_levels[:] = tiles % command.cfg.n_rows
    terrain.terrain_types[:] = tiles // command.cfg.n_rows
    terrain.env_origins[:] = terrain.terrain_origins[
        terrain.terrain_levels, terrain.terrain_types]


SPEC = BenchmarkSpec(prepare=prepare, bind_batch=bind_batch)
