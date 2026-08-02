"""TerrainMotionCommand — clips masked by the tile the env stands on.

uolm masks clips by which OBJECT a world simulates; perloco by which TILE it
stands on. Same hook, one difference that decides the design:

    uolm      env -> object   `sim.world_to_variant`, FIXED for the run
    perloco   env -> tile     `terrain_{types,levels}`, MOVES at runtime

The column is fixed at build but the row is what a curriculum promotes (mjlab
mutates `terrain_levels` in `update_env_origins` on reset). So the env->tile
lookup is read fresh every reset and **never cached** — a cached map keeps a
promoted env sampling its old tile's clips, which does not crash, does not log,
and quietly trains tracking against the wrong terrain.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING

import torch

from orcs.core.data.loader import ConcatMotionLoader
from orcs.core.data.scan import scan_grouped
from orcs.core.mdp.commands import MultiClipMotionCommand, MultiClipMotionCommandCfg

if TYPE_CHECKING:
    from mjlab.envs import ManagerBasedRlEnv

__all__ = ["TerrainMotionCommand", "TerrainMotionCommandCfg"]


def _tile_key(sample_dir: Path) -> str:
    """`.../<family>/level_<L>/sampleN` -> `<family>/level_<L>`. The staged
    PATH is the pairing, so there is no manifest to desync."""
    return f"{sample_dir.parent.parent.name}/{sample_dir.parent.name}"


class _TileMotionLoader(ConcatMotionLoader):
    tag = "perloco"


class TerrainMotionCommand(MultiClipMotionCommand):
    """Multi-clip motion tracking where the allowed clips follow the terrain."""

    cfg: TerrainMotionCommandCfg

    def _build_loader(self) -> ConcatMotionLoader:
        """Load in TILE order, remembering each clip's tile."""
        by_tile = scan_grouped(str(self.cfg.dataset_dir), _tile_key)
        motion_files: list[str] = []
        clip_tile: list[int] = []
        for tile_idx, key in enumerate(self.cfg.tile_keys):
            files = by_tile.get(key, [])
            if keep := self.cfg.clips.get(key):
                files = [f for f in files if Path(f).parent.name in keep]
            if not files:
                raise FileNotFoundError(
                    f"tile {key!r} is in the grid with no clips — an env would "
                    f"spawn on it with nothing to track. Widen the roster.")
            motion_files.extend(files)
            clip_tile.extend([tile_idx] * len(files))

        self._clip_tile = torch.tensor(clip_tile, device=self.device)
        return _TileMotionLoader(
            self.cfg.dataset_dir, self.device, motion_files=motion_files)

    def _init_task(self) -> None:
        terrain = self._env.scene.terrain
        if terrain is None or getattr(terrain, "terrain_levels", None) is None:
            raise ValueError(
                "TerrainMotionCommand needs a generated sub-terrain grid "
                "(TerrainEntityCfg(terrain_type='generator')) — without one "
                "there is no env->tile map to mask clips by.")
        self._terrain = terrain
        n_tiles = len(self.cfg.tile_keys)
        # tile -> its clips. Static; the env->TILE lookup is what stays live.
        self._tile_mask = torch.zeros(
            n_tiles, self.motion.n_clips, device=self.device)
        self._tile_mask[self._clip_tile, torch.arange(
            self.motion.n_clips, device=self.device)] = 1.0

        cols = n_tiles // self.cfg.n_rows
        print(f"[perloco] {cols} x {self.cfg.n_rows} = {n_tiles} tiles, "
              f"{self.motion.n_clips} clips "
              f"({self.motion.n_clips / n_tiles:.1f} per tile)")

    def _env_tile(self, env_ids: torch.Tensor) -> torch.Tensor:
        """(n,) tile index for each env, READ LIVE from the terrain entity."""
        return (self._terrain.terrain_types[env_ids] * self.cfg.n_rows
                + self._terrain.terrain_levels[env_ids])

    def _clip_allowance(self, env_ids: torch.Tensor) -> torch.Tensor:
        """(n, n_clips) — the clips staged against the tile this env is on."""
        return self._tile_mask[self._env_tile(env_ids)]


@dataclass(kw_only=True)
class TerrainMotionCommandCfg(MultiClipMotionCommandCfg):
    """Clip library keyed to the sub-terrain grid.

    `tile_keys` and `n_rows` come from the same `Roster` the grid was built
    from — that is what turns mjlab's `(terrain_level, terrain_type)` back into
    a staged directory.
    """

    tile_keys: tuple[str, ...] = ()
    n_rows: int = 1
    clips: dict[str, tuple[str, ...]] = field(default_factory=dict)

    def build(self, env: ManagerBasedRlEnv) -> TerrainMotionCommand:
        return TerrainMotionCommand(self, env)
