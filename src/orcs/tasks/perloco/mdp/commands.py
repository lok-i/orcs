"""TerrainMotionCommand — clips masked by the tile the env stands on.

uolm masks clips by which OBJECT a world simulates; perloco masks by which
TILE it stands on. Same hook, one difference that decides the whole design:

    uolm      env -> object   `sim.world_to_variant`, FIXED for the run
    perloco   env -> tile     `terrain_{types,levels}`, MOVES at runtime

`terrain_types` (the family column) is fixed at build, but `terrain_levels`
(the difficulty row) is what a terrain curriculum promotes — mjlab mutates it
in `TerrainEntity.update_env_origins` on reset. So the mask is read fresh in
`_clip_allowance` at every reset and **never cached**: a cached map would keep
a promoted env sampling the clips of the tile it no longer stands on, which
does not crash, does not log, and quietly trains tracking against the wrong
terrain.

Tile indexing is `col * n_rows + row`, over the SAME sorted roster
`orcs.tasks.perloco.terrain` builds the grid from — see `staged_roster`.
"""

from __future__ import annotations

from dataclasses import dataclass
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
    """`<root>/<family>/level_<L>/sampleN` -> `<family>/level_<L>`.

    The staged PATH is the tile<->clip pairing; there is no manifest to read
    and therefore none to desync.
    """
    return f"{sample_dir.parent.parent.name}/{sample_dir.parent.name}"


class _TileMotionLoader(ConcatMotionLoader):
    tag = "perloco"


class TerrainMotionCommand(MultiClipMotionCommand):
    """Multi-clip motion tracking where the allowed clips follow the terrain."""

    cfg: TerrainMotionCommandCfg

    def _build_loader(self) -> ConcatMotionLoader:
        """Load the library in TILE order, and remember each clip's tile.

        Tile-major ordering is not required for correctness (the mask is
        explicit) but it makes `clip_tile` contiguous, so the printed summary
        and any per-tile diagnostic read in grid order.
        """
        by_tile = scan_grouped(
            str(self.cfg.dataset_dir), _tile_key,
            exclude_motions=list(self.cfg.exclude_motions or ()),
        )
        motion_files: list[str] = []
        clip_tile: list[int] = []
        for tile_idx, key in enumerate(self.cfg.tile_keys):
            files = by_tile.get(key)
            if not files:
                raise FileNotFoundError(
                    f"tile {key!r} is in the grid but has no clips — an env "
                    f"would spawn on it with nothing to track. Restage, or "
                    f"narrow families=/levels=.")
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
        self._n_rows = len(self.cfg.levels)

        n_tiles = len(self.cfg.tile_keys)
        # (n_tiles, n_clips) — one row per tile, 1.0 on its own clips. Built
        # once (the tile->clip map IS static); it is the env->TILE lookup that
        # must stay live.
        self._tile_mask = torch.zeros(
            n_tiles, self.motion.n_clips, device=self.device)
        self._tile_mask[self._clip_tile, torch.arange(
            self.motion.n_clips, device=self.device)] = 1.0

        rows, cols = self._n_rows, n_tiles // self._n_rows
        print(f"[perloco] {cols} families x {rows} levels = {n_tiles} tiles, "
              f"{self.motion.n_clips} clips "
              f"({self.motion.n_clips / n_tiles:.1f} per tile)")

    def _env_tile(self, env_ids: torch.Tensor) -> torch.Tensor:
        """(n,) tile index for each env, READ LIVE from the terrain entity."""
        return (self._terrain.terrain_types[env_ids] * self._n_rows
                + self._terrain.terrain_levels[env_ids])

    def _clip_allowance(self, env_ids: torch.Tensor) -> torch.Tensor:
        """(n, n_clips) — the clips staged against the tile this env is on."""
        return self._tile_mask[self._env_tile(env_ids)]


@dataclass(kw_only=True)
class TerrainMotionCommandCfg(MultiClipMotionCommandCfg):
    """Clip library keyed to the sub-terrain grid.

    `families`/`levels` must be the SAME sorted roster the terrain generator
    was built from (both come from `terrain.staged_roster`) — they are what
    turns mjlab's `(terrain_level, terrain_type)` back into a staged directory.
    """

    families: tuple[str, ...] = ()
    levels: tuple[float, ...] = ()

    @property
    def tile_keys(self) -> tuple[str, ...]:
        """Tile index -> `<family>/level_<L>`, family-major (`col * n_rows + row`)."""
        return tuple(f"{f}/level_{level:.2f}"
                     for f in self.families for level in self.levels)

    def build(self, env: ManagerBasedRlEnv) -> TerrainMotionCommand:
        return TerrainMotionCommand(self, env)
