"""Freeze reviewed EgoHTR selections into PerLoco's ordinary staged layout."""

from __future__ import annotations

import json
import tempfile
from pathlib import Path

from orcs.core.paths import DATA_ROOT
from orcs.ego_htr.workspace import digest, key, write_json


def stage(ws, clips, run="accepted", out=None, device="cuda:0", batch=256):
    # Resolve/pin every input before constructing the FK simulation.
    from orcs.cli.stage_terrain_motions import stage_source
    from orcs.tasks.perloco.sources.egohtr import EgoHtrSource

    source = EgoHtrSource(ws.root, families=clips, run=run)
    output = Path(out or DATA_ROOT / "terrain_motions/egohtr").resolve()
    output.mkdir(parents=True, exist_ok=True)
    records = list(source.clips())
    identity = dict(format=1, fps=50.0,
                    clips={c.family: c.meta for c in records})
    # A changed reference needs a fresh output root: never silently replace the
    # dataset beneath a running learner. Repeating the same export is idempotent.
    reusable = True
    for clip in records:
        folder = output / clip.family
        manifest = folder / "export.json"
        if folder.exists():
            if not manifest.is_file():
                raise ValueError(f"Unmanaged staged crop at {folder}; choose a fresh --out")
            saved = json.loads(manifest.read_text())
            expected = dict(format=1, fps=50.0, source=clip.meta)
            if saved["identity"] != expected:
                raise ValueError(f"{clip.family} has another export; choose a fresh --out")
            for relative, sha in saved["files"].items():
                path = (folder / relative).resolve()
                if not path.is_relative_to(folder) or digest(path) != sha:
                    raise ValueError(f"Staged export changed: {path}")
        else:
            reusable = False

    if not reusable:
        with tempfile.TemporaryDirectory(prefix=".stage-", dir=output) as tmp:
            temporary = Path(tmp)
            stage_source(source, temporary, fps=50.0, batch=batch, device=device)
            for clip in records:
                folder = temporary / clip.family
                write_json(folder / "export.json", dict(
                    identity=dict(format=1, fps=50.0, source=clip.meta),
                    files={str(p.relative_to(folder)): digest(p)
                           for p in sorted(folder.rglob("*")) if p.is_file()},
                ))
                if not (output / clip.family).exists():
                    folder.rename(output / clip.family)

    # The selection is an ordinary PerLoco roster, independently reusable for
    # train and play. It never points back into the review workspace.
    roster = output / "rosters" / f"{key(identity)}.toml"
    roster.parent.mkdir(exist_ok=True)
    roster.write_text(
        "# Generated EgoHTR selection; terrain and motion are pinned in export.json.\n"
        + "families = " + json.dumps([c.family for c in records]) + "\nlevels = [0.0]\n"
    )
    return dict(root=str(output), roster=str(roster), reused=reusable,
                clips=[dict(clip=c.family, label=c.meta["label"], run=c.name,
                            motion_sha256=c.meta["motion_sha256"],
                            terrain_hulls=len(c.meta["terrain_mesh_sha256"]),
                            manifest=str(output / c.family / "export.json"))
                       for c in records],
                task="Orcs-PerLoco-EgoHtr-AdaptSonic")
