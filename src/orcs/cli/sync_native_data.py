"""Install shipped native references after the ordinary data/dependency sync.

EgoHTR motions are already at 50 Hz. This only verifies and copies their paired
files into PerLoco's existing layout; no simulator or retargeter is imported.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import tempfile
import tomllib
from importlib.resources import files
from pathlib import Path

from orcs.core.paths import DATA_ROOT, assets_source


def _selection(roster: Path | None = None) -> list[str]:
    resource = roster or files("orcs").joinpath("tasks/perloco/rosters/egohtr.toml")
    spec = tomllib.loads(resource.read_text())
    families = spec["families"]
    if not isinstance(families, list) or not families or len(set(families)) != len(families):
        raise ValueError("EgoHTR requires a nonempty list of distinct crop families")
    if spec.get("levels") != [0.0] or spec.get("clips"):
        raise ValueError("Shipped EgoHTR references have one level (0.0) and one sample per crop")
    for family in families:
        if not isinstance(family, str) or family in ("", ".", "..") or Path(family).name != family:
            raise ValueError(f"Invalid EgoHTR family: {family!r}")
    return sorted(families)


def lfs_include(checkout: Path, roster: Path | None = None) -> list[str]:
    """Union of native task selections, without importing task registration."""
    from orcs.core.data.retargeted import lfs_include as uolm_include

    return sorted(set(uolm_include(checkout)) | {
        f"data/unitree_g1/egohtr/{family}/sample1/motion.npz"
        for family in _selection(roster)
    })


def _digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _file(root: Path, relative: str) -> Path:
    path = (root / relative).resolve()
    if Path(relative).is_absolute() or not path.is_relative_to(root.resolve()):
        raise ValueError(f"Bundle path escapes its root: {relative}")
    if not path.is_file():
        raise FileNotFoundError(f"Missing processed EgoHTR file: {path}. Sync its dependency first.")
    return path


def _verify(path: Path, expected: str) -> None:
    if _digest(path) != expected:
        raise ValueError(f"EgoHTR checksum mismatch: {path}; check dependency revisions and LFS files")


def install_egohtr(
    retargeted_root: Path | None = None,
    asset_root: Path | None = None,
    out: Path | None = None,
    roster: Path | None = None,
) -> dict:
    """Copy the rostered processed bundle, refusing to replace different data."""
    retargeted_root = (retargeted_root or DATA_ROOT / "retargeted_motions").resolve()
    asset_root = (asset_root or assets_source()).resolve()
    out = (out or DATA_ROOT / "terrain_motions/egohtr").resolve()
    families = _selection(roster)
    # Validate every dependency before creating or changing the destination.
    bundles = []
    for family in families:
        sample = retargeted_root / "data/unitree_g1/egohtr" / family / "sample1"
        metadata_path = _file(sample, "metadata.json")
        meta = json.loads(metadata_path.read_text())
        if meta.get("family") != family or meta.get("level") != 0.0 or meta.get("fps") != 50.0:
            raise ValueError(f"Invalid native EgoHTR metadata: {metadata_path}")
        motion = _file(sample, "motion.npz")
        _verify(motion, meta["staged_motion_sha256"])
        tile_path = _file(asset_root, meta["terrain_path"])
        _verify(tile_path, meta["terrain_sha256"])
        tile = json.loads(tile_path.read_text())
        if tile.get("family") != family or tile.get("level") != 0.0 or not tile.get("meshes"):
            raise ValueError(f"Terrain does not match motion family: {tile_path}")
        payload = {
            "level_0.00/tile.json": tile_path,
            "level_0.00/sample0/motion.npz": motion,
            "level_0.00/sample0/metadata.json": metadata_path,
        }
        for mesh in tile["meshes"]:
            path = _file(tile_path.parent, mesh["file"])
            _verify(path, mesh["sha256"])
            payload[f"level_0.00/{mesh['file']}"] = path
        manifest = dict(
            identity=dict(format=1, fps=50.0, source=meta),
            files={relative: _digest(path) for relative, path in sorted(payload.items())},
        )
        bundles.append((family, payload, manifest))

    if out.exists():
        for family, _, manifest in bundles:
            folder = out / family
            recorded = folder / "export.json"
            if not recorded.is_file() or json.loads(recorded.read_text()) != manifest:
                raise FileExistsError(
                    f"{out} contains another export or incomplete data. Move it aside "
                    "to preserve it, or choose a fresh --out; no files were changed."
                )
            actual = {str(p.relative_to(folder)) for p in folder.rglob("*") if p.is_file()}
            if actual != set(manifest["files"]) | {"export.json"}:
                raise ValueError(f"Unexpected or missing files in installed EgoHTR crop: {folder}")
            for relative, sha in manifest["files"].items():
                _verify(_file(folder, relative), sha)
        return dict(root=str(out), clips=families, reused=True)

    out.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=".egohtr-", dir=out.parent) as temporary:
        prepared = Path(temporary) / "bundle"
        for family, payload, manifest in bundles:
            folder = prepared / family
            for relative, source in payload.items():
                target = folder / relative
                target.parent.mkdir(parents=True, exist_ok=True)
                shutil.copyfile(source, target)
                _verify(target, manifest["files"][relative])
            (folder / "export.json").write_text(json.dumps(manifest, indent=2) + "\n")
        prepared.rename(out)
    return dict(root=str(out), clips=families, reused=False)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--lfs-include", action="store_true", help="Print native roster download patterns")
    parser.add_argument("--retargeted-root", type=Path)
    parser.add_argument("--assets-root", type=Path)
    parser.add_argument("--out", type=Path)
    parser.add_argument("--roster", type=Path)
    args = parser.parse_args()
    if args.lfs_include:
        for pattern in lfs_include(args.retargeted_root or DATA_ROOT / "retargeted_motions", args.roster):
            print(f"include {pattern}")
        return
    print(json.dumps(install_egohtr(
        args.retargeted_root, args.assets_root, args.out, args.roster,
    ), indent=2))


if __name__ == "__main__":
    main()
