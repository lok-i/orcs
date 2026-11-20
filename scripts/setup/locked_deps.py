"""Apply reviewed source patches and verify editable installs from a consumer lock.

Standard library only: this also runs before ORCS and its dependencies are installed.
"""

from __future__ import annotations

import argparse
import importlib.metadata as metadata
import json
import subprocess
import tomllib
from pathlib import Path
from urllib.parse import unquote, urlparse


def git(repo: Path, *args: str, input: bytes | None = None) -> bytes:
    return subprocess.check_output(["git", "-C", str(repo), *args], input=input)


def patch_record(repo: Path) -> Path:
    path = Path(git(repo, "rev-parse", "--git-path", "consumer-source.patch").decode().strip())
    return path if path.is_absolute() else repo / path


def remove_owned_patch(repo: Path) -> None:
    record = patch_record(repo)
    if not record.exists():
        return
    diff = git(repo, "diff", "--binary", "HEAD")
    if not diff:
        record.unlink()
        return
    if diff != record.read_bytes() or git(repo, "diff", "--cached"):
        raise RuntimeError(f"Unreviewed edits in {repo}; refusing to replace them.")
    git(repo, "apply", "--reverse", "-", input=diff)
    record.unlink()


def apply_patches(root: Path, lock: dict, *, check: bool = False) -> None:
    for row in lock.values():
        patches = row.get("patches", [])
        if not patches:
            continue
        repo = root / row["path"]
        expected = b"".join((root / path).read_bytes() for path in patches)
        actual = git(repo, "diff", "--binary", "HEAD")
        if actual == expected:
            if not check:
                patch_record(repo).write_bytes(expected)
            print(f"[PATCH OK] {row['path']}")
            continue
        if check:
            raise RuntimeError(f"{repo} does not match its reviewed source patch.")
        remove_owned_patch(repo)
        if git(repo, "diff", "--binary", "HEAD"):
            raise RuntimeError(f"Unreviewed edits in {repo}; refusing to patch over them.")
        git(repo, "apply", "--check", "-", input=expected)
        git(repo, "apply", "-", input=expected)
        if git(repo, "diff", "--binary", "HEAD") != expected:
            raise RuntimeError(f"Applied patch differs from the reviewed diff: {repo}")
        patch_record(repo).write_bytes(expected)
        print(f"[PATCHED ] {row['path']}")


def check_installs(root: Path, lock: dict) -> None:
    paths = [root, *(root / r["path"] for r in lock.values() if r.get("pip_install"))]
    for path in paths:
        project = tomllib.loads((path / "pyproject.toml").read_text())["project"]
        name = project["name"]
        dist = metadata.distribution(name)
        origin = json.loads(dist.read_text("direct_url.json") or "{}")
        installed = Path(unquote(urlparse(origin.get("url", "")).path))
        if not origin.get("dir_info", {}).get("editable") or installed.resolve() != path.resolve():
            raise RuntimeError(f"{name} is not the editable checkout at {path}")
        if dist.version != project["version"]:
            raise RuntimeError(f"{name} metadata is stale; rerun sync_deps.sh")
        print(f"[IMPORT  ] {name} {dist.version}: {path}")
    try:
        legacy = metadata.version("mujoco")
    except metadata.PackageNotFoundError:
        return
    if legacy.startswith("3."):
        raise RuntimeError("Legacy mujoco 3.x overlaps mujoco-core. Use a fresh project environment.")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("patch", "check-patches", "before-checkout", "check-install"))
    parser.add_argument("root", type=Path)
    parser.add_argument("name", nargs="?")
    args = parser.parse_args()
    root = args.root.resolve()
    lock = json.loads((root / "deps.lock").read_text())
    if args.action == "before-checkout":
        remove_owned_patch(root / lock[args.name]["path"])
    elif args.action == "check-install":
        check_installs(root, lock)
    else:
        apply_patches(root, lock, check=args.action == "check-patches")


if __name__ == "__main__":
    main()
