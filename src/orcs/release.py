"""Download and verify the public ORCS model release."""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import tempfile
import urllib.request
from importlib.resources import files
from pathlib import Path, PurePosixPath
from typing import Any
from urllib.parse import quote

_MANIFEST_RESOURCE = "release.json"
_CHUNK_SIZE = 1024 * 1024


def release_manifest() -> dict[str, Any]:
    """Read and minimally validate the manifest bundled with ORCS."""
    manifest = json.loads(
        files("orcs").joinpath(_MANIFEST_RESOURCE).read_text(encoding="utf-8")
    )
    if manifest.get("schema") != "orcs.release.v1":
        raise RuntimeError("Unsupported ORCS release manifest schema")
    if not isinstance(manifest.get("models"), dict):
        raise RuntimeError("ORCS release manifest has no model map")
    return manifest


def released_model_ids() -> tuple[str, ...]:
    """Return task IDs that have a published model."""
    return tuple(release_manifest()["models"])


def release_cache_dir(manifest: dict[str, Any] | None = None) -> Path:
    """Return the versioned local cache directory for the current release."""
    manifest = manifest or release_manifest()
    root_override = os.environ.get("ORCS_RELEASE_ROOT")
    if root_override:
        root = Path(root_override).expanduser()
    else:
        cache_home = os.environ.get("XDG_CACHE_HOME")
        root = Path(cache_home).expanduser() if cache_home else Path.home() / ".cache"
        root = root / "orcs" / "releases"
    return root / str(manifest["revision"])


def _checkpoint_spec(task_id: str, manifest: dict[str, Any]) -> dict[str, Any]:
    try:
        spec = manifest["models"][task_id]["checkpoint"]
    except KeyError as exc:
        available = ", ".join(manifest["models"])
        raise ValueError(
            f"No released ORCS model for {task_id!r}. Available: {available}"
        ) from exc

    relative = PurePosixPath(spec["path"])
    if relative.is_absolute() or ".." in relative.parts:
        raise RuntimeError(f"Unsafe path in ORCS release manifest: {relative}")
    return spec


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(_CHUNK_SIZE), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _is_expected_file(path: Path, spec: dict[str, Any]) -> bool:
    return (
        path.is_file()
        and path.stat().st_size == int(spec["size_bytes"])
        and _sha256(path) == spec["sha256"]
    )


def _download_url(manifest: dict[str, Any], spec: dict[str, Any]) -> str:
    repo_id = quote(str(manifest["repo_id"]), safe="/")
    revision = quote(str(manifest["revision"]), safe="")
    remote_path = quote(str(spec["path"]), safe="/")
    return f"https://huggingface.co/{repo_id}/resolve/{revision}/{remote_path}"


def ensure_released_model(task_id: str, *, force: bool = False) -> Path:
    """Return a verified local checkpoint, downloading it when necessary."""
    manifest = release_manifest()
    spec = _checkpoint_spec(task_id, manifest)
    destination = release_cache_dir(manifest) / PurePosixPath(spec["path"])

    if not force and _is_expected_file(destination, spec):
        print(f"[orcs] released model already cached: {destination}")
        return destination

    destination.parent.mkdir(parents=True, exist_ok=True)
    url = _download_url(manifest, spec)
    print(f"[orcs] downloading released model: {task_id}")
    print(f"       {url}")

    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{destination.name}.", suffix=".download", dir=destination.parent
    )
    os.close(descriptor)
    temporary = Path(temporary_name)
    try:
        with urllib.request.urlopen(url, timeout=120) as response:  # noqa: S310
            with temporary.open("wb") as output:
                shutil.copyfileobj(response, output, length=_CHUNK_SIZE)
        if not _is_expected_file(temporary, spec):
            actual_size = temporary.stat().st_size
            actual_sha = _sha256(temporary)
            raise RuntimeError(
                "Released model failed integrity verification: "
                f"expected {spec['size_bytes']} bytes / {spec['sha256']}, "
                f"got {actual_size} bytes / {actual_sha}"
            )
        os.replace(temporary, destination)
    finally:
        temporary.unlink(missing_ok=True)

    print(f"[orcs] saved verified model: {destination}")
    return destination
