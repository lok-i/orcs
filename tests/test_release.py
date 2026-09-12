from __future__ import annotations

import hashlib
import io
from typing import get_args, get_type_hints

import pytest

import orcs.release as release

EXPECTED_TASKS = {
    "Orcs-Dodge-AdaptSonic",
    "Orcs-PerLoco-Grail-AdaptSonic",
    "Orcs-PerLoco-OmRe-AdaptSonic",
    "Orcs-Uolm-AdaptSonic",
}


def test_release_manifest_contract():
    manifest = release.release_manifest()
    assert manifest["repo_id"] == "lkrajan/orcs"
    assert manifest["revision"] == "v0.1.0"
    assert set(manifest["models"]) == EXPECTED_TASKS
    for task_id, model in manifest["models"].items():
        checkpoint = model["checkpoint"]
        assert checkpoint["path"] == f"{task_id}/checkpoint.pt"
        assert checkpoint["size_bytes"] > 0
        assert len(checkpoint["sha256"]) == 64


def test_play_accepts_release_agent():
    import mjlab.scripts.play as play

    choices = get_args(get_type_hints(play.PlayConfig)["agent"])
    assert "release" in choices


def _manifest(payload: bytes) -> dict:
    return {
        "schema": "orcs.release.v1",
        "repo_id": "example/orcs",
        "revision": "test",
        "models": {
            "Orcs-Test": {
                "checkpoint": {
                    "path": "Orcs-Test/checkpoint.pt",
                    "size_bytes": len(payload),
                    "sha256": hashlib.sha256(payload).hexdigest(),
                }
            }
        },
    }


def test_download_is_verified_atomic_and_cached(tmp_path, monkeypatch):
    payload = b"released model"
    manifest = _manifest(payload)
    calls = []

    monkeypatch.setenv("ORCS_RELEASE_ROOT", str(tmp_path))
    monkeypatch.setattr(release, "release_manifest", lambda: manifest)

    def open_url(url, timeout):
        calls.append((url, timeout))
        return io.BytesIO(payload)

    monkeypatch.setattr(release.urllib.request, "urlopen", open_url)
    path = release.ensure_released_model("Orcs-Test")
    assert path == tmp_path / "test" / "Orcs-Test" / "checkpoint.pt"
    assert path.read_bytes() == payload
    assert len(calls) == 1

    assert release.ensure_released_model("Orcs-Test") == path
    assert len(calls) == 1


def test_bad_download_does_not_replace_existing_file(tmp_path, monkeypatch):
    manifest = _manifest(b"expected")
    monkeypatch.setenv("ORCS_RELEASE_ROOT", str(tmp_path))
    monkeypatch.setattr(release, "release_manifest", lambda: manifest)
    destination = tmp_path / "test" / "Orcs-Test" / "checkpoint.pt"
    destination.parent.mkdir(parents=True)
    destination.write_bytes(b"old")
    monkeypatch.setattr(
        release.urllib.request, "urlopen", lambda *_args, **_kwargs: io.BytesIO(b"bad")
    )

    with pytest.raises(RuntimeError, match="integrity verification"):
        release.ensure_released_model("Orcs-Test")
    assert destination.read_bytes() == b"old"
