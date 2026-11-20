"""Source patch ownership: repeated setup must preserve unreviewed edits."""

import importlib.util
import subprocess
from pathlib import Path

import pytest

HELPER = Path(__file__).resolve().parents[1] / "scripts/setup/locked_deps.py"
spec = importlib.util.spec_from_file_location("locked_deps", HELPER)
locked = importlib.util.module_from_spec(spec)
spec.loader.exec_module(locked)


@pytest.fixture
def checkout(tmp_path):
    repo = tmp_path / "dep"
    repo.mkdir()
    subprocess.run(["git", "init", "-q", str(repo)], check=True)
    source = repo / "source.txt"
    source.write_text("upstream\n")
    locked.git(repo, "add", "source.txt")
    locked.git(repo, "-c", "user.name=Test", "-c", "user.email=test@example.invalid",
               "commit", "-qm", "upstream")
    source.write_text("reviewed change\n")
    patch = tmp_path / "source.patch"
    patch.write_bytes(locked.git(repo, "diff", "--binary", "HEAD"))
    source.write_text("upstream\n")
    lock = {"dep": {"path": "dep", "patches": ["source.patch"]}}
    return tmp_path, repo, source, patch, lock


def test_patch_is_idempotent_and_check_detects_drift(checkout):
    root, repo, source, patch, lock = checkout
    locked.apply_patches(root, lock)
    locked.apply_patches(root, lock)
    locked.apply_patches(root, lock, check=True)
    assert source.read_text() == "reviewed change\n"
    assert locked.patch_record(repo).read_bytes() == patch.read_bytes()
    source.write_text("user change\n")
    with pytest.raises(RuntimeError, match="reviewed source patch"):
        locked.apply_patches(root, lock, check=True)
    with pytest.raises(RuntimeError, match="Unreviewed edits"):
        locked.apply_patches(root, lock)
    assert source.read_text() == "user change\n"


def test_owned_patch_can_be_replaced_and_removed_before_checkout(checkout):
    root, repo, source, patch, lock = checkout
    locked.apply_patches(root, lock)
    source.write_text("next reviewed change\n")
    patch.write_bytes(locked.git(repo, "diff", "--binary", "HEAD"))
    source.write_text("reviewed change\n")
    locked.apply_patches(root, lock)
    assert source.read_text() == "next reviewed change\n"
    locked.remove_owned_patch(repo)
    assert source.read_text() == "upstream\n"
    assert not locked.patch_record(repo).exists()


def test_patch_never_replaces_unknown_or_staged_edits(checkout):
    root, repo, source, _, lock = checkout
    source.write_text("user change\n")
    with pytest.raises(RuntimeError, match="Unreviewed edits"):
        locked.apply_patches(root, lock)
    source.write_text("upstream\n")
    locked.apply_patches(root, lock)
    locked.git(repo, "add", "source.txt")
    with pytest.raises(RuntimeError, match="Unreviewed edits"):
        locked.remove_owned_patch(repo)
    assert source.read_text() == "reviewed change\n"
