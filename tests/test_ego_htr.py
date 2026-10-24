"""EgoHTR file/CLI contracts; real retarget/viewer behavior is checked separately."""

from __future__ import annotations

import json
import shutil
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import pytest

from orcs.ego_htr.robot import load_motion
from orcs.ego_htr.source import recipe_for
from orcs.ego_htr.workspace import Workspace, digest, read_json, write_json


@pytest.fixture
def workspace(tmp_path):
    ws = Workspace(tmp_path / "workspace")
    ws.create_clip(
        "rock", "03", dict(sequence="rock", start_frame=1200, end_frame=1203, fps=30)
    )
    return ws


def make_run(ws, label="candidate"):
    run = ws.new_run("03", label, assets={}, source=ws.clip("03")["source"])
    q = np.zeros((3, 36))
    q[:, 3] = 1
    np.savez_compressed(
        ws.run_dir(run) / "motion.npz",
        qpos=q,
        fps=30.0,
        smpl_scale=0.7,
        source_frames=np.arange(1200, 1203),
        frame_ids=np.arange(3),
        robot_joint_names=np.array([f"joint{i}" for i in range(29)]),
    )
    return run


def test_failed_and_late_solves_preserve_latest_and_accepted(workspace):
    ws = workspace
    baseline = ws.finish(make_run(ws, "accepted"), {"passed": True})
    ws.accept("03", baseline["id"])
    earlier = make_run(ws, "earlier")
    later = ws.finish(make_run(ws, "later"), {"passed": True})
    ws.finish(earlier, {"passed": True})
    ws.finish(make_run(ws, "failed"), {"error": "solver exit"}, "failed")
    assert ws.run("03")["id"] == later["id"]
    assert ws.run("03", "accepted")["id"] == baseline["id"]


def test_notes_bind_exact_revision_source_frame_and_hash(workspace):
    ws = workspace
    old = ws.finish(make_run(ws, "old"), {})
    ws.finish(make_run(ws, "new"), {})
    note = ws.comment(old, 2, "left heel", motion_hash=old["motion_sha256"])
    assert (note["run"], note["frame"], note["source_frame"]) == (old["id"], 2, 1202)
    assert note["source_time_s"] == 1202 / 30
    ws.append_note(
        "03", dict(type="disposition", note_id=note["id"], status="addressed")
    )
    assert ws.notes("03", pending=True) == [note]
    ws.append_note(
        "03", dict(type="disposition", note_id=note["id"], status="resolved")
    )
    assert ws.notes("03", pending=True) == []
    with pytest.raises(ValueError, match="changed"):
        ws.comment(old, 0, "bad hash", motion_hash="stale")
    with pytest.raises(ValueError, match="outside"):
        ws.comment(old, 3, "bad frame")


def test_prune_protects_active_results_and_keeps_history(workspace):
    ws = workspace
    old = ws.finish(make_run(ws, "old"), {})
    accepted = ws.finish(make_run(ws, "accepted"), {})
    ws.accept("03", accepted["id"])
    latest = ws.finish(make_run(ws, "latest"), {})
    running = make_run(ws, "running")
    for run in (accepted, latest, running):
        with pytest.raises(ValueError, match="Cannot prune"):
            ws.prune("03", [run["id"]], apply=True)
    for alias in ("latest", "accepted"):
        with pytest.raises(ValueError, match="Cannot prune"):
            ws.prune("03", [alias], apply=True)
    before = digest(ws.run_dir(old) / "motion.npz")
    assert not ws.prune("03", [old["id"]])["applied"]
    assert digest(ws.run_dir(old) / "motion.npz") == before
    ws.prune("03", [old["id"]], apply=True)
    assert ws.run("03", old["id"])["status"] == "pruned"
    assert list(ws.run_dir(old).iterdir()) == [ws.run_dir(old) / "run.json"]
    assert ws.new_run("03", "next")["id"].startswith("0004_")


def test_workspace_rejects_traversal_and_external_symlinks(workspace, tmp_path):
    ws = workspace
    (ws.root / "escape").symlink_to(tmp_path)
    for value in ("../outside", "escape/outside"):
        with pytest.raises(ValueError, match="escapes"):
            ws.path(value)
    with pytest.raises(ValueError, match="Invalid identifier"):
        ws.new_run("03", "../escape")


def test_run_allocation_and_notes_are_serialized(workspace):
    ws = workspace
    with ThreadPoolExecutor(max_workers=4) as pool:
        runs = list(pool.map(lambda _: make_run(ws), range(8)))
        notes = list(pool.map(lambda run: ws.comment(run, 0, run["id"]), runs))
    assert len({r["id"] for r in runs}) == 8
    assert {n["id"] for n in ws.notes("03")} == {n["id"] for n in notes}


def test_metadata_replace_is_atomic(workspace):
    target = workspace.root / "atomic.json"
    write_json(target, {"unchanged": True})
    with pytest.raises(ValueError):
        write_json(target, {"bad": float("nan")})
    assert read_json(target) == {"unchanged": True}
    assert not list(workspace.root.glob("*.tmp"))


def test_completed_run_cannot_be_finalized_again(workspace):
    run = workspace.finish(make_run(workspace), {"passed": True})
    before = digest(workspace.run_dir(run) / "run.json")
    with pytest.raises(ValueError, match="immutable"):
        workspace.finish(run, {"passed": False}, "failed")
    assert digest(workspace.run_dir(run) / "run.json") == before


def test_motion_contract_rejects_invalid_scale_and_quaternions(workspace):
    run = make_run(workspace)
    path = workspace.run_dir(run) / "motion.npz"
    data = load_motion(path)
    data["smpl_scale"] = np.array(0.0)
    np.savez(path, **data)
    with pytest.raises(ValueError, match="scale"):
        load_motion(path)
    data["smpl_scale"] = np.array(0.7)
    data["qpos"][0, 3] = 0.5
    np.savez(path, **data)
    with pytest.raises(ValueError, match="quaternions"):
        load_motion(path)


def test_prepared_workspace_is_relocatable(workspace, tmp_path):
    run = workspace.finish(make_run(workspace), {})
    moved = tmp_path / "moved"
    shutil.move(workspace.root, moved)
    ws = Workspace(moved)
    assert ws.run("03")["motion_sha256"] == digest(ws.run_dir(run) / "motion.npz")


def test_configure_preserves_venv_python_symlink(workspace, tmp_path):
    python = tmp_path / "venv/bin/python"
    python.parent.mkdir(parents=True)
    python.symlink_to(sys.executable)
    assert workspace.configure(umr_python=python)["umr_python"] == str(python)


def test_presets_are_portable_and_preserve_reviewed_parameters():
    presets = [recipe_for(preset=p) for p in ("03", "04", "05", "06")]
    assert [p["collision"]["mode"] for p in presets] == [
        "tiled",
        "tiled",
        "tiled",
        "whole",
    ]
    assert presets[0]["config"]["solver"]["egohtr_review_elbow_posture_target"] == 0.6
    assert (
        len(presets[0]["config"]["solver"]["robot_self_collision_exclude_body_pairs"])
        == 147
    )
    assert "/HDD/" not in json.dumps(presets)


def test_cli_help_imports_without_optional_environment():
    code = """
import runpy, sys
class Block:
    def find_spec(self, fullname, *args):
        if fullname.split('.')[0] in {'numpy','mujoco','viser','trimesh','torch','smplx','coacd'}:
            raise AssertionError('Heavy import: ' + fullname)
sys.meta_path.insert(0, Block())
sys.argv = ['ego_htr', '--help']
runpy.run_module('orcs.cli.ego_htr', run_name='__main__')
"""
    result = subprocess.run(
        [sys.executable, "-c", code], capture_output=True, text=True
    )
    assert result.returncode == 0, result.stderr
    assert "retarget" in result.stdout and "view" in result.stdout


def test_bounded_range_reader_never_falls_back_to_full_download():
    from orcs.ego_htr.remote import RemoteFile

    # The protocol refuses a server ignoring Range before consuming its body.
    class Response:
        status_code = 200
        headers = {}

        def close(self):
            pass

        def __enter__(self):
            return self

        def __exit__(self, *args):
            pass

        def iter_content(self, *args, **kwargs):
            raise AssertionError("Must not consume a full response")

    class Session:
        def get(self, *args, **kwargs):
            return Response()

    stream = RemoteFile("https://example.invalid/data", 1000)
    stream.session.close()
    stream.session = Session()
    with pytest.raises((ValueError, RuntimeError), match="206|Range|range"):
        stream.read(10)
