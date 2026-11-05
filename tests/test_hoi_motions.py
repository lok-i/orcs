"""HOI selection, conversion and temporary-source installation contracts."""

from __future__ import annotations

import copy
import hashlib
import io
import json
from dataclasses import asdict
from pathlib import Path

import numpy as np
import pytest

from orcs.cli import stage_hoi_motions as staging
from orcs.core.data.retargeted import get_motion_files_for_objects
from orcs.tasks.uolm.roster import (
    COMPANIONS,
    MotionRoster,
    load_roster,
    resolve_roster,
)


def row(name="one", frames=31):
    theta = np.linspace(0, np.pi / 2, frames)
    quaternion = np.column_stack((np.cos(theta / 2), np.zeros((frames, 2)), np.sin(theta / 2)))
    return dict(clip_id=f"omomo/unitree_g1/{name}", subject=name, dataset="omomo",
                robot="unitree_g1", object="suitcase", qc_pass=True, fps=30.,
                object_scale=.83, object_aug_scale=1.,
                dof_names=[f"joint_{i}" for i in range(29)],
                root_pos=np.tile([0., 0., .8], (frames, 1)).tolist(),
                root_rot=np.tile([1., 0., 0., 0.], (frames, 1)).tolist(),
                dof_pos=np.tile(np.arange(29), (frames, 1)).tolist(),
                object_pos=np.tile([0., 0., .3], (frames, 1)).tolist(),
                object_rot=quaternion.tolist(), contact_link_names=["left_palm_pad"],
                per_link_contact_flags=np.ones((frames, 1), bool).tolist(),
                robot_foot_ground_contact_flags=np.ones((frames, 2), bool).tolist(),
                object_ground_contact_sequence=np.zeros(frames, bool).tolist())


def test_roster_is_the_requested_63_with_control_and_short_clips():
    roster = load_roster()
    assert len(roster.motions) == len(set(roster.motions)) == 63
    assert "omomo/sub1_suitcase_010/sample1" in roster.motions
    assert roster.object_names == ("suitcase",)
    assert sum(t["destination"] == "short" for t in roster.spec["tags"].values()) == 14
    assert not roster.root().as_posix().endswith("retargeted_motions/data/unitree_g1")


def test_conversion_maps_names_and_differentiates_rotating_origin():
    r = row()
    # Physically equivalent quaternion signs must not create a spin spike.
    r["object_rot"][10:20] = (-np.asarray(r["object_rot"])[10:20]).tolist()
    names = r["dof_names"][::-1]
    state, vel, obj, labels = staging.convert_channels(r, names, [.2, 0., 0.])
    np.testing.assert_array_equal(state["joint_pos"][0], np.arange(29)[::-1])
    np.testing.assert_allclose(vel["joint_vel"], 0., atol=1e-4)
    times = np.arange(len(obj["obj_pos_w"])) / staging.FPS
    angle = times * np.pi / 2
    expected_pos = np.column_stack((-.2 * np.cos(angle), -.2 * np.sin(angle), np.full(len(angle), .3)))
    np.testing.assert_allclose(obj["obj_pos_w"], expected_pos, atol=8e-5)
    expected_vel = np.column_stack((.2 * np.pi / 2 * np.sin(angle), -.2 * np.pi / 2 * np.cos(angle), np.zeros(len(angle))))
    np.testing.assert_allclose(obj["obj_lin_vel_w"][2:-2], expected_vel[2:-2], atol=.003)
    np.testing.assert_allclose(obj["obj_ang_vel_w"][:, 2], np.pi / 2, atol=1e-4)
    assert len(labels["source_link_contact"]) == len(state["root_pos"]) == len(obj["obj_pos_w"])


def test_bad_source_names_and_zero_quaternions_are_rejected():
    r = row()
    with pytest.raises(ValueError, match="Joint-name"):
        staging.convert_channels(r, ["wrong"] * 29, [0., 0., 0.])
    r["root_rot"][0] = [0.] * 4
    with pytest.raises(ValueError, match="quaternion"):
        staging.convert_channels(r, r["dof_names"], [0., 0., 0.])


def test_parquet_reader_selects_exact_rows_without_video(tmp_path):
    pa = pytest.importorskip("pyarrow")
    pq = pytest.importorskip("pyarrow.parquet")
    rows = [row("one"), row("unselected"), row("two")]
    source = tmp_path / "source.parquet"
    pq.write_table(pa.Table.from_pylist(rows), source)
    roster = MotionRoster("test", dict(motions=["omomo/two/sample1", "omomo/one/sample1"],
                                      object_names=["suitcase"]), "unused")
    assert [r["subject"] for r in staging.read_rows(source, roster)] == ["two", "one"]
    rows[2]["object"] = "other"
    pq.write_table(pa.Table.from_pylist(rows), source)
    with pytest.raises(ValueError, match="identity"):
        staging.read_rows(source, roster)
    pq.write_table(pa.Table.from_pylist(rows[:2]), source)
    with pytest.raises(ValueError, match="exact"):
        staging.read_rows(source, roster)


def test_object_selection_is_ordered_exact_and_rejects_wrong_objects(tmp_path):
    for name, obj in (("one", "suitcase"), ("two", "suitcase"), ("extra", "tire")):
        sample = tmp_path / f"omomo/{name}/sample1"
        sample.mkdir(parents=True)
        np.savez(sample / "motion.npz", joint_pos=np.zeros((3, 29)))
        (sample / "metadata.json").write_text(json.dumps(dict(object_path=f"objects/{obj}/{obj}.obj")))
    select = ("omomo/two/sample1", "omomo/one/sample1")
    grouped, selected = get_motion_files_for_objects(["suitcase"], str(tmp_path), include_motions=select)
    assert [Path(f).parent.parent.name for f in selected] == ["two", "one"]
    assert grouped["suitcase"] == selected
    with pytest.raises(FileNotFoundError, match="missing"):
        get_motion_files_for_objects(["suitcase"], str(tmp_path), include_motions=("omomo/missing/sample1",))
    with pytest.raises(ValueError, match="unexpected object"):
        get_motion_files_for_objects(["suitcase"], str(tmp_path), include_motions=("omomo/extra/sample1",))
    with pytest.raises(ValueError, match="unique"):
        get_motion_files_for_objects(["suitcase"], str(tmp_path), include_motions=(select[0], select[0]))


@pytest.fixture
def install_fixture(tmp_path, monkeypatch):
    # Isolate the install lifecycle from simulation; numerical conversion and
    # actual geometry are exercised separately, including the real-data smoke.
    from orcs.cli import _motion_npz

    payload = b"pinned parquet fixture"
    mesh = b"v 0 0 0\nv 1 0 0\nv 0 1 0\nv 0 0 1\n"
    source = dict(repo="test/data", revision="a" * 40, parquet="source.parquet",
                  mesh="source.obj", sha256=hashlib.sha256(payload).hexdigest(),
                  mesh_sha256=hashlib.sha256(mesh).hexdigest(), size_bytes=len(payload))
    roster = MotionRoster("test", dict(root="unused", object_names=["suitcase"],
        motions=["omomo/one/sample1", "omomo/two/sample1"], source=source), "roster-hash")
    requests = []

    def download(url, **kwargs):
        requests.append(url)
        return io.BytesIO(payload if url.endswith(".parquet") else mesh)

    monkeypatch.setattr(staging.urllib.request, "urlopen", download)
    monkeypatch.setattr(staging, "read_rows", lambda *args: [row("one", 4), row("two", 4)])
    monkeypatch.setattr(staging, "mesh_frame", lambda *args: dict(offset=[0., 0., 0.]))

    class Fk:
        def __init__(self, **kwargs):
            self.il_names = row()["dof_names"]

        def __call__(self, state, vel):
            return {"body_pos_w": np.zeros((len(state["root_pos"]), 37, 3))}

    class Contacts:
        def __init__(self, *args):
            self.names = ["world", "left_wrist_yaw_link", "object"]
            self.geometry_sha256 = "test-geometry"

        def __call__(self, state, obj, labels):
            n = len(state["root_pos"])
            return np.zeros((n, 3, 3), dtype=np.int8), {}, np.zeros((n, 3))

    monkeypatch.setattr(_motion_npz, "Fk", Fk)
    monkeypatch.setattr(staging, "ContactReplay", Contacts)
    return roster, tmp_path / "native", requests


def test_install_cleans_downloads_is_atomic_and_reuses_without_network(install_fixture, monkeypatch):
    roster, out, requests = install_fixture
    manifest = staging.stage(roster, out)
    assert len(requests) == 2
    assert set(out.parent.iterdir()) == {out}
    assert len(manifest["files"]) == len(COMPANIONS) * 2
    assert not list(out.rglob("*.parquet"))
    monkeypatch.setattr(staging.urllib.request, "urlopen", lambda *a, **k: pytest.fail("unexpected download"))
    assert staging.stage(roster, out) == manifest
    assert staging.verify_install(roster, out) == manifest
    motion = out / roster.motions[0] / "motion.npz"
    motion.write_bytes(b"corrupted")
    with pytest.raises(ValueError, match="checksum"):
        staging.stage(roster, out)


def test_failed_install_leaves_no_partial_output_or_download(install_fixture, monkeypatch):
    roster, out, _ = install_fixture
    convert = staging.convert_channels

    def fail_second(r, *args):
        if r["subject"] == "two":
            raise ValueError("bad second clip")
        return convert(r, *args)

    monkeypatch.setattr(staging, "convert_channels", fail_second)
    with pytest.raises(ValueError, match="second clip"):
        staging.stage(roster, out)
    assert not out.exists()
    assert list(out.parent.iterdir()) == []


def test_roster_provenance_and_missing_contacts_block_partial_selection(install_fixture):
    roster, out, _ = install_fixture
    staging.stage(roster, out)
    wrong = MotionRoster(roster.name, roster.spec, "another-hash")
    with pytest.raises(ValueError, match="Stale"):
        resolve_roster(wrong, out)
    (out / roster.motions[-1] / "contact_matrix.npz").unlink()
    with pytest.raises(FileNotFoundError, match="Incomplete"):
        resolve_roster(roster, out)


def test_native_agent_recipe_matches_uolm_and_factory_uses_roster(install_fixture, monkeypatch):
    from orcs.tasks.uolm import _TASKS, env_cfg

    roster, out, _ = install_fixture
    staging.stage(roster, out)
    monkeypatch.setattr(env_cfg, "load_roster", lambda _: roster)
    cfg = env_cfg.uolm_env_cfg(roster="test", dataset_dir=str(out))
    cmd = cfg.commands["motion"]
    assert cmd.include_motions == roster.motions
    assert cmd.ordered_object_names == ("suitcase",)
    assert cmd.exclude_motions == ()
    assert set(cfg.scene.entities["object"].variants) == {"suitcase"}
    assert cfg.episode_length_s == pytest.approx(.1 + 2.)
    rows = {task: (env, agent) for task, env, agent in _TASKS}
    legacy = asdict(rows["Orcs-Uolm-AdaptSonic"][1]())
    suitcase = asdict(rows["Orcs-Uolm-Suitcase-AdaptSonic"][1]())
    assert legacy.pop("experiment_name") != suitcase.pop("experiment_name")
    assert legacy == suitcase
    play = env_cfg.uolm_env_cfg(roster="test", dataset_dir=str(out), play=True)
    assert play.commands["motion"].start_from_zero
    assert "virtual_object_force" not in play.events
    with pytest.raises(ValueError, match="disagree"):
        env_cfg.uolm_env_cfg(roster="test", dataset_dir=str(out), object_names=("tire",))


def test_source_checksum_failure_leaves_no_output(install_fixture):
    roster, out, _ = install_fixture
    spec = copy.deepcopy(roster.spec)
    spec["source"]["sha256"] = "0" * 64
    with pytest.raises(ValueError, match="checksum"):
        staging.stage(MotionRoster("bad", spec, "hash"), out)
    assert list(out.parent.iterdir()) == []
