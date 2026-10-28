"""Processed EgoHTR pairing, immutable install and sparse download contracts."""

from __future__ import annotations

import hashlib
import json
import subprocess
import sys
from pathlib import Path

import pytest

from orcs.cli.sync_native_data import install_egohtr, lfs_include
from orcs.core.data.retargeted import DEFAULT_OBJECT_NAMES


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


@pytest.fixture
def bundle(tmp_path):
    motions, assets, out = (tmp_path / part for part in ("motions", "assets", "installed"))
    roster = tmp_path / "roster.toml"
    roster.write_text('families = ["ascent", "window"]\nlevels = [0.0]\n')
    for family in ("ascent", "window"):
        tile = assets / "terrain/egohtr" / family
        (tile / "meshes").mkdir(parents=True)
        mesh = tile / "meshes/part.obj"
        mesh.write_text("v 0 0 0\nv 1 0 0\nv 0 1 0\nv 0 0 1\nf 1 2 3\n")
        descriptor = tile / "tile.json"
        descriptor.write_text(json.dumps(dict(
            family=family, level=0.0, size=[6.0, 6.0], floor_z=-0.04, boxes=[],
            meshes=[dict(file="meshes/part.obj", scale=[0.72]*3, sha256=sha(mesh))],
        )))
        sample = motions / "data/unitree_g1/egohtr" / family / "sample1"
        sample.mkdir(parents=True)
        # The installer treats already-processed references as opaque files.
        motion = sample / "motion.npz"
        motion.write_bytes(b"unchanged processed reference " + family.encode())
        (sample / "metadata.json").write_text(json.dumps(dict(
            family=family, level=0.0, fps=50.0, staged_motion_sha256=sha(motion),
            terrain_path=str(descriptor.relative_to(assets)), terrain_sha256=sha(descriptor),
        )))
    return motions, assets, out, roster


def test_install_is_byte_exact_portable_and_idempotent(bundle):
    motions, assets, out, roster = bundle
    assert not install_egohtr(*bundle)["reused"]
    assert install_egohtr(*bundle)["reused"]
    for family in ("ascent", "window"):
        sample = motions / "data/unitree_g1/egohtr" / family / "sample1"
        tile = out / family / "level_0.00"
        assert (tile / "sample0/motion.npz").read_bytes() == (sample / "motion.npz").read_bytes()
        assert (tile / "tile.json").read_bytes() == (assets / f"terrain/egohtr/{family}/tile.json").read_bytes()
        manifest = json.loads((out / family / "export.json").read_text())
        for relative, expected in manifest["files"].items():
            assert sha(out / family / relative) == expected
        assert str(motions) not in json.dumps(manifest)
        assert str(assets) not in json.dumps(manifest)


def test_bad_second_crop_does_not_partially_install(bundle):
    motions, assets, out, roster = bundle
    (assets / "terrain/egohtr/window/meshes/part.obj").write_text("corrupt")
    with pytest.raises(ValueError, match="checksum mismatch"):
        install_egohtr(*bundle)
    assert not out.exists()


def test_notice_upgrade_preserves_payload_and_rejects_other_metadata_changes(bundle):
    motions, _, out, _ = bundle
    install_egohtr(*bundle)
    original = {p.relative_to(out): sha(p) for p in out.rglob("*")
                if p.is_file() and p.name not in ("metadata.json", "export.json")}
    path = motions / "data/unitree_g1/egohtr/ascent/sample1/metadata.json"
    meta = json.loads(path.read_text())
    meta["source_notice"] = {"attribution": "Source authors", "license": "CC-BY-4.0"}
    path.write_text(json.dumps(meta))
    assert install_egohtr(*bundle)["refreshed_notices"]
    assert install_egohtr(*bundle)["reused"]
    assert all(sha(out / relative) == digest for relative, digest in original.items())
    assert json.loads((out / "ascent/level_0.00/sample0/metadata.json").read_text()) == meta
    meta["review_run"] = "different reference"
    path.write_text(json.dumps(meta))
    with pytest.raises(FileExistsError, match="another export"):
        install_egohtr(*bundle)


def test_notice_upgrade_does_not_hide_modified_payload(bundle):
    motions, _, out, _ = bundle
    install_egohtr(*bundle)
    path = motions / "data/unitree_g1/egohtr/ascent/sample1/metadata.json"
    meta = json.loads(path.read_text())
    meta["source_notice"] = {"attribution": "Source authors"}
    path.write_text(json.dumps(meta))
    target = out / "window/level_0.00/sample0/motion.npz"
    target.write_bytes(b"local edit")
    with pytest.raises(ValueError, match="checksum mismatch"):
        install_egohtr(*bundle)
    assert target.read_bytes() == b"local edit"
    assert "source_notice" not in json.loads(
        (out / "ascent/level_0.00/sample0/metadata.json").read_text())


def test_notice_upgrade_restores_existing_export_if_activation_fails(bundle, monkeypatch):
    motions, _, out, _ = bundle
    install_egohtr(*bundle)
    original = {p.relative_to(out): p.read_bytes() for p in out.rglob("*") if p.is_file()}
    path = motions / "data/unitree_g1/egohtr/ascent/sample1/metadata.json"
    meta = json.loads(path.read_text())
    meta["source_notice"] = {"attribution": "Source authors"}
    path.write_text(json.dumps(meta))
    rename = Path.rename

    def fail_activation(path, destination):
        if path.name == "bundle" and destination == out:
            raise OSError("simulated activation failure")
        return rename(path, destination)

    monkeypatch.setattr(Path, "rename", fail_activation)
    with pytest.raises(OSError, match="activation failure"):
        install_egohtr(*bundle)
    assert {p.relative_to(out): p.read_bytes() for p in out.rglob("*") if p.is_file()} == original


def test_installer_rejects_mismatched_terrain_even_with_valid_checksum(bundle):
    motions, assets, _, _ = bundle
    meta_path = motions / "data/unitree_g1/egohtr/ascent/sample1/metadata.json"
    meta = json.loads(meta_path.read_text())
    meta["terrain_path"] = "terrain/egohtr/window/tile.json"
    meta["terrain_sha256"] = sha(assets / meta["terrain_path"])
    meta_path.write_text(json.dumps(meta))
    with pytest.raises(ValueError, match="does not match"):
        install_egohtr(*bundle)


def test_installer_preserves_existing_exports_and_detects_drift(bundle):
    _, _, out, _ = bundle
    out.mkdir()
    marker = out / "do-not-delete.txt"
    marker.write_text("local work")
    with pytest.raises(FileExistsError, match="another export"):
        install_egohtr(*bundle)
    assert marker.read_text() == "local work"
    marker.unlink()
    out.rmdir()
    install_egohtr(*bundle)
    extra = out / "ascent/level_0.00/sample1/motion.npz"
    extra.parent.mkdir()
    extra.write_bytes(b"unexpected extra reference")
    with pytest.raises(ValueError, match="Unexpected or missing"):
        install_egohtr(*bundle)
    extra.unlink()
    target = out / "window/level_0.00/sample0/motion.npz"
    target.write_bytes(b"modified")
    with pytest.raises(ValueError, match="checksum mismatch"):
        install_egohtr(*bundle)
    assert target.read_bytes() == b"modified"


def test_terrain_references_cannot_escape_asset_root(bundle):
    motions, _, _, _ = bundle
    path = motions / "data/unitree_g1/egohtr/ascent/sample1/metadata.json"
    meta = json.loads(path.read_text())
    meta["terrain_path"] = "../outside.json"
    path.write_text(json.dumps(meta))
    with pytest.raises(ValueError, match="escapes"):
        install_egohtr(*bundle)


def test_sparse_selection_composes_uolm_and_packaged_egohtr_rosters(bundle):
    motions, _, _, roster = bundle
    expected = []
    for name in DEFAULT_OBJECT_NAMES:
        sample = motions / f"data/unitree_g1/fixture/{name}/sample1"
        sample.mkdir(parents=True)
        (sample / "metadata.json").write_text(json.dumps({"object_path": f"objects/{name}/model.xml"}))
        (sample / "motion.npz").write_text("git-lfs pointer")
        expected.append(f"data/unitree_g1/fixture/{name}/sample1/*.npz")
    expected += [f"data/unitree_g1/egohtr/{name}/sample1/motion.npz" for name in ("ascent", "window")]
    assert lfs_include(motions, roster) == sorted(expected)


def test_native_setup_import_does_not_load_simulator_or_retargeting():
    subprocess.run([sys.executable, "-c", (
        "import sys; import orcs.cli.sync_native_data; "
        "assert 'mjlab' not in sys.modules; assert 'torch' not in sys.modules; "
        "assert 'orcs.ego_htr.umr' not in sys.modules"
    )], check=True)


def test_ordinary_setup_installs_shipped_pairs_without_new_mode():
    setup = (Path(__file__).parents[1] / "scripts/setup/sync_data.sh").read_text()
    assert "python -m orcs.cli.sync_native_data" in setup
    assert "all|inhouse|omre|grail)" in setup
