"""Reviewed-reference export and movable mesh-tile contracts; no GPU required."""

from __future__ import annotations

import json
import shutil
from pathlib import Path

import mujoco
import numpy as np
import pytest

from orcs.cli.stage_terrain_motions import _write_tile
from orcs.ego_htr.workspace import Workspace, digest, write_json
from orcs.tasks.perloco.env_cfg import egohtr_env_cfg
from orcs.tasks.perloco.sources.egohtr import EgoHtrSource
from orcs.tasks.perloco.terrain import TileTerrainCfg


def reviewed(ws: Workspace, label: str):
    ident = f"sequence_{label}__0-1s"
    source = dict(sequence=f"sequence_{label}", start_frame=30, end_frame=36, fps=30.0)
    ws.create_clip(ident, label, source)
    asset = ws.root / "_cache" / label
    asset.mkdir(parents=True)
    mesh = asset / "tetra.obj"
    mesh.write_text(
        "v 0 0 0\nv 1 0 0\nv 0 1 0\nv 0 0 1\n"
        "f 1 3 2\nf 1 2 4\nf 1 4 3\nf 2 3 4\n"
    )
    terrain = asset / "terrain.xml"
    terrain.write_text(
        '<mujoco><asset><mesh name="terrain_hull_mesh_0" file="tetra.obj"/>'
        '</asset><worldbody><body name="terrain">'
        '<geom type="mesh" mesh="terrain_hull_mesh_0"/>'
        '</body></worldbody></mujoco>'
    )
    robot = asset / "robot.xml"
    robot.write_text("<mujoco/>")
    run = ws.new_run(ident, "reviewed", source=source,
                     assets=dict(terrain=ws.ref(terrain), orcs_robot=ws.ref(robot)))
    q = np.zeros((6, 36), dtype=np.float32)
    q[:, 0] = np.linspace(0.1, 0.3, 6)
    q[:, 2] = 0.8
    q[:, 3] = 1
    q[:, 7:] = np.arange(29) / 100
    path = ws.run_dir(run) / "motion.npz"
    np.savez(path, qpos=q, fps=np.array([30.0]), smpl_scale=np.array([0.5]),
             frame_ids=np.arange(6), source_frames=np.arange(30, 36),
             robot_joint_names=np.array([f"joint_{i}" for i in range(29)]))
    run.update(status="ready", motion_sha256=digest(path))
    write_json(ws.run_dir(run) / "run.json", run)
    clip = ws.clip(ident)
    clip.update(accepted=run["id"], latest=run["id"])
    write_json(ws.clip_dir(ident) / "clip.json", clip)
    return ident, q, run


def test_selection_pins_runs_and_scales_terrain_only_once(tmp_path):
    ws = Workspace(tmp_path / "review")
    a, q, run = reviewed(ws, "03")
    b, _, _ = reviewed(ws, "05")
    source = EgoHtrSource(ws.root, families=["03", b])
    clips, tiles = list(source.clips()), list(source.tiles())
    assert [c.family for c in clips] == [a, b]
    assert [t.key for t in tiles] == [c.tile_key for c in clips]
    np.testing.assert_array_equal(clips[0].root_pos, q[:, :3])
    np.testing.assert_array_equal(clips[0].joint_pos, q[:, 7:])
    assert tiles[0].meshes[0].scale == (0.5, 0.5, 0.5)
    assert tiles[0].floor_z < 0
    assert clips[0].meta["review_run"] == run["id"]
    assert clips[0].meta["motion_sha256"] == run["motion_sha256"]
    # Later review-pointer changes cannot rebind an already selected source.
    metadata = ws.clip(a)
    metadata["accepted"] = "different"
    write_json(ws.clip_dir(a) / "clip.json", metadata)
    assert list(source.clips())[0].name == run["id"]


def test_source_rejects_changed_reference_duplicate_and_unknown_clip(tmp_path):
    ws = Workspace(tmp_path)
    ident, _, run = reviewed(ws, "03")
    with pytest.raises(ValueError, match="Duplicate"):
        EgoHtrSource(ws.root, families=["03", ident])
    with pytest.raises(ValueError, match="expected one match"):
        EgoHtrSource(ws.root, families=["missing"])
    path = ws.run_dir(run) / "motion.npz"
    with path.open("ab") as stream:
        stream.write(b"changed")
    with pytest.raises(ValueError, match="Reviewed motion changed"):
        EgoHtrSource(ws.root, families=["03"])


def test_staged_meshes_survive_workspace_removal_and_keep_tile_frame(tmp_path):
    ws = Workspace(tmp_path / "review")
    ident, _, _ = reviewed(ws, "03")
    tile = next(EgoHtrSource(ws.root).tiles())
    root = tmp_path / "data"
    _write_tile(root, tile)
    staged = json.loads((root / tile.key / "tile.json").read_text())
    mesh = staged["meshes"][0]
    assert not Path(mesh["file"]).is_absolute()
    assert digest(root / tile.key / mesh["file"]) == mesh["sha256"]
    moved = tmp_path / "moved"
    root.rename(moved)
    shutil.rmtree(ws.root)
    spec = mujoco.MjSpec()
    spec.worldbody.add_body(name="terrain")
    cfg = TileTerrainCfg(root=moved, family=ident, levels=(0.0,), size=tile.size)
    output = cfg.function(0, spec, np.random.default_rng(0))
    model = spec.compile()
    assert model.nmesh == 1 and model.ngeom == 2
    assert model.nq == 0  # fixed terrain: no free joint copied from the UMR XML
    np.testing.assert_allclose(output.origin, [tile.size[0]/2, tile.size[1]/2, 0])
    # A ray above the tetrahedron hits z=.25 in the SAME tile-local frame.
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    distance = mujoco.mj_ray(
        model, data, output.origin + [0.125, 0.125, 1.0],
        np.array([0., 0., -1.]), np.array([1, 0, 0, 0, 0, 0], dtype=np.uint8),
        1, -1, np.zeros(1, dtype=np.int32),
    )
    assert distance == pytest.approx(0.75, abs=1e-6)


def test_builder_refuses_mesh_path_escape_and_undersized_tile(tmp_path):
    ws = Workspace(tmp_path / "review")
    ident, _, _ = reviewed(ws, "03")
    tile = next(EgoHtrSource(ws.root).tiles())
    root = tmp_path / "data"
    _write_tile(root, tile)
    spec = mujoco.MjSpec()
    spec.worldbody.add_body(name="terrain")
    cfg = TileTerrainCfg(root=root, family=ident, levels=(0.0,), size=(0.1, 0.1))
    with pytest.raises(ValueError, match="requires size"):
        cfg.function(0, spec, np.random.default_rng(0))
    path = root / tile.key / "tile.json"
    data = json.loads(path.read_text())
    data["meshes"][0]["file"] = "../../../escape.obj"
    write_json(path, data)
    cfg.size = tile.size
    with pytest.raises(ValueError, match="escapes"):
        cfg.function(0, spec, np.random.default_rng(0))


def test_multi_crop_factory_pairs_one_roster_without_review_dependencies(tmp_path):
    ws = Workspace(tmp_path / "review")
    a, _, _ = reviewed(ws, "03")
    b, _, _ = reviewed(ws, "05")
    root = tmp_path / "staged"
    for tile in EgoHtrSource(ws.root).tiles():
        _write_tile(root, tile)
        folder = root / tile.key / "sample0"
        folder.mkdir()
        np.savez(folder / "motion.npz", joint_pos=np.zeros((50, 29)), fps=[50])
    roster = tmp_path / "both.toml"
    roster.write_text(f'families = ["{a}", "{b}"]\nlevels = [0.0]\n')
    shutil.rmtree(ws.root)
    cfg = egohtr_env_cfg(roster=str(roster), dataset_root=str(root))
    command = cfg.commands["motion"]
    terrain = cfg.scene.terrain.terrain_generator
    assert command.tile_keys == (f"{a}/level_0.00", f"{b}/level_0.00")
    assert tuple(terrain.sub_terrains) == (a, b)
    assert terrain.num_rows == command.n_rows == 1
    assert all(t.render_z_scale is None for t in terrain.sub_terrains.values())
    assert command.dataset_dir == str(root)
    assert set(cfg.observations) == {"policy", "tokenizer", "augmentation", "critic"}
    assert cfg.sim.mujoco.nccdmax == 128
    assert "play_foot_friction" not in cfg.events
    assert cfg.events["rand_robot_friction"].params["ranges"] == (0.3, 1.6)
    play = egohtr_env_cfg(play=True, roster=str(roster), dataset_root=str(root))
    assert play.sim.mujoco.nccdmax == 128
    assert "rand_robot_friction" not in play.events
    friction = play.events["play_foot_friction"]
    assert friction.mode == "startup"
    assert friction.params["ranges"] == (1.0, 1.0)
    assert friction.params["axes"] == [0]
