"""Reviewed EgoHTR runs → PerLoco source records, without UMR or simulation.

Each selected crop owns one fixed terrain tile and one explicitly pinned motion.
Labels (03, ...) are conveniences at export; full clip IDs identify staged tiles.
"""

from __future__ import annotations

import xml.etree.ElementTree as ET

import numpy as np

from orcs.core.paths import REPO_ROOT
from orcs.ego_htr.robot import load_motion, terrain_meshes
from orcs.ego_htr.workspace import Workspace, digest
from orcs.tasks.perloco.terrain_spec import ClipSpec, MeshSpec, TileSpec


class EgoHtrSource:
    name = "egohtr"
    default_root = REPO_ROOT / "logs/ego_htr"

    def __init__(self, root, families=None, levels=None, *, run="accepted"):
        if levels is not None and tuple(levels) != (0.0,):
            raise ValueError("EgoHTR crops have one level: 0.0")
        ws = Workspace(root)
        names = families if families is not None else [c["id"] for c in ws.clips()]
        if not names:
            raise FileNotFoundError(f"No reviewed EgoHTR clips in {root}")
        self._tiles: list[TileSpec] = []
        self._clips: list[ClipSpec] = []
        selected = set()
        for name in names:
            clip = ws.clip(name)
            ident = clip["id"]
            if ident in selected:
                raise ValueError(f"Duplicate EgoHTR selection: {ident}")
            selected.add(ident)
            record = ws.run(ident, run)
            if record["status"] not in ("ready", "imported"):
                raise ValueError(f"{ident}/{record['id']} is not a completed motion")
            motion_file = ws.run_dir(record) / "motion.npz"
            motion_sha = digest(motion_file)
            if motion_sha != record["motion_sha256"]:
                raise ValueError(f"Reviewed motion changed: {motion_file}")
            motion = load_motion(motion_file)
            fps = float(motion["fps"].item())
            if fps != 30 or not np.all(np.diff(motion["source_frames"]) == 1):
                raise ValueError("EgoHTR export requires contiguous 30 Hz frames")
            if not np.array_equal(motion["source_frames"],
                                  record["source"]["start_frame"] + motion["frame_ids"]):
                raise ValueError("Motion source-frame mapping differs from its reviewed run")
            if len(motion["qpos"]) < 3:
                raise ValueError("At least three motion frames are required")
            scale = float(motion["smpl_scale"].item())
            terrain_file = ws.asset(record, "terrain")
            # Prepared proxies use vertices in the source crop frame. Refuse
            # transforms that this bridge would otherwise silently discard.
            tree = ET.parse(terrain_file)
            for node in tree.findall(".//body") + tree.findall(".//geom"):
                for field in ("pos", "quat", "euler", "axisangle", "xyaxes", "zaxis"):
                    if field in node.attrib:
                        raise ValueError(f"Unexpected prepared terrain transform: {field}")
            pieces = terrain_meshes(terrain_file)
            if not pieces:
                raise ValueError(f"No convex terrain pieces in {terrain_file}")
            meshes, vertices, hashes = [], [], []
            for node, path in pieces:
                if not path.is_relative_to(ws.root):
                    raise ValueError("Terrain mesh escapes the review workspace")
                mesh_scale = scale * np.fromstring(node.get("scale", "1 1 1"), sep=" ")
                if mesh_scale.shape != (3,) or not np.isfinite(mesh_scale).all() \
                        or np.any(mesh_scale <= 0):
                    raise ValueError("Invalid prepared terrain scale")
                if path.suffix.lower() != ".obj":
                    raise ValueError("Prepared EgoHTR collision pieces must be OBJ files")
                with path.open() as stream:
                    verts = np.array([list(map(float, line.split()[1:4]))
                                      for line in stream if line.startswith("v ")])
                if verts.ndim != 2 or verts.shape[1] != 3 or len(verts) < 4 \
                        or not np.isfinite(verts).all():
                    raise ValueError(f"Invalid collision vertices in {path}")
                vertices.append(verts * mesh_scale)
                meshes.append(MeshSpec(path, tuple(mesh_scale)))
                hashes.append(digest(path))
            verts = np.concatenate(vertices)
            q = motion["qpos"]
            # A yaw-aligned 1.6 x 1 m sensor can extend ~0.95 m from the root.
            half = np.maximum(np.abs(verts[:, :2]).max(axis=0) + 0.3,
                              np.abs(q[:, :2]).max(axis=0) + 1.3)
            size = tuple(np.ceil(2 * half * 10) / 10)
            self._tiles.append(TileSpec(
                family=ident, level=0.0, meshes=tuple(meshes), size=size,
                floor_z=float(verts[:, 2].min() - 0.15),
            ))
            self._clips.append(ClipSpec(
                name=record["id"], root_pos=q[:, :3], root_quat=q[:, 3:7],
                joint_pos=q[:, 7:], joint_names=tuple(motion["robot_joint_names"]),
                fps=fps, family=ident, level=0.0,
                meta=dict(
                    source="egohtr", label=clip.get("label", ident),
                    review_run=record["id"], motion_sha256=motion_sha,
                    terrain_xml_sha256=digest(terrain_file),
                    terrain_mesh_sha256=hashes,
                    native_robot_sha256=digest(ws.asset(record, "orcs_robot")),
                    source_sequence=record["source"]["sequence"],
                    source_start_frame=int(motion["source_frames"][0]),
                    source_end_frame=int(motion["source_frames"][-1]) + 1,
                    source_fps=fps, scene_scale=scale,
                    coordinates="review world; scaled terrain, unmodified G1 root",
                ),
            ))

    def tiles(self):
        return iter(self._tiles)

    def clips(self):
        return iter(self._clips)
