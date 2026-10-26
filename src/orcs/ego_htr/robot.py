"""Prepared robot assets and motion validation, independent of the UMR process."""

from __future__ import annotations

import shutil
import xml.etree.ElementTree as ET
from pathlib import Path

from .workspace import digest, key, write_json


def mesh_files(xml):
    xml = Path(xml).resolve()
    tree = ET.parse(xml)
    if tree.findall(".//include"):
        raise ValueError("Export a flattened MJCF before caching included models")
    compiler = tree.find("compiler")
    meshdir = compiler.get("meshdir", "") if compiler is not None else ""
    result = []
    for node in tree.findall("./asset/mesh"):
        if "file" in node.attrib:
            result.append(
                (node, (xml.parent / meshdir / node.attrib["file"]).resolve())
            )
    return tree, result


def cache_xml(ws, xml, kind="robot"):
    """Copy precisely the XML's referenced meshes; ignore stale collision hulls."""
    tree, references = mesh_files(xml)
    identity = dict(
        xml=digest(xml), meshes={n.get("name"): digest(p) for n, p in references}
    )
    folder = ws.root / "_cache" / kind / key(identity)
    if not (folder / "asset.json").exists():
        (folder / "meshes").mkdir(parents=True, exist_ok=True)
        for node, source in references:
            name = digest(source) + source.suffix
            dest = folder / "meshes" / name
            if not dest.exists():
                shutil.copy2(source, dest)
            node.set("file", "meshes/" + name)
        compiler = tree.find("compiler")
        if compiler is not None:
            compiler.attrib.pop("meshdir", None)
        tree.write(folder / "model.xml", encoding="unicode")
        write_json(
            folder / "asset.json",
            dict(identity=identity, xml_sha256=digest(folder / "model.xml")),
        )
    return folder / "model.xml"


def export_orcs(ws):
    import mjlab  # noqa: F401 -- complete plugin discovery first

    from orcs.assets.g1 import flat_hand_spec

    spec = flat_hand_spec()
    for mesh in spec.meshes:
        if mesh.file:
            mesh.file = str(
                (Path(spec.modelfiledir) / spec.compiler.meshdir / mesh.file).resolve()
            )
    spec.compiler.meshdir = ""
    target = ws.root / "_cache" / "orcs_export.xml"
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(spec.to_xml())
    try:
        return cache_xml(ws, target, "orcs-robot")
    finally:
        target.unlink(missing_ok=True)


def load_motion(path, model=None):
    import numpy as np

    with np.load(path, allow_pickle=False) as archive:
        result = {name: archive[name].copy() for name in archive.files}
    q = result["qpos"]
    if q.ndim != 2 or q.shape[1] != 36 or not len(q) or not np.isfinite(q).all():
        raise ValueError(
            "Expected a finite nonempty G1 trajectory with 36 qpos columns"
        )
    if not np.allclose(np.linalg.norm(q[:, 3:7], axis=1), 1, atol=1e-5):
        raise ValueError("Root quaternions are not normalized WXYZ values")
    fps = float(result["fps"].item())
    if not 0 < fps < 1000:
        raise ValueError("Invalid frame rate")
    scale = float(result["smpl_scale"].item())
    if not np.isfinite(scale) or scale <= 0:
        raise ValueError("Invalid human/scene scale")
    for name in ("frame_ids", "source_frames"):
        if (
            result[name].shape != (len(q),)
            or not np.issubdtype(result[name].dtype, np.integer)
            or (result[name] < 0).any()
            or (np.diff(result[name]) <= 0).any()
        ):
            raise ValueError(f"Invalid {name}")
    names = result["robot_joint_names"].tolist()
    if len(names) != 29 or len(set(names)) != 29:
        raise ValueError("Expected 29 unique G1 joint names")
    if model is not None:
        import mujoco

        expected = [
            model.joint(j).name
            for j in range(model.njnt)
            if model.jnt_type[j] == mujoco.mjtJoint.mjJNT_HINGE
        ]
        if names != expected or q.shape[1] != model.nq:
            raise ValueError("Motion and robot joint order differ")
    return result


def normalize_motion(source, target, start_frame, source_fps=30.0):
    """Read trusted UMR output, then remove object arrays and legacy path strings."""
    import numpy as np

    with np.load(source, allow_pickle=True) as old:
        fields = {k: old[k].copy() for k in ("qpos", "fps", "frame_ids", "smpl_scale")}
        fields["robot_joint_names"] = old["robot_joint_names"].astype(str)
    fields["source_frames"] = start_frame + fields["frame_ids"]
    fields["source_fps"] = np.array(source_fps)
    np.savez_compressed(target, **fields)
    return load_motion(target)


def model_for(ws, run):
    import mujoco

    return mujoco.MjModel.from_xml_path(str(ws.asset(run, "robot")))


def scene(ws, run, collision=False, native=False):
    """Build the same robot/scan/proxy view used by geometric audits and reports."""
    import mujoco
    import numpy as np
    import trimesh

    motion = load_motion(ws.run_dir(run) / "motion.npz")
    scale = float(motion["smpl_scale"].item())
    spec = mujoco.MjSpec.from_file(
        str(ws.asset(run, "orcs_robot" if native else "robot"))
    )
    spec.visual.global_.offwidth = 960
    spec.visual.global_.offheight = 720
    spec.visual.headlight.ambient[:] = (0.48, 0.48, 0.48)
    spec.visual.headlight.diffuse[:] = (0.7, 0.7, 0.7)
    spec.visual.headlight.specular[:] = (0.1, 0.1, 0.1)
    scan = trimesh.load(ws.asset(run, "scan"), force="mesh", process=False)
    vertices = np.asarray(scan.vertices) * scale
    spec.add_mesh(
        name="terrain_preview",
        uservert=vertices.ravel(),
        userface=scan.faces.ravel(),
        maxhullvert=16,
        inertia=mujoco.mjtMeshInertia.mjMESH_INERTIA_SHELL,
    )
    spec.worldbody.add_geom(
        name="terrain_preview",
        type=mujoco.mjtGeom.mjGEOM_MESH,
        meshname="terrain_preview",
        contype=0,
        conaffinity=0,
        rgba=(0.47, 0.51, 0.48, 1),
        group=2,
    )
    if collision:
        for i, (node, path) in enumerate(terrain_meshes(ws.asset(run, "terrain"))):
            mesh_scale = scale * np.fromstring(node.get("scale", "1 1 1"), sep=" ")
            spec.add_mesh(name=f"proxy_{i}", file=str(path), scale=mesh_scale)
            spec.worldbody.add_geom(
                name=f"proxy_{i}",
                type=mujoco.mjtGeom.mjGEOM_MESH,
                meshname=f"proxy_{i}",
                contype=2,
                conaffinity=1,
                rgba=(0.25, 0.55, 0.75, 1),
                group=3,
            )
    model = spec.compile()
    if collision:
        for g in range(model.ngeom):
            if model.geom_bodyid[g] and model.geom_contype[g]:
                model.geom_contype[g] = 1
                model.geom_conaffinity[g] = 2
    load_motion(ws.run_dir(run) / "motion.npz", model)
    return model, mujoco.MjData(model), scan, scale


def terrain_meshes(xml):
    return [
        (n, p)
        for n, p in mesh_files(xml)[1]
        if n.get("name", "").startswith("terrain_hull_mesh_")
    ]
