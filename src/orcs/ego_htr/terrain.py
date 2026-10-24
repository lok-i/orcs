"""Whole-body crops and bounded whole/tiled CoACD collision proxies."""

import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np


def clustered_mesh(v, f, spacing=0.025):
    cells = np.floor(v / spacing).astype(np.int32)
    _, inv = np.unique(cells, axis=0, return_inverse=True)
    counts = np.bincount(inv)
    newv = np.column_stack(
        [np.bincount(inv, weights=v[:, i]) / counts for i in range(3)]
    ).astype(np.float32)
    newf = inv[f].astype(np.int32)
    keep = (
        (newf[:, 0] != newf[:, 1])
        & (newf[:, 0] != newf[:, 2])
        & (newf[:, 1] != newf[:, 2])
    )
    newf = newf[keep]
    _, idx = np.unique(np.sort(newf, axis=1), axis=0, return_index=True)
    return newv, newf[idx]


def crop_mesh(v, f, j, margin=1.2):
    # Entire body sweep, not only the root. Preview-only simplification follows.
    lo = j[:, :22].min((0, 1))
    hi = j[:, :22].max((0, 1))
    lo[:2] -= margin
    hi[:2] += margin
    lo[2] -= 1.2
    hi[2] += 0.35
    mask = np.all((v >= lo) & (v <= hi), axis=1)
    keep = np.all(mask[f], axis=1)
    faces = f[keep]
    used, inv = np.unique(faces, return_inverse=True)
    if not len(faces):
        raise ValueError("The selected body sweep does not intersect the scene scan")
    verts = v[used]
    faces = inv.reshape(-1, 3).astype(np.int32)
    before = len(faces)
    verts, faces = clustered_mesh(verts, faces)
    return verts, faces, lo, hi, before


def scene_xml(directory, parts):
    directory = Path(directory)
    root = ET.Element("mujoco", model="egohtr_terrain")
    ET.SubElement(root, "compiler", angle="radian")
    asset = ET.SubElement(root, "asset")
    ET.SubElement(asset, "mesh", name="terrain_visual_mesh", file="terrain.obj")
    body = ET.SubElement(
        ET.SubElement(root, "worldbody"), "body", name="egohtr_terrain"
    )
    ET.SubElement(body, "freejoint", name="terrain_free")
    ET.SubElement(
        body,
        "geom",
        name="terrain_visual",
        type="mesh",
        mesh="terrain_visual_mesh",
        group="2",
        contype="0",
        conaffinity="0",
        rgba=".48 .52 .49 1",
        mass="0",
    )
    for i, path in enumerate(parts):
        name = f"terrain_hull_mesh_{i}"
        ET.SubElement(
            asset, "mesh", name=name, file=str(Path(path).relative_to(directory))
        )
        ET.SubElement(
            body,
            "geom",
            name=f"terrain_hull_{i}",
            type="mesh",
            mesh=name,
            group="3",
            contype="1",
            conaffinity="1",
            rgba=".2 .4 .8 .2",
            mass="1",
        )
    ET.ElementTree(root).write(directory / "terrain.xml", encoding="unicode")


def tile_worker(task):
    import coacd
    import trimesh

    i, key, verts, faces, settings, partdir = task
    coacd.set_log_level("warn")
    parts = coacd.run_coacd(coacd.Mesh(verts, faces), **settings)
    paths = []
    for j, (v, f) in enumerate(parts):
        path = partdir / f"hull_{i:03}_{j:03}.obj"
        trimesh.Trimesh(v, f, process=False).export(path)
        paths.append(path)
    return paths, dict(tile=key.tolist(), input_faces=len(faces), parts=len(parts))


def decompose(scan, joints, directory, mode="tiled", overrides=None):
    import multiprocessing
    import shutil
    from concurrent.futures import ProcessPoolExecutor, as_completed

    import coacd
    import trimesh

    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    shutil.copy2(scan, directory / "terrain.obj")
    mesh = trimesh.load(scan, force="mesh", process=True)
    partdir = directory / "collision"
    partdir.mkdir(exist_ok=True)
    coacd.set_log_level("warn")
    overrides = overrides or {}
    paths = []
    counts = []
    if mode == "whole":
        settings = dict(
            threshold=0.03,
            max_convex_hull=128,
            preprocess_mode="on",
            preprocess_resolution=60,
            resolution=2000,
            mcts_nodes=20,
            mcts_iterations=100,
            mcts_max_depth=3,
            merge=True,
            decimate=True,
            max_ch_vertex=64,
            seed=0,
        )
        settings.update(overrides.get("coacd", {}))
        parts = coacd.run_coacd(
            coacd.Mesh(
                np.asarray(mesh.vertices, np.float64), np.asarray(mesh.faces, np.int32)
            ),
            **settings,
        )
        for i, (v, f) in enumerate(parts):
            path = partdir / f"hull_{i:03}.obj"
            trimesh.Trimesh(v, f, process=False).export(path)
            paths.append(path)
    elif mode == "tiled":
        size = float(overrides.get("tile_size", 1.2))
        corridor = float(overrides.get("corridor", 0.3))
        if size <= 0 or corridor < 0:
            raise ValueError("Invalid tile size/corridor")
        cells = np.floor(mesh.triangles_center[:, :2] / size).astype(int)
        keys, inverse = np.unique(cells, axis=0, return_inverse=True)
        sweep = np.load(joints)[:, :22, :2].reshape(-1, 2)
        settings = dict(
            threshold=0.025,
            max_convex_hull=24,
            preprocess_mode="on",
            preprocess_resolution=80,
            resolution=1000,
            mcts_nodes=12,
            mcts_iterations=40,
            mcts_max_depth=3,
            merge=True,
            decimate=True,
            max_ch_vertex=32,
            seed=0,
        )
        settings.update(overrides.get("coacd", {}))
        tasks = []
        for i, k in enumerate(keys):
            lo = k * size
            hi = lo + size
            if (
                np.linalg.norm(
                    np.maximum(np.maximum(lo - sweep, sweep - hi), 0.0), axis=1
                ).min()
                > corridor
            ):
                continue
            faces = mesh.faces[inverse == i]
            if len(faces) < 20:
                continue
            ids, faces = np.unique(faces, return_inverse=True)
            verts = mesh.vertices[ids]
            if np.linalg.matrix_rank(verts - verts.mean(0)) < 3:
                continue
            tasks.append(
                (
                    i,
                    k,
                    np.asarray(verts, np.float64),
                    faces.reshape(-1, 3).astype(np.int32),
                    settings,
                    partdir,
                )
            )
        with ProcessPoolExecutor(
            max_workers=int(overrides.get("workers", 4)),
            mp_context=multiprocessing.get_context("spawn"),
        ) as pool:
            for future in as_completed([pool.submit(tile_worker, t) for t in tasks]):
                newpaths, record = future.result()
                paths.extend(newpaths)
                counts.append(record)
    else:
        raise ValueError("Collision mode must be whole or tiled")
    if not paths:
        raise ValueError("No usable collision pieces")
    paths.sort()
    scene_xml(directory, paths)
    return dict(
        method=mode,
        settings=settings,
        parts=len(paths),
        bytes=sum(p.stat().st_size for p in paths),
        tiles=counts,
        caution="Open-scan approximation; inspect holes and tile boundaries.",
    )
