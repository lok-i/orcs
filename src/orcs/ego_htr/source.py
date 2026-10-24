"""Local EgoHTR inspection and exact-matrix SMPL-X source preparation."""

from __future__ import annotations

import json
import tempfile
import zipfile
from importlib.resources import files
from pathlib import Path

from .workspace import digest, key, merge, read_json, safe_name, write_json


def inspect_source(ws, sequence):
    import numpy as np

    safe_name(sequence)
    directory = Path(ws.settings()["data_root"]) / "sequences" / sequence
    result = dict(sequence=sequence, files=[])
    for path in sorted(directory.glob("*")):
        if not path.is_file():
            continue
        record = dict(name=path.name, bytes=path.stat().st_size)
        if path.suffix == ".npz":
            members = []
            with zipfile.ZipFile(path) as archive:
                for member in archive.infolist():
                    item = dict(
                        name=member.filename,
                        compressed_bytes=member.compress_size,
                        bytes=member.file_size,
                    )
                    if member.filename.endswith(".npy"):
                        with archive.open(member) as stream:
                            version = np.lib.format.read_magic(stream)
                            shape, _, dtype = np.lib.format._read_array_header(
                                stream, version
                            )
                            item.update(
                                shape=list(shape),
                                dtype=str(dtype),
                                object_array=bool(dtype.hasobject),
                            )
                    members.append(item)
            record["members"] = members
        result["files"].append(record)
    if not result["files"]:
        raise FileNotFoundError(f"No local sequence at {directory}")
    return result


def recipe_for(path=None, preset=None):
    base = files("orcs.ego_htr").joinpath("presets")
    recipe = (
        read_json(path)
        if path
        else json.loads(base.joinpath(f"{safe_name(preset or '03')}.json").read_text())
    )
    common = json.loads(base.joinpath("base.json").read_text())
    recipe["config"] = merge(common, recipe.get("config", {}))
    if recipe.pop("review_assembly", False):
        recipe["config"].setdefault("solver", {})[
            "robot_self_collision_exclude_body_pairs"
        ] = json.loads(base.joinpath("assembly_exclusions.json").read_text())
    return recipe


def prepare(ws, path=None, preset=None, clip=None, revision="latest", stage="all"):
    from .robot import cache_xml, export_orcs, terrain_meshes
    from .umr import isolated_command, patch_status, run_process, runtime_config

    settings = ws.settings()
    if not settings["umr_python"]:
        raise ValueError("Configure --umr-python before preparing a source")
    if clip:
        parent = ws.run(clip, revision)
        recipe = parent.get(
            "recipe", dict(source=parent["source"], config=parent["config"])
        )
        run = ws.new_run(
            clip,
            f"prepare-{stage}",
            parent=parent["id"],
            **{k: parent[k] for k in ("source", "assets", "config", "preparation")},
            recipe=recipe,
            gender=recipe.get("gender", "male"),
        )
        run["assets"] = dict(run["assets"])
    else:
        recipe = recipe_for(path, preset)
        source = recipe["source"]
        safe_name(source["sequence"])
        fps = float(source.get("fps", 30))
        if fps != 30:
            raise ValueError("This EgoHTR bridge currently supports 30 Hz source data")
        source = dict(
            source,
            start_frame=int(
                source.get("start_frame", round(source.get("start", 0) * fps))
            ),
            end_frame=int(source.get("end_frame", round(source.get("end", 0) * fps))),
            fps=fps,
        )
        if not 0 <= source["start_frame"] < source["end_frame"]:
            raise ValueError("Invalid source interval")
        ident = recipe.get(
            "id",
            f"{source['sequence']}__{source['start_frame'] / fps:g}-{source['end_frame'] / fps:g}s",
        )
        ws.create_clip(
            ident, recipe.get("label", ident), source, recipe.get("title", ident)
        )
        robot = cache_xml(ws, Path(settings["umr_root"]) / "assets/g1/g1_29dof.xml")
        native = export_orcs(ws)
        run = ws.new_run(
            ident,
            f"prepare-{stage}",
            source=source,
            config=recipe["config"],
            recipe=recipe,
            preparation={},
            assets=dict(robot=ws.ref(robot), orcs_robot=ws.ref(native)),
        )
    directory = ws.run_dir(run)
    try:
        with tempfile.TemporaryDirectory(prefix="egohtr-prepare-") as temp:
            temp = Path(temp)
            if not clip or stage in ("source", "all"):
                source = run["source"]
                sequence = (
                    Path(settings["data_root"]) / "sequences" / source["sequence"]
                )
                inputs = dict(
                    optimized=Path(
                        recipe.get("optimized", sequence / "full_opt_sequence.npz")
                    ),
                    vertices=Path(
                        recipe.get("scene_vertices", sequence / "scene_vertices.npy")
                    ),
                    faces=Path(recipe.get("scene_faces", sequence / "scene_faces.npy")),
                )
                identity = dict(
                    inputs={k: digest(v) for k, v in inputs.items()},
                    source=source,
                    model=digest(settings["smplx_model"]),
                    gender=recipe.get("gender", "male"),
                    crop=recipe.get("crop", {}),
                    schema=1,
                )
                target = ws.root / "_cache/source" / key(identity)
                if not (target / "asset.json").exists():
                    write_json(
                        temp / "request.json",
                        dict(
                            kind="source",
                            output=str(target),
                            inputs={k: str(v.resolve()) for k, v in inputs.items()},
                            model=settings["smplx_model"],
                            source=source,
                            recipe=recipe,
                            identity=identity,
                        ),
                    )
                    run_process(
                        isolated_command(
                            settings["umr_python"],
                            "_prepare",
                            "--request",
                            temp / "request.json",
                        ),
                        directory / "run.log",
                    )
                meta = read_json(target / "asset.json")
                run["preparation"] = meta["preparation"]
                run["assets"].update(
                    source=ws.ref(target),
                    scan=ws.ref(target / "terrain.obj"),
                    motion_source=ws.ref(target / "motion.npz"),
                    terrain=ws.ref(cache_xml(ws, target / "terrain.xml", "terrain")),
                )
            if stage in ("collision", "all"):
                source = ws.asset(run, "source")
                options = recipe.get("collision", {"mode": "tiled"})
                identity = dict(
                    source=digest(source / "terrain.obj"),
                    joints=digest(source / "egohtr_joints.npy"),
                    options=options,
                )
                target = ws.root / "_cache/collision-build" / key(identity)
                if not (target / "asset.json").exists():
                    write_json(
                        temp / "request.json",
                        dict(
                            kind="collision",
                            source=str(source),
                            output=str(target),
                            options=options,
                        ),
                    )
                    run_process(
                        isolated_command(
                            settings["umr_python"],
                            "_prepare",
                            "--request",
                            temp / "request.json",
                        ),
                        directory / "run.log",
                    )
                run["assets"]["terrain"] = ws.ref(
                    cache_xml(ws, target / "terrain.xml", "terrain")
                )
                run["collision"] = read_json(target / "asset.json")
            if stage in ("build", "train", "all"):
                identity = dict(
                    source=run["assets"]["source"],
                    robot=run["assets"]["robot"],
                    model=digest(settings["smplx_model"]),
                    correspondence=run["config"].get("correspondence"),
                    template=run["config"].get("smpl_template"),
                    umr=patch_status(settings["umr_root"]),
                )
                target = ws.root / "_cache/correspondence" / key(identity)
                target.mkdir(parents=True, exist_ok=True)
                run["assets"]["correspondence"] = ws.ref(target)
                cfg = runtime_config(ws, run, temp)
                write_json(temp / "runtime.json", cfg)
                stages = ["build", "train"] if stage == "all" else [stage]
                for selected in stages:
                    if selected == "train" and not (target / "dataset.npz").exists():
                        raise ValueError("Run the correspondence build stage first")
                    run_process(
                        [
                            settings["umr_python"],
                            Path(settings["umr_root"])
                            / "scripts/humanoid_retarget_pipeline.py",
                            "--config",
                            temp / "runtime.json",
                            "--stage",
                            selected,
                            "--skip-view",
                        ],
                        directory / "run.log",
                    )
                write_json(
                    target / "asset.json",
                    dict(
                        identity=identity,
                        files={
                            str(p.relative_to(target)): digest(p)
                            for p in target.rglob("*")
                            if p.is_file() and p.name != "asset.json"
                        },
                    ),
                )
        run["summary"] = "Prepared inputs; run retarget with this revision as --parent."
        if stage in ("collision", "all") and not terrain_meshes(
            ws.asset(run, "terrain")
        ):
            raise ValueError("Empty collision proxy")
        return ws.finish(run, dict(passed=True, stage=stage), "prepared")
    except BaseException as error:
        ws.finish(run, dict(error=f"{type(error).__name__}: {error}"), "failed")
        raise


def prepare_worker(request):
    import csv

    import numpy as np

    record = read_json(request)
    output = Path(record["output"])
    output.mkdir(parents=True, exist_ok=True)
    if record["kind"] == "collision":
        from .terrain import decompose

        source = Path(record["source"])
        options = record["options"]
        stats = decompose(
            source / "terrain.obj",
            source / "egohtr_joints.npy",
            output,
            options.get("mode", "tiled"),
            options,
        )
        write_json(output / "asset.json", stats)
        return
    import smplx
    import torch
    import trimesh
    from scipy.spatial.transform import Rotation

    from .terrain import crop_mesh, scene_xml

    torch.set_num_threads(4)
    source = record["source"]
    recipe = record["recipe"]
    a, b = source["start_frame"], source["end_frame"]
    with np.load(record["inputs"]["optimized"]) as archive:
        if b > len(archive["smplx_egomc_transl"]):
            raise ValueError("Clip ends beyond source motion")
        roots = archive["smplx_egomc_global_orient"][a:b].copy()
        body = archive["smplx_egomc_body_pose"][a:b].copy()
        trans = archive["smplx_egomc_transl"][a:b].astype(np.float32)
        betas = archive["smplx_egomc_betas"][a].astype(np.float32)
        if np.ptp(archive["smplx_egomc_betas"][a:b], axis=0).max() > 1e-5:
            raise ValueError("Subject shape changes within clip")
        joints = archive["smplx_egomc_joints"][a:b, :22].copy()
        skeleton = (
            archive["smplx_skeleton"].copy() if "smplx_skeleton" in archive else None
        )
    v = np.load(record["inputs"]["vertices"], mmap_mode="r")
    f = np.load(record["inputs"]["faces"], mmap_mode="r")
    vertices, faces, lo, hi, original_faces = crop_mesh(
        v, f, joints, margin=recipe.get("crop", {}).get("margin", 1.2)
    )
    origin = vertices.astype(np.float64).mean(0)
    origin[2] = vertices[:, 2].min() - 0.15
    vertices = vertices - origin
    trans -= origin
    gender = recipe.get("gender", "male")
    layer = smplx.SMPLXLayer(
        record["model"], ext="npz", gender=gender, use_pca=False, flat_hand_mean=True
    )
    allv = []
    allj = []
    for k in range(0, b - a, 24):
        count = min(24, b - a - k)
        with torch.no_grad():
            res = layer(
                global_orient=torch.from_numpy(roots[k : k + count]),
                body_pose=torch.from_numpy(body[k : k + count]),
                transl=torch.from_numpy(trans[k : k + count]),
                betas=torch.from_numpy(np.repeat(betas[None], count, 0)),
            )
        allv.append(res.vertices.numpy())
        allj.append(res.joints.numpy())
    allv = np.concatenate(allv)
    allj = np.concatenate(allj)
    error = np.linalg.norm(allj[:, :22] + origin - joints, axis=-1)
    if error.max() >= 0.005:
        raise ValueError(f"Source alignment failed: {error.max():.6f} m")
    np.save(output / "egohtr_vertices.npy", allv)
    np.save(output / "egohtr_joints.npy", allj)
    np.save(output / "egohtr_faces.npy", layer.faces)
    if skeleton is not None:
        np.save(output / "skeleton.npy", skeleton)
    poses = np.zeros((b - a, 165), np.float32)
    poses[:, :3] = Rotation.from_matrix(roots).as_rotvec()
    poses[:, 3:66] = (
        Rotation.from_matrix(body.reshape(-1, 3, 3)).as_rotvec().reshape(b - a, 63)
    )
    # Axis-angle is only the upstream container. The tracked patch uses exact
    # matrix-derived geometry above for all source contact/surface objectives.
    np.savez_compressed(
        output / "motion.npz",
        poses=poses,
        pose_aa=poses,
        trans=trans,
        transl=trans,
        betas=betas,
        beta=betas,
        gender=gender,
        fps=np.array([30.0]),
        mocap_frame_rate=np.array([30.0]),
        source_data_dir="",
        source_dataset="egohtr",
        source_format="smplx_npz",
        source_sequence_key="egohtr",
        samp_output_up="z",
        human_scale=np.array([1.0]),
        human_scale_mode="off",
    )
    trimesh.Trimesh(vertices, faces, process=False).export(output / "terrain.obj")
    scene_xml(output, [])
    with (output / "prop_terrain.csv").open("w") as stream:
        writer = csv.writer(stream)
        writer.writerow(["px", "py", "pz", "qx", "qy", "qz", "qw"])
        writer.writerows([[0, 0, 0, 0, 0, 0, 1]] * (b - a))
    input_dir = Path(record["inputs"]["optimized"]).parent
    import shutil

    for path in input_dir.glob("egomc_footlock_*Foot.npy"):
        shutil.copy2(path, output / path.name)
    prep = dict(
        title=recipe.get("title", source["sequence"]),
        source=source["sequence"],
        start=a / 30,
        end=b / 30,
        start_frame=a,
        end_frame_exclusive=b,
        origin_source_m=origin.tolist(),
        azimuth=recipe.get("azimuth", 45.0),
        elevation=recipe.get("elevation", -30.0),
        alignment=dict(
            max_joint_error_m=float(error.max()), mean_joint_error_m=float(error.mean())
        ),
        preview_crop_bounds_source_m=[lo.tolist(), hi.tolist()],
        full_resolution_crop_faces=original_faces,
        preview_vertex_cluster_m=0.025,
        coordinate_rule="robot_world = (source_world - origin_source_m) * smpl_scale",
    )
    write_json(
        output / "asset.json", dict(identity=record["identity"], preparation=prep)
    )
