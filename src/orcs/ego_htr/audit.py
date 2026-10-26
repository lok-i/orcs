"""All-frame kinematic checks; no claims about controller stability."""

from __future__ import annotations

import tempfile
from pathlib import Path

from .robot import load_motion, model_for, scene, terrain_meshes
from .workspace import digest, read_json, write_json


def stats(values):
    import numpy as np

    a = np.asarray(values)
    a = a[np.isfinite(a)]
    return (
        dict(
            count=int(a.size),
            mean=float(a.mean()),
            p95=float(np.percentile(a, 95)),
            max=float(a.max()),
        )
        if a.size
        else dict(count=0)
    )


def audit(ws, run, surface=True, debug=False):
    import mujoco
    import numpy as np

    model = model_for(ws, run)
    if (
        run.get("motion_sha256")
        and digest(ws.run_dir(run) / "motion.npz") != run["motion_sha256"]
    ):
        raise ValueError("Motion changed after run completion")
    motion = load_motion(ws.run_dir(run) / "motion.npz", model)
    q = motion["qpos"]
    fps = float(motion["fps"].item())
    if float(run["source"]["fps"]) != 30.0:
        raise ValueError(
            "The current EgoHTR/UMR review constraints require 30 Hz source data"
        )
    if np.any(
        motion["source_frames"] != run["source"]["start_frame"] + motion["frame_ids"]
    ):
        raise ValueError("Source frame mapping differs from the clip")
    if motion["source_frames"][-1] >= run["source"]["end_frame"]:
        raise ValueError("Motion exceeds the requested source interval")
    selection = run["config"].get("motion", {})
    stop = selection.get("end", -1)
    count = run["source"]["end_frame"] - run["source"]["start_frame"]
    stop = count if stop < 0 else min(stop, count)
    stride = selection.get("stride", 1)
    expected = np.arange(selection.get("start", 0), stop, stride)
    limit = selection.get("max_frames", 0)
    if limit:
        expected = expected[:limit]
    if not np.array_equal(motion["frame_ids"], expected) or not np.isclose(
        fps, 30.0 / stride
    ):
        raise ValueError(
            "Output frames/rate do not match the requested source selection"
        )
    if "targets" in run["assets"] and stride != 1:
        raise ValueError(
            "Review velocity constraints currently require source stride 1"
        )
    data = mujoco.MjData(model)
    native = mujoco.MjModel.from_xml_path(str(ws.asset(run, "orcs_robot")))
    native_data = mujoco.MjData(native)
    addresses = [native.joint(str(n)).qposadr[0] for n in motion["robot_joint_names"]]
    collision, collision_data, scan, scale = scene(ws, run, collision=True)
    native_scene, native_scene_data, _, _ = scene(ws, run, collision=True, native=True)
    foot_ids = np.array(
        [
            [model.body(f"{s}_ankle_roll_sphere_{i}_link").id for i in range(1, 6)]
            for s in ("left", "right")
        ]
    )
    ankles = [model.body(f"{s}_ankle_roll_link").id for s in ("left", "right")]
    exclusions = {
        frozenset(x)
        for x in run["config"]
        .get("solver", {})
        .get("robot_self_collision_exclude_body_pairs", [])
    }
    self_depth = []
    umr_self = []
    terrain = []
    native_terrain = []
    native_worst = []
    separation = []
    points = []
    ankle_points = []
    worst = []
    palm_errors = []
    hand_targets = None
    if "targets" in run["assets"]:
        with np.load(ws.asset(run, "targets")) as archive:
            if "hand_points" in archive:
                hand_targets = archive["hand_points"][motion["frame_ids"]]
                hand_local = archive["hand_local_point"]
    for frame, pose in enumerate(q):
        data.qpos[:] = pose
        mujoco.mj_kinematics(model, data)
        mujoco.mj_comPos(model, data)
        mujoco.mj_collision(model, data)
        if hand_targets is not None:
            for side, prefix in enumerate(("left", "right")):
                body = model.body(f"{prefix}_wrist_yaw_link").id
                palm = data.xpos[body] + data.xmat[body].reshape(3, 3) @ hand_local
                palm_errors.append(np.linalg.norm(palm - hand_targets[frame, side]))
        lateral = data.xmat[model.body("pelvis").id].reshape(3, 3)[:, 1].copy()
        lateral[2] = 0
        lateral /= max(np.linalg.norm(lateral), 1e-9)
        separation.append(
            float((data.xpos[ankles[0]] - data.xpos[ankles[1]]) @ lateral)
        )
        probes = data.xpos[foot_ids].copy()
        probes[:, :, 2] -= 0.005
        points.append(probes)
        ankle_points.append(data.xpos[ankles].copy())
        contacts = []
        for contact in data.contact[: data.ncon]:
            a, b = [
                model.body(int(model.geom_bodyid[g])).name
                for g in (contact.geom1, contact.geom2)
            ]
            if a != "world" and b != "world" and frozenset((a, b)) not in exclusions:
                contacts.append(float(contact.dist))
        umr_self.append(max(0.0, -min(contacts, default=0.0)))
        native_data.qpos[:7] = pose[:7]
        native_data.qpos[addresses] = pose[7:]
        mujoco.mj_kinematics(native, native_data)
        mujoco.mj_comPos(native, native_data)
        mujoco.mj_collision(native, native_data)
        depths = [
            float(c.dist)
            for c in native_data.contact[: native_data.ncon]
            if native.geom_bodyid[c.geom1] > 0 and native.geom_bodyid[c.geom2] > 0
        ]
        self_depth.append(max(0.0, -min(depths, default=0.0)))
        native_scene_data.qpos[:7] = pose[:7]
        native_scene_data.qpos[addresses] = pose[7:]
        mujoco.mj_kinematics(native_scene, native_scene_data)
        mujoco.mj_comPos(native_scene, native_scene_data)
        mujoco.mj_collision(native_scene, native_scene_data)
        native_contacts = [
            c
            for c in native_scene_data.contact[: native_scene_data.ncon]
            if (native_scene.geom_bodyid[c.geom1] == 0)
            != (native_scene.geom_bodyid[c.geom2] == 0)
        ]
        native_contact = min(native_contacts, key=lambda c: c.dist, default=None)
        native_depth = (
            max(0.0, -float(native_contact.dist)) if native_contact is not None else 0.0
        )
        native_terrain.append(native_depth)
        if native_depth > 0.002:
            g = (
                native_contact.geom1
                if native_scene.geom_bodyid[native_contact.geom1]
                else native_contact.geom2
            )
            native_worst.append(
                dict(
                    frame=frame,
                    source_frame=int(motion["source_frames"][frame]),
                    depth_m=native_depth,
                    geom=native_scene.geom(g).name,
                )
            )
        collision_data.qpos[:] = pose
        mujoco.mj_kinematics(collision, collision_data)
        mujoco.mj_comPos(collision, collision_data)
        mujoco.mj_collision(collision, collision_data)
        contacts = [
            c
            for c in collision_data.contact[: collision_data.ncon]
            if (collision.geom_bodyid[c.geom1] == 0)
            != (collision.geom_bodyid[c.geom2] == 0)
        ]
        contact = min(contacts, key=lambda c: c.dist, default=None)
        depth = max(0.0, -float(contact.dist)) if contact is not None else 0.0
        terrain.append(depth)
        if depth > 0.002:
            g = contact.geom1 if collision.geom_bodyid[contact.geom1] else contact.geom2
            worst.append(
                dict(
                    frame=frame,
                    source_frame=int(motion["source_frames"][frame]),
                    depth_m=depth,
                    body=collision.body(int(collision.geom_bodyid[g])).name,
                )
            )
    velocities = np.empty((len(q) - 1, model.nv))
    for i in range(len(velocities)):
        mujoco.mj_differentiatePos(model, velocities[i], 1 / fps, q[i], q[i + 1])
    violations = []
    for joint in range(model.njnt):
        if (
            model.jnt_type[joint] == mujoco.mjtJoint.mjJNT_HINGE
            and model.jnt_limited[joint]
        ):
            values = q[:, model.jnt_qposadr[joint]]
            lo, hi = model.jnt_range[joint]
            override = (
                run["config"]
                .get("robot", {})
                .get("joint_limits", {})
                .get(model.joint(joint).name)
            )
            if override:
                lo, hi = max(lo, float(override[0])), min(hi, float(override[1]))
            violations.append(
                max(0.0, float(np.maximum(lo - values, values - hi).max()))
            )
    notes = ws.notes(run["clip"])
    source_to_output = {
        int(frame): i for i, frame in enumerate(motion["source_frames"])
    }
    frames = sorted(
        {
            source_to_output[n["source_frame"]]
            for n in notes
            if n["type"] == "comment" and n["source_frame"] in source_to_output
        }
    )
    names = motion["robot_joint_names"].tolist()
    root = {
        name: dict(
            component=stats(np.abs(velocities[:, sl])),
            norm=stats(np.linalg.norm(velocities[:, sl], axis=1)),
        )
        for name, sl in [
            ("translation_mps", slice(0, 3)),
            ("rotation_radps", slice(3, 6)),
        ]
    }
    checks = dict(
        motion_sha256=digest(ws.run_dir(run) / "motion.npz"),
        frames=len(q),
        fps=fps,
        scene_scale=scale,
        model_sha256=dict(
            umr=digest(ws.asset(run, "robot")), orcs=digest(ws.asset(run, "orcs_robot"))
        ),
        runtime=dict(mujoco=mujoco.__version__, numpy=np.__version__),
        geometry=dict(
            scan_vertices=len(scan.vertices),
            scan_faces=len(scan.faces),
            collision_pieces=len(terrain_meshes(ws.asset(run, "terrain"))),
            collision_mesh_bytes=sum(
                path.stat().st_size
                for _, path in terrain_meshes(ws.asset(run, "terrain"))
            ),
        ),
        hand_reference_error_m=stats(palm_errors),
        root_vs_scaled_source_m=stats(
            np.linalg.norm(
                q[:, :3]
                - np.load(ws.asset(run, "source") / "egohtr_joints.npy")[
                    motion["frame_ids"], 0
                ]
                * scale,
                axis=1,
            )
        ),
        joint_limit_max_violation_rad=max(violations, default=0.0),
        orcs_self_m=stats(self_depth),
        umr_self_m=stats(umr_self),
        terrain_penetration_m=stats(terrain),
        orcs_terrain_penetration_m=stats(native_terrain),
        worst_orcs_terrain_contacts=sorted(native_worst, key=lambda r: -r["depth_m"])[
            :15
        ],
        worst_terrain_contacts=sorted(worst, key=lambda r: -r["depth_m"])[:15],
        crossed_foot_frames=int((np.asarray(separation) < 0).sum()),
        min_foot_separation_m=float(min(separation)),
        joint_speed_radps=stats(np.abs(velocities[:, 6:])),
        root_velocity=root,
        joint_acceleration_radps2=stats(
            np.abs(np.diff(velocities[:, 6:], axis=0) * fps)
        ),
        joint_jerk_radps3=stats(
            np.abs(np.diff(velocities[:, 6:], n=2, axis=0) * fps**2)
        ),
        root_acceleration_mps2=stats(np.abs(np.diff(velocities[:, :3], axis=0) * fps)),
        knee_deg_percentiles=np.rad2deg(
            np.percentile(
                q[:, [7 + names.index(s + "_knee_joint") for s in ("left", "right")]],
                [0, 50, 100],
                axis=0,
            )
        ).tolist(),
        elbow_coordinate_deg_percentiles=np.rad2deg(
            np.percentile(
                q[:, [7 + names.index(s + "_elbow_joint") for s in ("left", "right")]],
                [0, 50, 100],
                axis=0,
            )
        ).tolist(),
        annotated={
            str(f): dict(
                orcs_self_m=self_depth[f],
                terrain_m=terrain[f],
                foot_separation_m=separation[f],
            )
            for f in frames
        },
        limitations=[
            "Kinematic only; no policy/dynamics or training-memory validation.",
            "Terrain distances use the convex proxy, which can fill open-scan holes.",
            "Stance annotations are sparse; ankle speed includes rotation, not pure slip.",
        ],
    )
    if surface:
        from .umr import isolated_command, run_process

        settings = ws.settings()
        if not settings["umr_python"]:
            raise ValueError(
                "Scan audit needs configured geometry interpreter (--umr-python)"
            )
        with tempfile.TemporaryDirectory(prefix="egohtr-audit-") as temp:
            temp = Path(temp)
            np.savez_compressed(temp / "probes.npz", points=points, ankles=ankle_points)
            write_json(
                temp / "request.json",
                dict(workspace=str(ws.root), run=run, frames=frames),
            )
            run_process(
                isolated_command(
                    settings["umr_python"],
                    "_surface-audit",
                    "--request",
                    temp / "request.json",
                    "--probes",
                    temp / "probes.npz",
                    "--output",
                    temp / "result.json",
                ),
                ws.run_dir(run) / "run.log",
            )
            checks["support"] = read_json(temp / "result.json")
    review = "targets" in run["assets"]
    gates = dict(joint_limits=checks["joint_limit_max_violation_rad"] < 1e-5)
    if review:
        solver = run["config"].get("solver", {})
        gates.update(
            native_self=checks["orcs_self_m"]["max"] < 0.002,
            crossing=checks["crossed_foot_frames"] == 0,
            joint_speed=checks["joint_speed_radps"].get("max", 0)
            <= solver.get("egohtr_review_max_joint_velocity", 10.0) + 0.001,
            root_translation=root["translation_mps"]["component"].get("max", 0)
            <= solver.get("egohtr_review_max_root_translation_velocity", 1.0) + 0.001,
            root_rotation=root["rotation_radps"]["component"].get("max", 0)
            <= solver.get("egohtr_review_max_root_rotation_velocity", 2.0) + 0.03,
        )
        if solver.get("egohtr_review_orcs_terrain_cost", 0.0) > 0:
            gates["native_terrain"] = (
                checks["orcs_terrain_penetration_m"]["max"] < 0.002
            )
    checks.update(
        gates=gates,
        passed=all(gates.values()),
        review_required=[
            "Terrain contacts, support gaps and smoothness are reported; they are not zero-error gates."
        ],
    )
    if debug:
        folder = ws.run_dir(run) / "debug"
        folder.mkdir(exist_ok=True)
        np.savez_compressed(
            folder / "audit.npz",
            orcs_self_m=self_depth,
            terrain_m=terrain,
            orcs_terrain_m=native_terrain,
            separation_m=separation,
            points=points,
            velocity=velocities,
        )
    return checks


def surface_audit(request, probes, output):
    import numpy as np
    import trimesh
    from trimesh.ray.ray_pyembree import RayMeshIntersector

    from .robot import terrain_meshes
    from .workspace import Workspace

    record = read_json(request)
    ws = Workspace(record["workspace"])
    run = record["run"]
    motion = load_motion(ws.run_dir(run) / "motion.npz")
    scale = float(motion["smpl_scale"].item())
    with np.load(probes) as p:
        points = p["points"]
        ankles = p["ankles"]
    scan = trimesh.load(ws.asset(run, "scan"), force="mesh", process=False)
    scan.apply_scale(scale)
    ray = RayMeshIntersector(scan)
    flat = points.reshape(-1, 3)
    origins = flat.copy()
    origins[:, 2] += 0.3
    loc, rays, _ = ray.intersects_location(
        origins, np.tile([0, 0, -1.0], (len(flat), 1)), multiple_hits=True
    )
    gap = np.full(len(flat), np.nan)
    if len(rays):
        order = np.argsort(np.abs(loc[:, 2] - flat[rays, 2]))
        first = np.unique(rays[order], return_index=True)[1]
        chosen = order[first]
        gap[rays[chosen]] = flat[rays[chosen], 2] - loc[chosen, 2]
    gap = gap.reshape(points.shape[:-1])
    dist = []
    for chunk in np.array_split(flat, 30):
        if len(chunk):
            dist.extend(trimesh.proximity.closest_point(scan, chunk)[1])
    distances = np.asarray(dist).reshape(len(points), 2, 5).min(axis=2)
    flags = []
    for side in ("Left", "Right"):
        path = ws.asset(run, "source") / f"egomc_footlock_{side}Foot.npy"
        flags.append(
            np.load(path).reshape(-1)[motion["source_frames"]].astype(bool)
            if path.exists()
            else np.zeros(len(points), bool)
        )
    flags = np.asarray(flags).T
    speed = np.linalg.norm(np.diff(ankles, axis=0), axis=2) * float(
        motion["fps"].item()
    )
    result = dict(
        annotated_stance_foot_scan_distance_m=stats(distances[flags]),
        annotated_stance_ankle_speed_mps=stats(speed[flags[:-1] & flags[1:]]),
        either_foot_scan_distance_m=stats(distances.min(axis=1)),
        annotated={},
    )
    targets = None
    if "targets" in run["assets"]:
        with np.load(ws.asset(run, "targets")) as n:
            targets = {k: n[k][motion["frame_ids"]] for k in ("points", "stance")}
        errors = np.linalg.norm(points - targets["points"], axis=3)
        result["stance_goal_error_m"] = stats(errors[targets["stance"]])
    for f in record["frames"]:
        result["annotated"][str(f)] = {}
        for side, name in enumerate(("left", "right")):
            g = gap[f, side]
            valid = np.isfinite(g)
            result["annotated"][str(f)][name] = dict(
                planned_stance=bool(targets["stance"][f, side])
                if targets is not None
                else None,
                vertical_scan_gap_m=[float(x) if np.isfinite(x) else None for x in g],
                closest_scan_gap_m=float(np.nanmin(np.abs(g))) if valid.any() else None,
                near_scan_probes=int((np.abs(g) < 0.03).sum()),
            )
    human = (
        np.load(ws.asset(run, "source") / "egohtr_joints.npy")[motion["frame_ids"]][
            :, [10, 11]
        ].reshape(-1, 3)
        * scale
    )
    origins = human.copy()
    origins[:, 2] += 0.35
    hulls = []
    for node, path in terrain_meshes(ws.asset(run, "terrain")):
        hull = trimesh.load(path, force="mesh")
        hull.apply_scale(scale * np.fromstring(node.get("scale", "1 1 1"), sep=" "))
        hulls.append(hull)
    proxy = trimesh.util.concatenate(hulls)

    def heights(mesh):
        loc, ids, _ = RayMeshIntersector(mesh).intersects_location(
            origins, np.tile([0, 0, -1.0], (len(human), 1)), multiple_hits=True
        )
        values = np.full(len(human), -np.inf)
        np.maximum.at(values, ids, loc[:, 2])
        values[~np.isfinite(values)] = np.nan
        return values

    a, b = heights(scan), heights(proxy)
    valid = np.isfinite(a) & np.isfinite(b)
    result["proxy_minus_scan_height_m"] = stats(b[valid] - a[valid])
    result["caveat"] = (
        "Unsigned nearest scan distances and nearest-layer vertical ray gaps do not establish support forces."
    )
    write_json(output, result)
