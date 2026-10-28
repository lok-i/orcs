"""Terrain-supported sole planning and optional accepted-palm references."""

import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np
import trimesh
from scipy.ndimage import binary_closing
from scipy.signal import savgol_filter
from trimesh.ray.ray_pyembree import RayMeshIntersector

LOCAL = np.array(
    [
        [-0.05, 0.025, -0.035],
        [-0.05, -0.025, -0.035],
        [0.12, 0.03, -0.035],
        [0.12, -0.03, -0.035],
        [0.14, 0, -0.035],
    ]
)
CENTER = np.array([0.04, 0, -0.035])


def unit(v):
    return v / np.maximum(np.linalg.norm(v, axis=-1, keepdims=True), 1e-9)


def override_blend(start, end, count, fade):
    """Smooth an override at interior boundaries while honoring clip edges."""
    if not 0 <= start < end <= count:
        raise ValueError("Target override outside clip")
    values = np.ones(end - start)
    fade = min(int(fade), len(values) // 2)
    if fade > 0 and start > 0:
        values[:fade] = np.sin(np.linspace(0, np.pi / 2, fade)) ** 2
    if fade > 0 and end < count:
        values[-fade:] = np.cos(np.linspace(0, np.pi / 2, fade)) ** 2
    return values


def ray_height(ray, points, expected, *, upward_only=False):
    origin = points.copy()
    origin[:, 2] = expected + 0.22
    loc, ids, faces = ray.intersects_location(
        origin, np.tile([0.0, 0.0, -1.0], (len(origin), 1)), multiple_hits=True
    )
    height = np.full(len(origin), np.nan)
    if upward_only and len(ids):
        keep = ray.mesh.face_normals[faces, 2] > 0.25
        loc, ids = loc[keep], ids[keep]
    # Choose the nearest layer around the anticipated sole. Proxy support
    # planning additionally rejects downward faces: proximity alone can choose
    # the underside of a thin slab and place a sole inside its collider.
    if len(ids):
        order = np.argsort(np.abs(loc[:, 2] - expected[ids]))
        first = np.unique(ids[order], return_index=True)[1]
        chosen = order[first]
        height[ids[chosen]] = loc[chosen, 2]
    return height


def build(source, terrain, scale, output, options=None):
    options = options or {}
    surface = options.get("support_surface", "scan")
    if surface not in ("scan", "proxy"):
        raise ValueError("support_surface must be 'scan' or 'proxy'")
    clearance = float(options.get("sole_clearance", 0.004))
    if not np.isfinite(clearance) or clearance < 0:
        raise ValueError("sole_clearance must be finite and nonnegative")
    source, terrain, output = Path(source), Path(terrain), Path(output)
    joints = np.load(source / "egohtr_joints.npy").astype(float) * scale
    n = len(joints)
    ankles = savgol_filter(joints[:, [7, 8]], 9, 2, axis=0)
    left = unit(savgol_filter(joints[:, 1] - joints[:, 2], 15, 2, axis=0) * [1, 1, 0])
    feet_forward = unit(
        savgol_filter(joints[:, [10, 11]] - joints[:, [7, 8]], 11, 2, axis=0)
        * [1, 1, 0]
    )
    centers = ankles + 0.04 * feet_forward
    centers[:, :, 2] -= 0.035
    # Widen the source's human-scale crossing steps to suit the G1 pelvis/feet.
    sep = np.sum((centers[:, 0] - centers[:, 1]) * left, axis=1)
    widening = np.maximum(0.15 - sep, 0) / 2
    centers[:, 0] += widening[:, None] * left
    centers[:, 1] -= widening[:, None] * left
    speed = np.linalg.norm(np.gradient(ankles, axis=0) * 30, axis=2)
    stance = speed < 0.55
    for side in range(2):
        stance[:, side] = binary_closing(stance[:, side], structure=np.ones(5))
    stance[:2] = speed[:2] < 0.55
    stance[-2:] = speed[-2:] < 0.55
    for override in options.get("stance_overrides", []):
        side = ["left", "right"].index(override["side"])
        start, end = int(override["start"]), int(override["end"])
        if not 0 <= start < end <= n:
            raise ValueError("Stance override outside clip")
        stance[start:end, side] = override.get("stance", True)
    # These are walking clips; extend the slower foot's support during noisy
    # transitions, unless both feet have substantial speed (a real swing/flight).
    for f in range(n):
        if not stance[f].any() and speed[f].min() < 0.85:
            stance[f, np.argmin(speed[f])] = True
    scan = trimesh.load(source / "terrain.obj", force="mesh")
    scan.apply_scale(scale)
    scan_ray = RayMeshIntersector(scan)
    tree = ET.parse(terrain)
    paths = [
        terrain.parent / a.attrib["file"]
        for a in tree.findall("./asset/mesh")
        if a.attrib["name"].startswith("terrain_hull_mesh_")
    ]
    proxy = trimesh.util.concatenate([trimesh.load(p, force="mesh") for p in paths])
    proxy.apply_scale(scale)
    proxy_ray = RayMeshIntersector(proxy)
    offsets = np.array(
        [
            (x, y, 0)
            for x in np.arange(-0.20, 0.201, 0.025)
            for y in np.arange(-0.20, 0.201, 0.025)
            if x * x + y * y <= 0.20**2
        ]
    )
    goals = np.zeros((n, 2, 5, 3))
    weights = np.full((n, 2), 0.03)
    segments = []
    for f in range(n):
        for side in range(2):
            x = feet_forward[f, side]
            y = np.cross([0, 0, 1], x)
            R = np.stack([x, y, [0, 0, 1]], axis=1)
            goals[f, side] = centers[f, side] + (LOCAL - CENTER) @ R.T
    for side in range(2):
        edges = np.diff(np.r_[False, stance[:, side], False].astype(int))
        starts = np.flatnonzero(edges == 1)
        ends = np.flatnonzero(edges == -1)
        for start, end in zip(starts, ends, strict=True):
            if end - start < 3:
                continue
            center = np.median(centers[start:end, side], axis=0)
            x = unit(np.median(feet_forward[start:end, side], axis=0))
            y = np.cross([0, 0, 1], x)
            xy = (LOCAL - CENTER)[:, :2] @ np.stack([x, y])
            candidates = center[None, :] + offsets
            points = candidates[:, None, :] + xy[None, :, :]
            flat = points.reshape(-1, 3)
            expect = np.repeat(candidates[:, 2], 5)
            heights = ray_height(scan_ray, flat, expect).reshape(-1, 5)
            proxy_heights = ray_height(
                proxy_ray, flat, expect, upward_only=surface == "proxy"
            ).reshape(-1, 5)
            support_heights = proxy_heights if surface == "proxy" else heights
            best = None
            A = np.c_[xy[:, :2], np.ones(5)]
            for i, hs in enumerate(support_heights):
                valid = np.isfinite(hs)
                if valid.sum() < 3:
                    continue
                a, b, c = np.linalg.lstsq(A[valid], hs[valid], rcond=None)[0]
                slope = np.hypot(a, b)
                if slope > 0.65:
                    continue
                plane = A @ np.array([a, b, c])
                residual = hs[valid] - plane[valid]
                # Clear the chosen support surface. In scan mode, also
                # discourage patches with substantial convex proxy overfill.
                raise_z = max(0.0, float(residual.max())) + clearance
                mismatch = (
                    np.nanmax(proxy_heights[i] - plane)
                    if np.isfinite(proxy_heights[i]).any()
                    else 0.2
                )
                score = (
                    np.sum(offsets[i, :2] ** 2)
                    + 2 * (c - center[2]) ** 2
                    + 12 * np.mean(residual**2)
                    + 4 * max(0.0, mismatch - 0.015) ** 2
                    + 0.003 * slope * slope
                    + 0.002 * (5 - valid.sum())
                )
                if best is None or score < best[0]:
                    best = (
                        score,
                        i,
                        a,
                        b,
                        c + raise_z,
                        float(mismatch),
                        int(valid.sum()),
                    )
            if best is None:
                segments.append(
                    dict(
                        side=side,
                        start=int(start),
                        end=int(end),
                        status="no usable local patch",
                    )
                )
                continue
            _, i, a, b, z, mismatch, hits = best
            normal = unit(np.array([-a, -b, 1.0]))
            x = unit(x - normal * np.dot(x, normal))
            y = np.cross(normal, x)
            R = np.stack([x, y, normal], axis=1)
            target_center = candidates[i].copy()
            target_center[2] = z
            target = target_center + (LOCAL - CENTER) @ R.T
            goals[start:end, side] = target
            weights[start:end, side] = 1.0
            # Blend the approach/departure over 5 samples, including target
            # orientation. The solver remains responsible for collisions.
            for k in range(1, 6):
                alpha = (6 - k) / 6
                for f in (start - k, end + k - 1):
                    if 0 <= f < n and not stance[f, side]:
                        goals[f, side] = (1 - alpha) * goals[f, side] + alpha * target
                        weights[f, side] = max(weights[f, side], 0.03 + 0.2 * alpha)
            segments.append(
                dict(
                    side=side,
                    start=int(start),
                    end=int(end),
                    target_center=target_center.tolist(),
                    normal=normal.tolist(),
                    xy_adjustment_m=float(
                        np.linalg.norm(target_center[:2] - center[:2])
                    ),
                    proxy_minus_plane_m=mismatch,
                    scan_hits=hits,
                )
            )
    for override in options.get("weight_boosts", []):
        side = ["left", "right"].index(override["side"])
        start, end = int(override["start"]), int(override["end"])
        if not 0 <= start < end <= n:
            raise ValueError("Weight boost outside clip")
        weights[start:end, side] *= (
            1.0
            + float(override["gain"]) * np.sin(np.linspace(0, np.pi, end - start)) ** 2
        )
    # Swing feet can inherit an exaggerated human toe-out angle even when the
    # step direction is forward. Rotate only the five-point target footprint,
    # preserving its centroid and the existing terrain/contact plan.
    for override in options.get("foot_heading_overrides", []):
        side = ["left", "right"].index(override["side"])
        start, end = int(override["start"]), int(override["end"])
        blend = override_blend(start, end, n, override.get("fade", 4))
        if override.get("direction", "forward") != "forward":
            raise ValueError("Foot heading direction must be 'forward'")
        for local_frame, frame in enumerate(range(start, end)):
            forward = unit(np.cross(left[frame], [0.0, 0.0, 1.0]))
            normal = np.array([0.0, 0.0, 1.0])
            if override.get("preserve_tilt", False):
                # Change heading within the existing sole plane. Flattening
                # a planted footprint here discards the fitted terrain slope.
                centered = goals[frame, side] - goals[frame, side].mean(axis=0)
                _, _, vh = np.linalg.svd(centered, full_matrices=False)
                normal = vh[-1]
                if normal[2] < 0:
                    normal = -normal
                forward = unit(forward - normal * np.dot(forward, normal))
            lateral = np.cross(normal, forward)
            rotation = np.stack([forward, lateral, normal], axis=1)
            centroid = goals[frame, side].mean(axis=0)
            desired = centroid + (LOCAL - LOCAL.mean(axis=0)) @ rotation.T
            alpha = blend[local_frame]
            goals[frame, side] = (1 - alpha) * goals[frame, side] + alpha * desired
            if "weight" in override:
                weights[frame, side] = max(
                    weights[frame, side],
                    (1 - alpha) * weights[frame, side]
                    + alpha * float(override["weight"]),
                )
    posture = options.get("joint_posture_overrides", [])
    posture_names = list(dict.fromkeys(o["joint"] for o in posture))
    posture_targets = np.zeros((n, len(posture_names)))
    posture_costs = np.zeros((n, len(posture_names)))
    for override in posture:
        joint = posture_names.index(override["joint"])
        start, end = int(override["start"]), int(override["end"])
        blend = override_blend(start, end, n, override.get("fade", 6))
        cost = float(override["cost"]) * blend
        target = float(override["target"])
        old = posture_costs[start:end, joint]
        total = old + cost
        active = total > 0
        combined = posture_targets[start:end, joint]
        combined[active] = (
            combined[active] * old[active] + target * cost[active]
        ) / total[active]
        posture_targets[start:end, joint] = combined
        posture_costs[start:end, joint] = total
    heading_cost_scale = np.ones(n)
    for override in options.get("heading_cost_overrides", []):
        start, end = int(override["start"]), int(override["end"])
        blend = override_blend(start, end, n, override.get("fade", 6))
        scale = float(override["scale"])
        heading_cost_scale[start:end] = np.maximum(
            heading_cost_scale[start:end], 1 + blend * (scale - 1)
        )
    np.savez_compressed(
        output,
        points=goals,
        weights=weights,
        left_axis=left,
        local_points=LOCAL,
        stance=weights > 0.99,
        joint_posture_names=np.asarray(posture_names, dtype=str),
        joint_posture_targets=posture_targets,
        joint_posture_costs=posture_costs,
        heading_cost_scale=heading_cost_scale,
    )
    return dict(
        segments=segments,
        source=f"Filtered source ankles/toes plus local {surface} plane fits",
        overrides=options,
    )


def add_hands(reference, feet, output, robot):
    import mujoco

    from .robot import load_motion

    model = mujoco.MjModel.from_xml_path(str(robot))
    data = mujoco.MjData(model)
    motion = load_motion(reference, model)
    with np.load(feet) as archive:
        targets = {key: archive[key].copy() for key in archive.files}
    if len(motion["qpos"]) != len(targets["points"]):
        raise ValueError("Hand reference frame count differs")
    local = np.array([0.07, 0.0, 0.0])
    ids = [model.body(side + "_wrist_yaw_link").id for side in ["left", "right"]]
    palms = []
    for pose in motion["qpos"]:
        data.qpos[:] = pose
        mujoco.mj_kinematics(model, data)
        palms.append([data.xpos[i] + data.xmat[i].reshape(3, 3) @ local for i in ids])
    targets["hand_points"] = np.array(palms)
    targets["hand_local_point"] = local
    np.savez_compressed(output, **targets)
