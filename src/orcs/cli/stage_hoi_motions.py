"""Fetch, stage and discard HOI-Retarget source data; runtime reads native NPZs.

The packaged roster pins both source files and the exact motion selection.
Downloads live in an owned temporary directory, including on failed conversion.
An explicit --source-parquet/--source-mesh is borrowed and never deleted.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import tempfile
import urllib.request
from pathlib import Path

import numpy as np

FPS = 50.0
CONTACT_EPS = 0.02


def digest(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def _verified(path: Path, sha256: str, size: int | None = None) -> Path:
    if not path.is_file():
        raise FileNotFoundError(path)
    if (size is not None and path.stat().st_size != size) or digest(path) != sha256:
        raise ValueError(f"Source checksum/size mismatch: {path}")
    return path


def _download(source: dict, key: str, destination: Path) -> Path:
    relative = source[key]
    url = f"https://huggingface.co/datasets/{source['repo']}/resolve/{source['revision']}/{relative}"
    print(f"[HOI] download {relative}", flush=True)
    with urllib.request.urlopen(url, timeout=120) as response, destination.open("wb") as stream:
        shutil.copyfileobj(response, stream, length=1024 * 1024)
    sha = source["sha256"] if key == "parquet" else source[f"{key}_sha256"]
    return _verified(destination, sha, source.get("size_bytes") if key == "parquet" else None)


def _vertices(path: Path) -> np.ndarray:
    with path.open() as stream:
        vertices = [list(map(float, line.split()[1:4])) for line in stream if line.startswith("v ")]
    return np.asarray(vertices, dtype=np.float64)


def mesh_frame(source_mesh: Path, native_mesh: Path) -> dict:
    """Fit native_vertices = scale * source_vertices + offset, by vertex ID.

    A residual guard rejects different meshes/rotations rather than guessing a
    convention. This frame calibration does not resize either runtime asset.
    """
    a, b = _vertices(source_mesh), _vertices(native_mesh)
    if a.shape != b.shape or a.ndim != 2 or a.shape[1] != 3 or len(a) < 4:
        raise ValueError("Source/native object meshes do not share vertex correspondence")
    aa, bb = a - a.mean(0), b - b.mean(0)
    scale = float(np.sum(aa * bb) / np.sum(aa * aa))
    offset = b.mean(0) - scale * a.mean(0)
    residual = float(np.linalg.norm(scale * a + offset - b, axis=1).max())
    if not np.isfinite(scale) or scale <= 0 or residual > 2e-5:
        raise ValueError(f"Object frame calibration failed: residual={residual}, scale={scale}")
    return dict(scale=scale, offset=offset.tolist(), max_residual_m=residual,
                native_mesh_sha256=digest(native_mesh))


def read_rows(parquet: Path, roster) -> list[dict]:
    import pyarrow.parquet as pq

    columns = [
        "clip_id", "subject", "dataset", "robot", "object", "qc_pass", "fps",
        "object_scale", "object_aug_scale", "dof_names", "root_pos", "root_rot", "dof_pos",
        "object_pos", "object_rot", "contact_link_names", "per_link_contact_flags",
        "robot_foot_ground_contact_flags", "object_ground_contact_sequence",
    ]
    ids = [f"{m.split('/')[0]}/unitree_g1/{m.split('/')[1]}" for m in roster.motions]
    rows = pq.read_table(parquet, columns=columns, filters=[("clip_id", "in", ids)]).to_pylist()
    by_id = {r["clip_id"]: r for r in rows}
    if len(rows) != len(ids) or set(by_id) != set(ids):
        raise ValueError(f"Source does not contain the exact {len(ids)} distinct roster clips")
    ordered = [by_id[key] for key in ids]
    for row in ordered:
        if (row["robot"] != "unitree_g1" or row["object"] not in roster.object_names
                or not row["qc_pass"] or not np.isclose(row["fps"], 30.)
                or not np.isclose(row["object_scale"], .83)
                or not np.isclose(row["object_aug_scale"], 1.)):
            raise ValueError(f"Unexpected source identity/scale/QC: {row['clip_id']}")
    return ordered


def _quaternions(value) -> np.ndarray:
    q = np.asarray(value, dtype=np.float32)
    norm = np.linalg.norm(q, axis=-1)
    if q.ndim != 2 or q.shape[1] != 4 or not np.isfinite(q).all() or (norm < 1e-6).any():
        raise ValueError("Invalid source quaternion stream")
    q = q / norm[:, None]
    for i in range(1, len(q)):
        if np.dot(q[i - 1], q[i]) < 0:
            q[i] *= -1
    return q


def convert_channels(row: dict, joint_names: list[str], offset: list[float]) -> tuple:
    """One time grid for robot, transformed object, and discrete source contacts."""
    import torch
    from mjlab.utils.lab_api.math import quat_apply

    from orcs.cli._motion_math import _grid, _lerp, _slerp, _velocities

    source_names = row["dof_names"]
    if len(source_names) != len(set(source_names)) or set(source_names) != set(joint_names):
        raise ValueError(f"Joint-name mismatch: {row['clip_id']}")
    channels = {k: np.asarray(row[k], dtype=np.float32) for k in
                ("root_pos", "dof_pos", "object_pos")}
    n = len(channels["root_pos"])
    expected = {"root_pos": (n, 3), "dof_pos": (n, 29), "object_pos": (n, 3)}
    if n < 3 or any(a.shape != expected[k] or not np.isfinite(a).all()
                    for k, a in channels.items()):
        raise ValueError(f"Invalid source channels: {row['clip_id']}")
    rq, oq = (_quaternions(row[k]) for k in ("root_rot", "object_rot"))
    if len(rq) != n or len(oq) != n:
        raise ValueError("Robot and object timelines disagree")
    duration = (n - 1) / float(row["fps"])
    phase = torch.arange(0, duration, 1 / FPS, dtype=torch.float32) / duration
    grid = _grid(n, phase)
    joint_order = [source_names.index(name) for name in joint_names]
    state = dict(
        root_pos=_lerp(torch.from_numpy(channels["root_pos"]), *grid),
        root_quat=_slerp(torch.from_numpy(rq), *grid),
        joint_pos=_lerp(torch.from_numpy(channels["dof_pos"][:, joint_order]), *grid),
    )
    # Convert the frame before interpolation and differentiation: the origin
    # offset rotates with the object and changes its linear velocity too.
    oq = torch.from_numpy(oq)
    op = torch.from_numpy(channels["object_pos"]) - quat_apply(
        oq, torch.tensor(offset, dtype=torch.float32).expand(n, -1))
    obj = dict(root_pos=_lerp(op, *grid), root_quat=_slerp(oq, *grid))
    obj_vel = _velocities({**obj, "joint_pos": obj["root_pos"]}, 1 / FPS)
    source_indices = (phase * (n - 1)).round().long().numpy()
    contact = np.asarray(row["per_link_contact_flags"], dtype=bool)
    if contact.shape != (n, len(row["contact_link_names"])):
        raise ValueError("Invalid source contact shape")
    labels = dict(
        source_link_names=np.asarray(row["contact_link_names"]),
        source_link_contact=contact[source_indices],
        source_foot_ground=np.asarray(row["robot_foot_ground_contact_flags"], dtype=bool)[source_indices],
        source_object_ground=np.asarray(row["object_ground_contact_sequence"], dtype=bool)[source_indices],
    )
    return state, _velocities(state, 1 / FPS), dict(
        obj_pos_w=obj["root_pos"].numpy(), obj_quat_w=obj["root_quat"].numpy(),
        obj_lin_vel_w=obj_vel["root_lin_vel"].numpy(),
        obj_ang_vel_w=obj_vel["root_ang_vel"].numpy(),
    ), labels


class ContactReplay:
    """MuJoCo narrowphase on the runtime geometry, without dynamic stepping."""

    def __init__(self, object_name: str, il_names: list[str]):
        import mujoco
        from mjlab.entity import EntityCfg
        from mjlab.scene import Scene, SceneCfg
        from mjlab.terrains import TerrainEntityCfg
        from mocke.sonic import profile

        from orcs.assets import get_g1_flat_hand_cfg
        from orcs.assets.objects import object_spec

        scene = Scene(SceneCfg(num_envs=1, env_spacing=0.,
            terrain=TerrainEntityCfg(terrain_type="plane"), entities={
                "robot": profile.robot_cfg(base=get_g1_flat_hand_cfg()),
                "object": EntityCfg(spec_fn=lambda: object_spec(object_name)),
            }), device="cpu")
        self.model = m = scene.compile()
        self.data = mujoco.MjData(m)
        self.robot_bodies = [i for i in range(m.nbody) if m.body(i).name.startswith("robot/")]
        self.object_body = m.body("object/object").id
        self.names = ["world", *[m.body(i).name.removeprefix("robot/") for i in self.robot_bodies], "object"]
        self.body_column = np.zeros(m.nbody, dtype=int)
        self.body_column[self.robot_bodies] = np.arange(1, len(self.robot_bodies) + 1)
        self.body_column[self.object_body] = len(self.names) - 1
        self.joints = [int(m.joint("robot/" + name).qposadr[0]) for name in il_names]
        self.root = int(m.joint("robot/floating_base_joint").qposadr[0])
        self.obj = int(m.joint("object/object_joint").qposadr[0])
        collision = np.flatnonzero((m.geom_contype | m.geom_conaffinity) != 0)
        self.floor = [int(g) for g in collision if self.body_column[m.geom_bodyid[g]] == 0]
        # Pair margin is the sum: include geometry pairs separated by <= eps.
        m.geom_margin[collision] = CONTACT_EPS / 2
        self.floor_groups = [
            [int(g) for g in collision if m.geom_bodyid[g] == m.body(name).id]
            for name in ("object/object", "robot/left_ankle_roll_link", "robot/right_ankle_roll_link")
        ]
        self.geometry_sha256 = hashlib.sha256(b"".join(
            np.asarray(getattr(m, key)).tobytes() for key in (
                "geom_type", "geom_size", "geom_pos", "geom_quat", "geom_contype",
                "geom_conaffinity", "mesh_vert", "mesh_face",
            )
        )).hexdigest()

    def __call__(self, state: dict, obj: dict, labels: dict) -> tuple[np.ndarray, dict, np.ndarray]:
        import mujoco

        m, d = self.model, self.data
        n = len(obj["obj_pos_w"])
        matrix = np.zeros((n, len(self.names), len(self.names)), dtype=np.int8)
        floor_distances = np.zeros((n, 3), dtype=np.float32)
        for k in range(n):
            d.qpos[self.root:self.root + 3] = state["root_pos"][k].numpy()
            d.qpos[self.root + 3:self.root + 7] = state["root_quat"][k].numpy()
            d.qpos[self.joints] = state["joint_pos"][k].numpy()
            d.qpos[self.obj:self.obj + 3] = obj["obj_pos_w"][k]
            d.qpos[self.obj + 3:self.obj + 7] = obj["obj_quat_w"][k]
            mujoco.mj_forward(m, d)
            for c in d.contact:
                if c.dist > CONTACT_EPS:
                    continue
                i, j = self.body_column[m.geom_bodyid[[c.geom1, c.geom2]]]
                if i != j:
                    matrix[k, i, j] = matrix[k, j, i] = 1
            for i, group in enumerate(self.floor_groups):
                floor_distances[k, i] = min(
                    mujoco.mj_geomDistance(m, d, g, f, 2., None)
                    for g in group for f in self.floor)
        mapping = {"left_palm_pad": "left_wrist_yaw_link", "right_palm_pad": "right_wrist_yaw_link"}
        recall = {}
        for i, link in enumerate(labels["source_link_names"].tolist()):
            source = labels["source_link_contact"][:, i]
            body = mapping.get(link, link)
            target = matrix[:, self.names.index(body), -1].astype(bool)
            recall[link] = float(np.mean(target[source])) if source.any() else None
        report = dict(source_contact_recall=recall,
                      floor_distance_min_m=floor_distances.min(0).tolist(),
                      floor_distance_median_m=np.median(floor_distances, axis=0).tolist())
        return matrix, report, floor_distances


def verify_install(roster, out: Path) -> dict:
    from orcs.tasks.uolm.roster import COMPANIONS, resolve_roster

    resolve_roster(roster, out)
    manifest = json.loads((out / "manifest.json").read_text())
    expected = {f"{motion}/{name}" for motion in roster.motions for name in COMPANIONS}
    if set(manifest.get("files", {})) != expected:
        raise ValueError("Installed manifest does not describe the exact roster payload")
    actual = {p.relative_to(out).as_posix() for p in out.rglob("*") if p.is_file()}
    if actual != expected | {"manifest.json"}:
        raise ValueError(f"Unexpected/missing files in native install: {out}")
    for relative, sha in manifest["files"].items():
        if digest(out / relative) != sha:
            raise ValueError(f"Native checksum mismatch: {relative}")
    return manifest


def stage(roster, out: Path, *, source_parquet: Path | None = None,
          source_mesh: Path | None = None, device: str = "cpu") -> dict:
    """Install atomically. Repeated syncs verify native files without downloading."""
    if out.exists():
        manifest = verify_install(roster, out)
        print(f"[HOI] verified {len(roster.motions)} clips at {out}; no download needed", flush=True)
        return manifest
    from orcs.cli._motion_npz import Fk, il_body_names
    from orcs.core.paths import assets_source
    from orcs.tasks.uolm.roster import COMPANIONS, STAGING_VERSION

    if len(roster.object_names) != 1:
        raise ValueError("HOI staging currently requires one object per roster")
    source = roster.spec["source"]
    out.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=".hoi-stage-", dir=out.parent) as temporary:
        work = Path(temporary)
        parquet = (_verified(source_parquet, source["sha256"], source["size_bytes"])
                   if source_parquet is not None else _download(source, "parquet", work / "source.parquet"))
        mesh = (_verified(source_mesh, source["mesh_sha256"]) if source_mesh is not None
                else _download(source, "mesh", work / "source.obj"))
        name = roster.object_names[0]
        native_mesh = assets_source() / f"omomo_objects/{name}/{name}.obj"
        frame = mesh_frame(mesh, native_mesh)
        rows = read_rows(parquet, roster)
        fk = Fk(batch=256, fps=FPS, device=device)
        replay = ContactReplay(name, fk.il_names)
        prepared = work / "native"
        manifest = dict(staging_version=STAGING_VERSION, roster_sha256=roster.sha256,
                        motions=list(roster.motions), source=source, fps=FPS, frame=frame,
                        geometry_sha256=replay.geometry_sha256, files={}, diagnostics={})
        for i, (motion, row) in enumerate(zip(roster.motions, rows, strict=True)):
            state, vel, obj, labels = convert_channels(row, fk.il_names, frame["offset"])
            matrix, report, floor = replay(state, obj, labels)
            body = fk({k: v.to(device) for k, v in state.items()},
                      {k: v.to(device) for k, v in vel.items()})
            sample = prepared / motion
            sample.mkdir(parents=True)
            np.savez_compressed(sample / "motion.npz", fps=np.float32(FPS),
                joint_pos=state["joint_pos"].numpy(), joint_vel=vel["joint_vel"].numpy(),
                joint_names=np.asarray(fk.il_names), body_names=np.asarray(il_body_names()), **body)
            np.savez_compressed(sample / "object_motion.npz", **obj,
                                contact=matrix[:, -1, 1:-1].any(1).astype(np.float32))
            np.savez_compressed(sample / "contact_matrix.npz", matrix=matrix,
                body_names=np.asarray(replay.names), eps=CONTACT_EPS, object_collision="cvx_dcmp",
                floor_distances_m=floor, **labels)
            metadata = dict(object_path=f"omomo_objects/{name}/{name}.obj", fps=FPS,
                source_clip_id=row["clip_id"], source=source, frame=frame,
                staging_version=STAGING_VERSION, roster_sha256=roster.sha256,
                tags=roster.spec.get("tags", {}).get(row["subject"], {}), diagnostics=report,
                modifications="Named-joint mapping, object origin conversion, 30 to 50 Hz resampling, "
                              "native FK/velocities and geometry contacts. Source ground height preserved.")
            (sample / "metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")
            for filename in COMPANIONS:
                manifest["files"][f"{motion}/{filename}"] = digest(sample / filename)
            manifest["diagnostics"][motion] = report
            print(f"[HOI] {i + 1}/{len(rows)} {row['subject']}: {len(matrix)} frames", flush=True)
        (prepared / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
        verify_install(roster, prepared)
        prepared.rename(out)
    print(f"[HOI] installed {len(rows)} clips at {out}; temporary sources removed", flush=True)
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--roster", default="suitcase", help="Packaged roster name or TOML file")
    parser.add_argument("--out", type=Path)
    parser.add_argument("--source-parquet", type=Path, help="Borrow a verified local source; never delete it")
    parser.add_argument("--source-mesh", type=Path, help="Borrow a verified local source mesh; never delete it")
    parser.add_argument("--device", default="cpu", help="FK device; source replay always uses CPU")
    parser.add_argument("--check", action="store_true", help="Verify native output without fetching/staging")
    args = parser.parse_args()
    import mjlab  # noqa: F401 — finish plugin discovery before importing task modules

    from orcs.tasks.uolm.roster import load_roster

    roster = load_roster(args.roster)
    out = roster.root(args.out)
    if args.check:
        verify_install(roster, out)
        print(f"[HOI] verified {len(roster.motions)} clips at {out}")
    else:
        stage(roster, out, source_parquet=args.source_parquet,
              source_mesh=args.source_mesh, device=args.device)


if __name__ == "__main__":
    main()
