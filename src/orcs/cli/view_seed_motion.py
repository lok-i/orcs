"""Inspect an SMPL source beside its assisted G1 kinematic retarget.

The browser viewer synchronizes the original SMPL stick figure, the recorded
simulated robot/object, and world-frame assistance arrows.  It is a pure
post-processor: no policy or simulation rollout is executed.
"""

from __future__ import annotations

import argparse
import time
from contextlib import ExitStack
from pathlib import Path

import mujoco
import numpy as np
import viser
from mjviser import ViserMujocoScene

from orcs.cli.pseudo_retarget import (
    _build_cfg,
    _isolated_flat_dataset,
    _resolve_source,
)
from orcs.core.data.seeds import SeedMotion
from orcs.core.data.smpl import SMPL_PARENTS


def _smpl_segments(joints: np.ndarray) -> np.ndarray:
    return np.asarray(
        [[joints[parent], joints[joint]] for joint, parent in enumerate(SMPL_PARENTS)
         if parent >= 0],
        dtype=np.float32,
    )


def _force_arrows(seed: SeedMotion, frame: int, scale: float) -> tuple[np.ndarray, np.ndarray]:
    starts = seed.body_pos_w[frame]
    forces = seed.assist_force_w[frame]
    points = np.stack((starts, starts + scale * forces), axis=1).astype(np.float32)
    colors = np.repeat(np.array([[255, 126, 32]], dtype=np.uint8), len(starts), axis=0)
    if seed.object_pos_w is not None and seed.object_assist_force_w is not None:
        object_start = seed.object_pos_w[frame]
        object_force = seed.object_assist_force_w[frame]
        points = np.concatenate(
            (points, np.array([[object_start, object_start + scale * object_force]])),
            axis=0,
        ).astype(np.float32)
        colors = np.concatenate((colors, np.array([[210, 80, 255]], dtype=np.uint8)))
    return points, colors


def _force_display_scale(seed: SeedMotion) -> float:
    magnitudes = np.linalg.norm(seed.assist_force_w, axis=-1).reshape(-1)
    if seed.object_assist_force_w is not None:
        magnitudes = np.concatenate(
            (magnitudes, np.linalg.norm(seed.object_assist_force_w, axis=-1))
        )
    p95 = float(np.percentile(magnitudes, 95))
    return 0.75 / max(p95, 1e-6)


def _load_obj_mesh(path: Path) -> tuple[np.ndarray, np.ndarray]:
    """Load the small OBJ subset used by reconstructed source overlays."""
    vertices: list[tuple[float, float, float]] = []
    faces: list[tuple[int, int, int]] = []
    for line in path.read_text(errors="ignore").splitlines():
        fields = line.split()
        if not fields:
            continue
        if fields[0] == "v" and len(fields) >= 4:
            vertices.append(tuple(float(value) for value in fields[1:4]))
        elif fields[0] == "f" and len(fields) >= 4:
            indices = [int(value.split("/", 1)[0]) - 1 for value in fields[1:]]
            faces.extend(
                (indices[0], indices[index], indices[index + 1])
                for index in range(1, len(indices) - 1)
            )
    if not vertices or not faces:
        raise ValueError(f"source object mesh has no vertices/faces: {path}")
    return np.asarray(vertices, dtype=np.float32), np.asarray(faces, dtype=np.int32)


def _set_qpos(data, robot, seed: SeedMotion, frame: int, object_entity=None) -> None:
    free = robot.indexing.free_joint_q_adr.cpu().numpy()
    joints = robot.indexing.joint_q_adr.cpu().numpy()
    data.qpos[free[0:3]] = seed.robot_root_pos_w[frame]
    data.qpos[free[3:7]] = seed.robot_root_quat_w[frame]
    data.qpos[joints] = seed.joint_pos[frame]
    if object_entity is not None and seed.object_pos_w is not None:
        obj_free = object_entity.indexing.free_joint_q_adr.cpu().numpy()
        data.qpos[obj_free[0:3]] = seed.object_pos_w[frame]
        data.qpos[obj_free[3:7]] = seed.object_quat_w[frame]


def main() -> None:
    from orcs.tasks.uolm.sources.reconstructed import MOTION_SETS

    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument(
        "--scene", choices=("uolm", "perloco-grail"), default="uolm"
    )
    parser.add_argument(
        "--motion-set",
        choices=tuple(MOTION_SETS),
        default=None,
        help="UOLM reconstructed scene; inferred from cache metadata when omitted",
    )
    parser.add_argument(
        "--source",
        default=None,
        help="source sample/raw clip; omitted when --seed embeds a persistent source",
    )
    parser.add_argument(
        "--seed",
        type=Path,
        default=None,
        help="kinematic retarget; defaults to <source>/seed_state.npz",
    )
    parser.add_argument("--object", default=None)
    parser.add_argument("--z-up", action="store_true")
    parser.add_argument("--port", type=int, default=8080)
    args = parser.parse_args()

    if args.source is None and args.seed is None:
        parser.error("provide either --source or --seed")

    with ExitStack() as stack:
        seed = None
        if args.seed is not None:
            seed = SeedMotion.load(args.seed.expanduser().resolve())
        source = args.source
        if source is None:
            assert seed is not None
            embedded_source = Path(seed.source_path).expanduser()
            if not embedded_source.exists():
                parser.error(
                    "the seed's embedded source no longer exists; pass --source"
                )
            source = str(embedded_source)

        sample, temporary = stack.enter_context(
            _resolve_source(source, object_path=args.object, z_up=args.z_up)
        )
        motion_set = args.motion_set
        metadata_file = sample / "metadata.json"
        if motion_set is None and metadata_file.exists():
            import json

            motion_set = json.loads(metadata_file.read_text()).get("motion_set")
        if seed is None and temporary:
            parser.error("a raw source requires --seed")
        if seed is None:
            seed = SeedMotion.load(sample / "seed_state.npz")
        with np.load(sample / "smpl_motion.npz") as d:
            smpl = d["smpl_joints_viz_w"].astype(np.float32)
        source_object_pos = source_object_quat = None
        if (sample / "object_motion.npz").exists():
            with np.load(sample / "object_motion.npz") as d:
                if "obj_pos_w" in d and "obj_quat_w" in d:
                    source_object_pos = d["obj_pos_w"].astype(np.float32)
                    source_object_quat = d["obj_quat_w"].astype(np.float32)
        if len(smpl) != seed.num_frames:
            raise ValueError(
                f"source has {len(smpl)} frames but seed has {seed.num_frames}"
            )

        flat_root = None
        if args.scene == "uolm":
            flat_root, _ = stack.enter_context(_isolated_flat_dataset(sample))
        cfg, object_name = _build_cfg(
            args.scene, sample, flat_root, stack, motion_set=motion_set
        )

        from mjlab.envs import ManagerBasedRlEnv

        raw_env = ManagerBasedRlEnv(cfg=cfg, device="cpu", render_mode=None)
        stack.callback(raw_env.close)
        print("[view-seed] scene compiled; preparing playback model")
        robot = raw_env.scene["robot"]
        object_entity = raw_env.scene[object_name] if object_name else None
        model = raw_env.sim.mj_model
        data = mujoco.MjData(model)
        origin = raw_env.scene.env_origins[0].cpu().numpy()

        server = viser.ViserServer(port=args.port)
        print(f"[view-seed] server bound on port {args.port}; building scene")
        mj_scene = ViserMujocoScene(server, model, num_envs=1)
        server.scene.add_grid("/ground", width=8.0, height=8.0)

        joints0 = smpl[0] + origin
        smpl_points = server.scene.add_point_cloud(
            "/source/smpl_joints",
            joints0,
            colors=(45, 220, 235),
            point_size=0.045,
            point_shape="circle",
        )
        smpl_bones = server.scene.add_line_segments(
            "/source/smpl_bones",
            _smpl_segments(joints0),
            colors=(45, 220, 235),
            line_width=4.0,
        )
        arrow_points, arrow_colors = _force_arrows(seed, 0, _force_display_scale(seed))
        force_arrows = server.scene.add_arrows(
            "/assistance/forces",
            arrow_points,
            arrow_colors,
            shaft_radius=0.008,
            head_radius=0.025,
            head_length=0.06,
        )
        object_target = None
        if seed.object_target_pos_w is not None:
            object_target = server.scene.add_icosphere(
                "/assistance/object_target",
                radius=0.06,
                color=(210, 80, 255),
                position=seed.object_target_pos_w[0],
            )
        source_object = None
        if motion_set is not None and source_object_pos is not None:
            spec = MOTION_SETS[motion_set]
            if spec.object_half_extent is not None:
                source_object = server.scene.add_box(
                    "/source/object",
                    color=(45, 220, 235),
                    dimensions=(2.0 * spec.object_half_extent,) * 3,
                    wireframe=True,
                    position=source_object_pos[0] + origin,
                    wxyz=source_object_quat[0],
                )
            else:
                from orcs.core.paths import assets_source

                object_dir = assets_source() / spec.object_asset
                vertices, faces = _load_obj_mesh(
                    object_dir / f"{object_dir.name}.obj"
                )
                vertices *= np.asarray(spec.object_scale, dtype=np.float32)
                source_object = server.scene.add_mesh_simple(
                    "/source/object",
                    vertices,
                    faces,
                    color=(45, 220, 235),
                    wireframe=True,
                    side="double",
                    position=source_object_pos[0] + origin,
                    wxyz=source_object_quat[0],
                )

        with server.gui.add_folder("Playback"):
            gui_frame = server.gui.add_slider(
                "frame", min=0, max=seed.num_frames - 1, step=1, initial_value=0
            )
            gui_play = server.gui.add_checkbox("play", initial_value=True)
            gui_speed = server.gui.add_slider(
                "speed", min=0.1, max=3.0, step=0.1, initial_value=1.0
            )
        with server.gui.add_folder("Layers"):
            gui_smpl = server.gui.add_checkbox("original SMPL", initial_value=True)
            gui_forces = server.gui.add_checkbox("virtual forces", initial_value=True)
            gui_target = server.gui.add_checkbox("object target", initial_value=True)
            gui_source_object = server.gui.add_checkbox(
                "original object", initial_value=True
            )
        gui_info = server.gui.add_markdown("")

        display_scale = _force_display_scale(seed)

        def render(frame: int) -> None:
            _set_qpos(data, robot, seed, frame, object_entity)
            mujoco.mj_forward(model, data)
            mj_scene.update_from_mjdata(data)

            joints = smpl[frame] + origin
            smpl_points.points = joints
            smpl_bones.points = _smpl_segments(joints)
            smpl_points.visible = gui_smpl.value
            smpl_bones.visible = gui_smpl.value

            points, colors = _force_arrows(seed, frame, display_scale)
            force_arrows.points = points
            force_arrows.colors = colors
            force_arrows.visible = gui_forces.value
            if object_target is not None and seed.object_target_pos_w is not None:
                object_target.position = seed.object_target_pos_w[frame]
                object_target.visible = gui_target.value
            if (
                source_object is not None
                and source_object_pos is not None
                and source_object_quat is not None
            ):
                source_object.position = source_object_pos[frame] + origin
                source_object.wxyz = source_object_quat[frame]
                source_object.visible = gui_source_object.value

            force_norm = np.linalg.norm(seed.assist_force_w[frame], axis=-1)
            object_force = (
                float(np.linalg.norm(seed.object_assist_force_w[frame]))
                if seed.object_assist_force_w is not None else 0.0
            )
            object_error = (
                float(np.linalg.norm(seed.object_pos_w[frame] - source_object_pos[frame]))
                if seed.object_pos_w is not None and source_object_pos is not None
                else 0.0
            )
            gui_info.content = (
                f"**frame {frame}/{seed.num_frames - 1}** · "
                f"`{frame / seed.fps:.2f} s` · "
                f"{'✅ valid' if seed.valid[frame] else '🚫 invalid'}\n\n"
                f"body error mean/max: `{seed.body_tracking_error[frame].mean():.3f}` / "
                f"`{seed.body_tracking_error[frame].max():.3f} m`  \n"
                f"robot force max: `{force_norm.max():.1f} N` · "
                f"object force: `{object_force:.1f} N` · "
                f"budget scale: `{seed.assist_saturation[frame]:.3f}`  \n"
                f"object source/seed error: `{object_error:.3f} m`"
            )

        @gui_frame.on_update
        def _(_event) -> None:
            render(gui_frame.value)

        render(0)
        print(f"[view-seed] {seed.num_frames} frames — http://localhost:{args.port}")
        last = time.monotonic()
        while True:
            now = time.monotonic()
            if gui_play.value and now - last >= 1.0 / (seed.fps * gui_speed.value):
                gui_frame.value = (gui_frame.value + 1) % seed.num_frames
                last = now
            time.sleep(0.005)


if __name__ == "__main__":
    main()
