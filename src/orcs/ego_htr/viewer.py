"""Viser G1/terrain review. Only the selected terrain is resident; no dynamics."""

from __future__ import annotations

import threading
import time
from queue import SimpleQueue

import mujoco
import numpy as np
import trimesh
import viser
from mjviser.conversions import merge_geoms

from .robot import load_motion, terrain_meshes
from .workspace import digest


def variants(ws, cid):
    clip = ws.clip(cid)
    result = {}
    if clip.get("latest"):
        result["Latest"] = clip["latest"]
    if clip.get("accepted"):
        result["Accepted"] = clip["accepted"]
    for run in reversed(ws.runs(cid)):
        if run["status"] in ("ready", "imported"):
            result[run["id"]] = run["id"]
    return result


def load_clip(ws, cid, revision):
    run = ws.run(cid, revision)
    model = mujoco.MjModel.from_xml_path(str(ws.asset(run, "robot")))
    motion_path = ws.run_dir(run) / "motion.npz"
    sha = digest(motion_path)
    if sha != run["motion_sha256"]:
        raise ValueError("Motion changed after run completion")
    motion = load_motion(motion_path, model)
    scale = float(motion["smpl_scale"].item())
    scan = trimesh.load(ws.asset(run, "scan"), force="mesh", process=False)
    scan.vertices *= scale
    hulls = []
    for node, path in terrain_meshes(ws.asset(run, "terrain")):
        mesh = trimesh.load(path, force="mesh", process=False)
        mesh.vertices *= scale * np.fromstring(node.get("scale", "1 1 1"), sep=" ")
        hulls.append(mesh)
    if not hulls:
        raise ValueError("Prepared scene has no collision hulls")
    return dict(
        run=run,
        model=model,
        motion_hash=sha,
        qpos=motion["qpos"],
        frames=motion["frame_ids"],
        source_frames=motion["source_frames"],
        fps=float(motion["fps"].item()),
        scale=scale,
        prep=run["preparation"],
        scan=scan,
        proxy=trimesh.util.concatenate(hulls),
        hulls=len(hulls),
    )


class Viewer:
    def __init__(self, ws, host="127.0.0.1", port=8085, cid=None, start_time=0.0):
        self.ws = ws
        self.snapshot_lock = threading.Lock()
        self.displayed = None
        self.commands = SimpleQueue()
        self.cid = self.variant = self.clip = None
        self.bodies = {}
        self.mesh_handles = []
        self.last_frame = -1
        self.fraction = 0.0
        self.server = viser.ViserServer(
            host=host, port=port, label="EgoHTR · G1 review"
        )
        s = self.server
        s.gui.configure_theme(
            dark_mode=True, control_width="medium", show_share_button=False
        )
        s.scene.set_up_direction("+z")
        s.scene.configure_default_lights(enabled=False)
        s.scene.configure_environment_map(
            "studio", background=False, environment_intensity=0.3
        )
        s.scene.add_light_ambient("/lights/fill", intensity=0.5)
        s.scene.add_light_directional(
            "/lights/key", intensity=1.5, wxyz=(0.92388, 0.27060, 0.27060, 0)
        )
        self.robot = s.scene.add_frame("/robot", show_axes=False)
        self.refresh_labels()
        selected = next(
            (
                label
                for label, ident in self.labels.items()
                if cid and cid in (ident, ws.clip(ident)["label"])
            ),
            None,
        )
        if cid and selected is None:
            s.stop()
            raise ValueError(f"No reviewable motion for {cid}")
        selected = selected or next(
            (
                label
                for label, ident in self.labels.items()
                if ws.clip(ident)["label"] == "03"
            ),
            None,
        )
        selected = selected or next(iter(self.labels), "No prepared clips")
        s.gui.add_markdown(
            "**G1 / EgoHTR**\n\nDrag to orbit · right-drag to pan · scroll to zoom. Kinematic replay."
        )
        with s.gui.add_folder("Clip"):
            self.choose = s.gui.add_dropdown(
                "Motion",
                tuple(self.labels) or ("No prepared clips",),
                initial_value=selected,
            )
            self.version = s.gui.add_dropdown("Version", ("Latest",))
            s.gui.add_button("Reload files").on_click(
                lambda _: self.commands.put("reload")
            )
            self.info = s.gui.add_markdown(
                "No clips prepared yet. Ask your agent to prepare a clip, then Reload files."
            )
        with s.gui.add_folder("Playback"):
            self.play = s.gui.add_checkbox("Play", False)
            self.loop = s.gui.add_checkbox("Loop", False)
            self.speed = s.gui.add_slider("Speed", 0.1, 2.0, 0.1, 1.0)
            self.frame = s.gui.add_slider("Frame", 0, 1, 1, 0)
            self.seconds = s.gui.add_number(
                "Clip time (s)", 0.0, min=0, max=10, step=1 / 30
            )
            self.stamp = s.gui.add_markdown("")
            for label, cmd in [("Previous frame", "prev"), ("Next frame", "next")]:
                s.gui.add_button(label).on_click(lambda _, c=cmd: self.commands.put(c))
            self.frame.on_update(
                lambda e: self.commands.put("scrub") if e.client else None
            )
            self.seconds.on_update(
                lambda e: (
                    self.commands.put(("seek", e.target.value)) if e.client else None
                )
            )
        with s.gui.add_folder("Terrain and camera"):
            self.surface = s.gui.add_dropdown(
                "Surface", ("Scan", "Collision", "Overlay", "Hidden")
            )
            self.wire = s.gui.add_checkbox("Wireframe", False)
            self.opacity = s.gui.add_slider("Terrain opacity", 0.1, 1.0, 0.05, 1.0)
            self.show_robot = s.gui.add_checkbox("Show G1", True)
            self.show_path = s.gui.add_checkbox("Root path", False)
            for label, cmd in [("Fit whole clip", "fit"), ("Focus on G1", "focus")]:
                s.gui.add_button(label).on_click(
                    lambda e, c=cmd: self.commands.put((c, e.client))
                )
        with s.gui.add_folder("Feedback"):
            self.note = s.gui.add_text("Note at this frame", "", multiline=True)
            s.gui.add_button("Save note").on_click(lambda _: self.queue_note())
            self.note_status = s.gui.add_markdown(
                "Notes retain the exact displayed revision and frame."
            )
        if self.labels:
            self.version.options = tuple(variants(ws, self.labels[selected]))
            self.load(self.labels[selected], "Latest", round(start_time * 30))
        s.on_client_connect(lambda client: self.commands.put(("fit", client)))

    def refresh_labels(self):
        self.labels = {
            f"{c['label']} · {c['title']}": c["id"]
            for c in self.ws.clips()
            if variants(self.ws, c["id"])
        }

    def queue_note(self):
        # Snapshot at callback time, not later when the playback loop drains it.
        with self.snapshot_lock:
            snapshot = self.displayed
        if snapshot and self.note.value.strip():
            self.commands.put(
                (
                    "note",
                    dict(
                        snapshot,
                        text=self.note.value.strip(),
                        surface=self.surface.value,
                    ),
                )
            )

    def load(self, cid, variant, frame=0):
        self.play.value = False
        self.info.content = "Loading terrain…"
        self.server.flush()
        choices = variants(self.ws, cid)
        clip = load_clip(self.ws, cid, choices[variant])
        with self.server.atomic():
            for handle in self.mesh_handles:
                handle.remove()
            for handle in self.bodies.values():
                handle.remove()
            self.mesh_handles = []
            self.bodies = {}
            self.model = clip["model"]
            self.data = mujoco.MjData(self.model)
            for body in range(1, self.model.nbody):
                ids = np.flatnonzero(
                    (self.model.geom_bodyid == body) & (self.model.geom_group == 1)
                ).tolist()
                if ids:
                    self.bodies[body] = self.server.scene.add_mesh_trimesh(
                        f"/robot/{self.model.body(body).name}",
                        merge_geoms(self.model, ids),
                    )
            for key, color in [("scan", (115, 130, 112)), ("proxy", (54, 173, 229))]:
                mesh = clip[key]
                self.mesh_handles.append(
                    self.server.scene.add_mesh_simple(
                        "/terrain/" + key,
                        np.asarray(mesh.vertices, np.float32),
                        np.asarray(mesh.faces, np.uint32),
                        color=color,
                        side="double",
                        cast_shadow=False,
                        receive_shadow=True,
                        flat_shading=True,
                    )
                )
            points = clip["qpos"][:, :3]
            self.mesh_handles.append(
                self.server.scene.add_line_segments(
                    "/trajectory",
                    np.stack((points[:-1], points[1:]), axis=1).astype(np.float32),
                    colors=(255, 188, 70),
                    thickness=2,
                    thickness_units="screen",
                    visible=False,
                )
            )
            self.cid, self.variant, self.clip = cid, variant, clip
            self.frame.max = max(1, len(points) - 1)
            self.frame.value = min(max(0, frame), len(points) - 1)
            self.seconds.max = (len(points) - 1) / clip["fps"]
            self.last_frame = -1
            self.fraction = 0
            p = clip["prep"]
            run = clip["run"]
            source = run["source"]
            self.info.content = (
                f"**{p['title']}** · {run['id']}\n\n"
                f"Source `{source['sequence']}` · {source['start_frame'] / source['fps']:g}–{source['end_frame'] / source['fps']:g} s\n\n"
                f"{len(points)} frames @ {clip['fps']:g} Hz · scene scale {clip['scale']:.3f}×\n\n"
                f"Scan: {len(clip['scan'].faces):,} triangles · collision: {clip['hulls']} convex pieces."
            )
        self.draw()
        print(f"Loaded {cid} / {clip['run']['id']}: {len(points)} frames", flush=True)

    def camera(self, client=None, focus=False):
        if self.clip is None:
            return
        q = self.clip["qpos"]
        target = (
            q[min(int(self.frame.value), len(q) - 1), :3].copy()
            if focus
            else q[:, :3].mean(0)
        )
        target[2] -= 0.2
        distance = 2.5 if focus else max(3.0, np.ptp(q[:, :2], axis=0).max() * 1.35 + 1)
        az = np.deg2rad(self.clip["prep"].get("azimuth", 45))
        el = np.deg2rad(-self.clip["prep"].get("elevation", -30))
        offset = distance * np.array(
            [np.cos(az) * np.cos(el), np.sin(az) * np.cos(el), np.sin(el)]
        )
        for c in [client] if client else self.server.get_clients().values():
            with c.atomic():
                c.camera.up_direction = (0, 0, 1)
                c.camera.position = target + offset
                c.camera.look_at = target

    def draw(self):
        if self.clip is None:
            return
        scan, proxy, path = self.mesh_handles
        scan.visible = self.surface.value in ("Scan", "Overlay")
        proxy.visible = self.surface.value in ("Collision", "Overlay")
        scan.wireframe = proxy.wireframe = self.wire.value
        scan.opacity = self.opacity.value
        proxy.opacity = self.opacity.value * (
            0.4 if self.surface.value == "Overlay" else 1
        )
        self.robot.visible = self.show_robot.value
        path.visible = self.show_path.value
        frame = min(int(self.frame.value), len(self.clip["qpos"]) - 1)
        if frame == self.last_frame:
            return
        self.data.qpos[:] = self.clip["qpos"][frame]
        mujoco.mj_kinematics(self.model, self.data)
        with self.server.atomic():
            for body, handle in self.bodies.items():
                handle.position = self.data.xpos[body]
                handle.wxyz = self.data.xquat[body]
            self.seconds.value = frame / self.clip["fps"]
            source_frame = int(self.clip["source_frames"][frame])
            self.stamp.content = (
                f"**{self.variant} · {frame / self.clip['fps']:.3f} s**\n\n"
                f"Frame {frame} · source {source_frame / self.clip['run']['source']['fps']:.3f} s (frame {source_frame})"
            )
        with self.snapshot_lock:
            self.displayed = dict(
                run=self.clip["run"], frame=frame, motion_hash=self.clip["motion_hash"]
            )
        self.last_frame = frame

    def tick(self, dt):
        cid = self.labels.get(self.choose.value)
        if cid and cid != self.cid:
            self.version.options = tuple(variants(self.ws, cid))
            self.version.value = "Latest"
            self.load(cid, "Latest")
            self.camera()
        elif self.clip and self.version.value != self.variant:
            self.load(cid, self.version.value, int(self.frame.value))
        while not self.commands.empty():
            command = self.commands.get()
            if isinstance(command, tuple):
                action, value = command
                if action == "note":
                    record = self.ws.comment(**value)
                    self.note_status.content = (
                        f"Saved **{record['run']}**, frame **{record['frame']}**."
                    )
                    if self.note.value.strip() == value["text"]:
                        self.note.value = ""
                elif action == "seek" and self.clip:
                    self.play.value = False
                    self.frame.value = int(
                        np.clip(
                            round(value * self.clip["fps"]),
                            0,
                            len(self.clip["qpos"]) - 1,
                        )
                    )
                elif action in ("fit", "focus"):
                    self.camera(value, focus=action == "focus")
            elif command == "reload":
                self.refresh_labels()
                self.choose.options = tuple(self.labels) or ("No prepared clips",)
                cid = self.labels.get(self.choose.value) or next(
                    iter(self.labels.values()), None
                )
                if cid:
                    self.choose.value = next(
                        label for label, ident in self.labels.items() if ident == cid
                    )
                    choices = variants(self.ws, cid)
                    self.version.options = tuple(choices)
                    if self.version.value not in choices:
                        self.version.value = "Latest"
                    self.load(cid, self.version.value, int(self.frame.value))
            elif self.clip and command in ("prev", "next", "scrub"):
                self.play.value = False
                self.fraction = 0
                if command != "scrub":
                    self.frame.value = int(
                        np.clip(
                            self.frame.value + (1 if command == "next" else -1),
                            0,
                            len(self.clip["qpos"]) - 1,
                        )
                    )
        if self.clip and self.play.value:
            self.fraction += dt * self.clip["fps"] * self.speed.value
            advance = int(self.fraction)
            self.fraction -= advance
            frame = int(self.frame.value) + advance
            last = len(self.clip["qpos"]) - 1
            if frame > last:
                frame = frame % (last + 1) if self.loop.value else last
                if not self.loop.value:
                    self.play.value = False
            self.frame.value = frame
        self.draw()

    def run(self):
        before = time.monotonic()
        try:
            while True:
                now = time.monotonic()
                try:
                    self.tick(min(now - before, 0.2))
                except (ValueError, FileNotFoundError) as error:
                    self.play.value = False
                    self.info.content = f"Review error: {error}"
                before = now
                time.sleep(1 / 60)
        except KeyboardInterrupt:
            pass
        finally:
            self.server.stop()
