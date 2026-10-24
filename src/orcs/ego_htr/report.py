"""Optional review artifacts; never a prerequisite for retargeting or playback."""

from __future__ import annotations

import html
import json
import os
import subprocess

from .workspace import digest, write_json

os.environ.setdefault("MUJOCO_GL", "egl")

# SMPL-X's first 22 joints, shared by the source reference clips.
PARENTS = (-1, 0, 0, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 9, 9, 12, 13, 14, 16, 17, 18, 19)


def video_info(path, count, fps):
    info = json.loads(
        subprocess.check_output(
            [
                "ffprobe",
                "-v",
                "error",
                "-select_streams",
                "v:0",
                "-count_frames",
                "-show_entries",
                "stream=nb_read_frames,duration",
                "-of",
                "json",
                str(path),
            ],
            text=True,
        )
    )["streams"][0]
    if (
        int(info["nb_read_frames"]) != count
        or abs(float(info["duration"]) - count / fps) > 1 / fps
    ):
        raise ValueError(f"Video timing does not match motion: {path}")
    return info


class Render:
    def __init__(self, ws, run, surface="scan", human=False):
        os.environ.setdefault("MUJOCO_GL", "egl")
        import mujoco
        import numpy as np

        from .robot import load_motion, scene

        self.run = run
        self.motion = load_motion(ws.run_dir(run) / "motion.npz")
        self.model, self.data, _, scale = scene(
            ws, run, collision=surface == "collision"
        )
        self.human = (
            np.load(ws.asset(run, "source") / "egohtr_joints.npy")[:, :22] * scale
            if human
            else None
        )
        self.camera = mujoco.MjvCamera()
        self.camera.type = mujoco.mjtCamera.mjCAMERA_FREE
        q = self.motion["qpos"]
        self.camera.lookat[:] = q[:, :3].mean(0)
        self.camera.lookat[2] -= 0.2
        self.camera.distance = max(
            3.0, float(np.ptp(q[:, :2], axis=0).max()) * 1.35 + 1.0
        )
        self.camera.azimuth = run["preparation"].get("azimuth", 45.0)
        self.camera.elevation = run["preparation"].get("elevation", -30.0)
        self.option = mujoco.MjvOption()
        self.option.geomgroup[:] = 0
        self.option.geomgroup[1] = not human
        self.option.geomgroup[3 if surface == "collision" else 2] = 1
        self.renderer = mujoco.Renderer(
            self.model, height=720, width=960, max_geom=10000
        )

    def frame(self, index):
        import mujoco
        import numpy as np
        from PIL import Image, ImageDraw

        self.data.qpos[:] = self.motion["qpos"][index]
        mujoco.mj_kinematics(self.model, self.data)
        mujoco.mj_comPos(self.model, self.data)
        self.renderer.update_scene(
            self.data, camera=self.camera, scene_option=self.option
        )
        if self.human is not None:
            joints = self.human[int(self.motion["frame_ids"][index])]
            scene = self.renderer.scene
            for j, parent in enumerate(PARENTS):
                if parent < 0:
                    continue
                geom = scene.geoms[scene.ngeom]
                mujoco.mjv_initGeom(
                    geom,
                    mujoco.mjtGeom.mjGEOM_CAPSULE,
                    np.zeros(3),
                    np.zeros(3),
                    np.eye(3).ravel(),
                    np.array([0.95, 0.65, 0.15, 1]),
                )
                mujoco.mjv_connector(
                    geom,
                    mujoco.mjtGeom.mjGEOM_CAPSULE,
                    0.015,
                    joints[parent],
                    joints[j],
                )
                scene.ngeom += 1
        image = Image.fromarray(self.renderer.render())
        draw = ImageDraw.Draw(image)
        draw.rectangle((0, 0, image.width, 54), fill=(15, 23, 34))
        source_frame = int(self.motion["source_frames"][index])
        draw.text(
            (16, 8),
            f"{self.run['id']} | frame {index} | source frame {source_frame}",
            fill="white",
        )
        draw.text(
            (16, 30),
            "Human reference" if self.human is not None else "G1 kinematic replay",
            fill="white",
        )
        return image

    def close(self):
        self.renderer.close()


def media(ws, run, folder, surface, video=False, human=False):
    import numpy as np
    from PIL import Image

    renderer = Render(ws, run, surface, human)
    name = "human" if human else surface
    count = len(renderer.motion["qpos"])
    fps = float(renderer.motion["fps"].item())
    samples = set(np.linspace(0, count - 1, 6).astype(int).tolist())
    process = None
    pictures = []
    try:
        if video:
            process = subprocess.Popen(
                [
                    "ffmpeg",
                    "-hide_banner",
                    "-loglevel",
                    "error",
                    "-y",
                    "-f",
                    "rawvideo",
                    "-pix_fmt",
                    "rgb24",
                    "-s",
                    "960x720",
                    "-r",
                    str(fps),
                    "-i",
                    "-",
                    "-an",
                    "-c:v",
                    "libx264",
                    "-preset",
                    "fast",
                    "-crf",
                    "20",
                    "-pix_fmt",
                    "yuv420p",
                    "-movflags",
                    "+faststart",
                    str(folder / f"{name}.mp4"),
                ],
                stdin=subprocess.PIPE,
            )
        for index in range(count) if video else sorted(samples):
            frame = renderer.frame(index)
            if process:
                process.stdin.write(np.asarray(frame).tobytes())
            if index in samples:
                pictures.append(frame)
        if process:
            process.stdin.close()
            if process.wait() != 0:
                raise RuntimeError("ffmpeg failed")
            video_info(folder / f"{name}.mp4", count, fps)
        sheet = Image.new("RGB", (960 * 3, 720 * 2), (15, 23, 34))
        for index, picture in enumerate(pictures):
            sheet.paste(picture, (960 * (index % 3), 720 * (index // 3)))
        sheet.save(folder / f"{name}_contact_sheet.jpg", quality=90)
        pictures[len(pictures) // 2].save(folder / f"{name}_poster.jpg", quality=92)
    finally:
        renderer.close()
        if process and process.poll() is None:
            process.kill()
            process.wait()


def note_images(ws, run, folder):
    """Compare at the annotated source frame, not a guessed temporal offset."""
    import numpy as np
    from PIL import Image

    from .robot import load_motion

    motion = load_motion(ws.run_dir(run) / "motion.npz")
    events = ws.notes(run["clip"])
    dismissed = {
        n["note_id"]
        for n in events
        if n["type"] == "disposition" and n["status"] == "dismissed"
    }
    notes = [n for n in events if n["type"] == "comment" and n["id"] not in dismissed]
    result = []
    if not notes:
        return result
    after = Render(ws, run)
    try:
        for note in notes:
            matches = np.flatnonzero(motion["source_frames"] == note["source_frame"])
            old = ws.run(run["clip"], note["run"])
            if not len(matches) or old["status"] == "pruned":
                result.append(
                    dict(
                        note=note["id"],
                        skipped="Source frame or historical motion unavailable",
                    )
                )
                continue
            if digest(ws.run_dir(old) / "motion.npz") != note["motion_sha256"]:
                raise ValueError("Annotated historical motion hash changed")
            before = Render(ws, old)
            try:
                # Same camera makes contact/pose differences easier to compare.
                before.camera.lookat[:] = after.camera.lookat
                for attr in ("azimuth", "elevation", "distance"):
                    setattr(before.camera, attr, getattr(after.camera, attr))
                old_frame = int(note["frame"])
                new_frame = int(matches[0])
                image = Image.new("RGB", (1920, 720))
                image.paste(before.frame(old_frame), (0, 0))
                image.paste(after.frame(new_frame), (960, 0))
                filename = f"note_{len(result):03d}.jpg"
                image.save(folder / filename, quality=92)
                result.append(
                    dict(
                        note=note["id"],
                        text=note["text"],
                        image=filename,
                        before_run=old["id"],
                        after_run=run["id"],
                        before_frame=old_frame,
                        after_frame=new_frame,
                        source_frame=note["source_frame"],
                        max_joint_change_rad=float(
                            np.abs(
                                before.motion["qpos"][old_frame, 7:]
                                - motion["qpos"][new_frame, 7:]
                            ).max()
                        ),
                    )
                )
            finally:
                before.close()
    finally:
        after.close()
    return result


def report(ws, clip, revision="latest", compare="accepted", videos=False, human=False):
    from .audit import audit

    run = ws.run(clip, revision)
    folder = ws.run_dir(run) / "reports"
    folder.mkdir(exist_ok=True)
    checks = audit(ws, run)
    comparison = ws.run(clip, compare) if compare else None
    result = dict(
        run=run["id"],
        motion_sha256=digest(ws.run_dir(run) / "motion.npz"),
        checks=checks,
    )
    if comparison:
        result["comparison"] = dict(run=comparison["id"], checks=audit(ws, comparison))
    for surface in ("scan", "collision"):
        media(ws, run, folder, surface, videos)
    if human:
        media(ws, run, folder, "scan", videos, human=True)
    result["notes"] = note_images(ws, run, folder)
    write_json(folder / "report.json", result)
    sections = [
        f"<h1>{html.escape(run['id'])}</h1>",
        "<p>Kinematic review. Use Viser for exact frame feedback.</p>",
    ]
    for name in ("scan", "collision", "human") if human else ("scan", "collision"):
        sections.append(f"<h2>{name}</h2>")
        if videos:
            sections.append(
                f'<video controls preload="metadata" poster="{name}_poster.jpg" src="{name}.mp4"></video>'
            )
        sections.append(f'<a href="{name}_contact_sheet.jpg">Contact sheet</a>')
    for note in result["notes"]:
        if "image" in note:
            sections.append(
                f'<p>{html.escape(note["text"])}</p><img src="{note["image"]}">'
            )
    sections.append('<p><a href="report.json">Measurements and provenance</a></p>')
    page = (
        '<!doctype html><meta charset="utf-8"><title>EgoHTR review</title><style>body{background:#101722;color:#dde6ef;font:16px system-ui;max-width:1200px;margin:30px auto}video,img{width:100%}a{color:#73c7ff}</style>'
        + "\n".join(sections)
    )
    # Every generated gallery link must resolve locally.
    import re

    for reference in re.findall(r'(?:href|src|poster)="([^"]+)"', page):
        if not (folder / reference).is_file():
            raise ValueError(f"Broken report link: {reference}")
    (folder / "index.html").write_text(page)
    return dict(
        report=str(folder / "index.html"),
        motion_sha256=result["motion_sha256"],
        notes=len(result["notes"]),
    )
