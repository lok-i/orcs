"""Browse and CURATE staged (terrain, motion) pairs in the browser.

    python scripts/view_terrain_motions.py --source omni
    # -> http://localhost:8080

This is a deliverable, not a convenience. The uolm dataset ships a per-sample
`retargeted_motion.mp4`; OmniRetarget ships nothing, so this is how the training
roster gets chosen — and how the one thing staging cannot check itself gets
checked: **does the motion sit ON its terrain, or through it?**

What you see is exactly what RSI will write. The skeleton is drawn from
`motion.npz`'s `body_pos_w`, the same array the training loader slices, so a
pairing or frame-convention bug shows up here rather than as a mysterious
tracking regression 10k iterations in. No URDF loader, no new dependency.

Curate with Keep/Drop; "Write exclude file" emits `exclude.txt` in the
`exclude_motions` grammar that `orcs.core.data.scan` already understands, so
the roster is a config line, never a deleted file.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import viser
from mocke.mdp.joint_maps import G1_TRACKED_BODIES

from orcs.core.paths import DATA_ROOT

_IL = {name: idx for name, idx in G1_TRACKED_BODIES}

_BONES = [
    ("pelvis", "left_hip_roll_link"), ("left_hip_roll_link", "left_knee_link"),
    ("left_knee_link", "left_ankle_roll_link"),
    ("pelvis", "right_hip_roll_link"), ("right_hip_roll_link", "right_knee_link"),
    ("right_knee_link", "right_ankle_roll_link"),
    ("pelvis", "torso_link"),
    ("torso_link", "left_shoulder_roll_link"),
    ("left_shoulder_roll_link", "left_elbow_link"),
    ("left_elbow_link", "left_wrist_yaw_link"),
    ("torso_link", "right_shoulder_roll_link"),
    ("right_shoulder_roll_link", "right_elbow_link"),
    ("right_elbow_link", "right_wrist_yaw_link"),
]
_BONE_IDX = np.array([[_IL[a], _IL[b]] for a, b in _BONES])
_JOINT_IDX = np.array(sorted(_IL.values()))

_BOX_COLOR = (110, 150, 200)
_BONE_COLOR = (250, 190, 60)
_JOINT_COLOR = (250, 120, 60)
_FOOT_COLOR = (80, 230, 140)
_FOOT_IDX = np.array([_IL["left_ankle_roll_link"], _IL["right_ankle_roll_link"]])


def _tiles(root: Path) -> dict[str, dict]:
    """`<family>/level_<L>` -> {tile.json payload, clip paths}. The PATH is the
    pairing — there is no manifest to consult."""
    out = {}
    for tile_json in sorted(root.rglob("tile.json")):
        d = tile_json.parent
        clips = sorted(p for p in d.glob("sample*/motion.npz"))
        if not clips:
            continue
        out[f"{d.parent.name}/{d.name}"] = {
            "tile": json.loads(tile_json.read_text()),
            "clips": clips,
        }
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--source", default="omni")
    ap.add_argument("--root", type=Path, default=None)
    ap.add_argument("--port", type=int, default=8080)
    args = ap.parse_args()

    root = args.root or DATA_ROOT / "terrain_motions" / args.source
    tiles = _tiles(root)
    if not tiles:
        raise SystemExit(f"no staged tiles under {root} — run stage_terrain_motions.py")
    keys = sorted(tiles)
    families = sorted({k.split("/")[0] for k in keys})
    print(f"[view] {len(keys)} tiles, {len(families)} families — {root}")

    server = viser.ViserServer(port=args.port)
    server.scene.set_up_direction("+z")
    server.scene.add_grid("/grid", width=8.0, height=8.0, position=(0, 0, 0.0))

    dropped: set[str] = set()
    state: dict = {"clip": None, "playing": True}

    with server.gui.add_folder("Tile"):
        gui_family = server.gui.add_dropdown("family", tuple(families))
        gui_level = server.gui.add_dropdown("level", ("level_1.00",))
        gui_info = server.gui.add_markdown("")
    with server.gui.add_folder("Playback"):
        gui_frame = server.gui.add_slider("frame", 0, 1, 1, 0)
        gui_play = server.gui.add_checkbox("play", True)
        gui_speed = server.gui.add_slider("speed", 0.1, 3.0, 0.1, 1.0)
    with server.gui.add_folder("Curate"):
        gui_keep = server.gui.add_button("Keep")
        gui_drop = server.gui.add_button("Drop")
        gui_status = server.gui.add_markdown("")
        gui_write = server.gui.add_button("Write exclude file")

    def _levels_of(family: str) -> tuple[str, ...]:
        return tuple(sorted(k.split("/")[1] for k in keys if k.startswith(family + "/")))

    def _key() -> str:
        return f"{gui_family.value}/{gui_level.value}"

    def _draw_tile(key: str) -> None:
        server.scene.reset()
        server.scene.add_grid("/grid", width=8.0, height=8.0)
        for i, b in enumerate(tiles[key]["tile"]["boxes"]):
            server.scene.add_box(
                f"/tile/box{i}",
                color=_BOX_COLOR,
                dimensions=tuple(2 * h for h in b["half"]),  # viser wants full extents
                position=tuple(b["pos"]),
                wxyz=tuple(b["quat"]),
                opacity=0.85,
            )

    def _load(key: str) -> None:
        d = np.load(tiles[key]["clips"][0])
        bp = d["body_pos_w"].astype(np.float32)
        state["clip"] = bp
        gui_frame.max = len(bp) - 1
        gui_frame.value = 0
        _draw_tile(key)
        top = max(b["pos"][2] + b["half"][2] for b in tiles[key]["tile"]["boxes"])
        foot_min = float(bp[:, _FOOT_IDX, 2].min())
        warn = "  ⚠ **sinks below ground**" if foot_min < -0.02 else ""
        gui_info.content = (
            f"**{key}** — {len(bp)} frames @ 50 Hz\n\n"
            f"box top `{top:.2f} m` · lowest ankle `{foot_min:+.3f} m`{warn}"
        )
        _status()

    def _status() -> None:
        k = _key()
        mark = "🚫 DROPPED" if k in dropped else "✅ kept"
        gui_status.content = f"{mark} — {len(dropped)} dropped of {len(keys)}"

    def _render(frame: int) -> None:
        bp = state["clip"]
        if bp is None:
            return
        pts = bp[frame]
        server.scene.add_point_cloud(
            "/robot/joints", points=pts[_JOINT_IDX], colors=_JOINT_COLOR,
            point_size=0.035, point_shape="circle")
        server.scene.add_point_cloud(
            "/robot/feet", points=pts[_FOOT_IDX], colors=_FOOT_COLOR,
            point_size=0.05, point_shape="circle")
        server.scene.add_line_segments(
            "/robot/bones", points=pts[_BONE_IDX], colors=_BONE_COLOR, line_width=4.0)

    @gui_family.on_update
    def _(_evt) -> None:
        levels = _levels_of(gui_family.value)
        gui_level.options = levels
        if gui_level.value not in levels:
            gui_level.value = levels[0]
        _load(_key())

    @gui_level.on_update
    def _(_evt) -> None:
        _load(_key())

    @gui_play.on_update
    def _(_evt) -> None:
        state["playing"] = gui_play.value

    @gui_keep.on_click
    def _(_evt) -> None:
        dropped.discard(_key())
        _status()

    @gui_drop.on_click
    def _(_evt) -> None:
        dropped.add(_key())
        _status()

    @gui_write.on_click
    def _(_evt) -> None:
        # `exclude_motions` grammar: "<dataset>/<motion>" drops that motion in
        # that dataset — here that is exactly "<family>/level_<L>".
        path = root / "exclude.txt"
        path.write_text("".join(f"{k}\n" for k in sorted(dropped)))
        gui_status.content = f"wrote {len(dropped)} entries -> `{path}`"
        print(f"[view] wrote {len(dropped)} exclusions -> {path}")

    gui_level.options = _levels_of(gui_family.value)
    gui_level.value = gui_level.options[0]
    _load(_key())

    print(f"[view] http://localhost:{args.port}")
    t = 0.0
    while True:
        if state["playing"] and state["clip"] is not None:
            t += gui_speed.value
            if t >= 1.0:
                gui_frame.value = int((gui_frame.value + int(t)) % (gui_frame.max + 1))
                t -= int(t)
        _render(int(gui_frame.value))
        time.sleep(1 / 50)


if __name__ == "__main__":
    main()
