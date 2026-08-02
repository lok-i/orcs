"""Browse and CURATE staged (terrain, motion) pairs — as the actual G1.

    python scripts/view_terrain_motions.py --source omni
    # -> http://localhost:8080

A deliverable, not a convenience. The uolm dataset ships a per-sample
`retargeted_motion.mp4`; OmniRetarget ships nothing, so this is how the
training roster gets chosen — and how the one thing staging cannot check itself
gets checked: **does the motion sit ON its terrain, or through it?**

What you see is the REAL MODEL, not a depiction of it:

  robot    `orcs.assets.get_g1_flat_hand_cfg()` — the same entity the env
           builds, posed by writing `motion.npz` into `qpos` and running
           `mj_forward`. No skeleton, no bone list, no second opinion about
           what a link is.
  terrain  `perloco.terrain.TileTerrainCfg.function()` — the same builder the
           env grid calls, at the same tile-local offset.

That is the whole design: a viewer built out of an ABSTRACTION of the data can
only ever disagree with the data decoratively. An earlier version drew a
14-node stick figure and its limbs stretched — which turned out to be a real
joint-order bug in staging, but took a bone-length audit to tell apart from
"the skeleton is a bad drawing". This version cannot have that ambiguity: a
pairing, frame or joint-order bug shows up as a broken robot on a real tile.

It is also a pre-flight of the env's terrain builder — a box placed wrong here
is a box placed wrong in training.

Curate with Keep/Drop; "Write exclude file" emits `exclude.txt` in the
`exclude_motions` grammar that `orcs.core.data.scan` already understands, so
the roster is a config line, never a deleted file.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import mujoco
import numpy as np
import viser
from mjlab.entity import Entity
from mjviser import ViserMujocoScene
from mocke.mdp.joint_maps import IL2MJ

from orcs.assets import get_g1_flat_hand_cfg
from orcs.core.paths import DATA_ROOT
from orcs.tasks.perloco.terrain import TILE_SIZE, TileTerrainCfg


def _tiles(root: Path) -> dict[str, dict]:
    """`<family>/level_<L>` -> {level, clip paths}. The PATH is the pairing —
    there is no manifest to consult."""
    out: dict[str, dict] = {}
    for tile_json in sorted(root.rglob("tile.json")):
        d = tile_json.parent
        clips = sorted(d.glob("sample*/motion.npz"))
        if clips:
            out[f"{d.parent.name}/{d.name}"] = {
                "family": d.parent.name,
                "level": float(d.name[len("level_"):]),
                "clips": clips,
            }
    return out


def _build_model(root: Path, family: str, level: float) -> mujoco.MjModel:
    """G1 + one staged tile, compiled — through the ENV's terrain builder.

    `TileTerrainCfg` writes into a body named `terrain` (that is mjlab's
    contract, and the reason the generator can rename every tile geom
    afterwards), so we make one before calling it.
    """
    spec = Entity(get_g1_flat_hand_cfg()).spec
    spec.worldbody.add_body(name="terrain")
    TileTerrainCfg(
        root=root, family=family, levels=(level,), size=TILE_SIZE,
    ).function(0.0, spec, np.random.default_rng(0))
    return spec.compile()


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
    families = sorted({tiles[k]["family"] for k in keys})
    # Tile-local offset of the tile's own origin — what `env_origins` supplies
    # at runtime. Applying it HERE is what makes the viewer's robot-vs-terrain
    # placement the same arithmetic RSI does.
    origin = np.array([0.5 * TILE_SIZE[0], 0.5 * TILE_SIZE[1], 0.0])
    print(f"[view] {len(keys)} tiles, {len(families)} families — {root}")

    server = viser.ViserServer(port=args.port)
    dropped: set[str] = set()
    state: dict = {"scene": None, "model": None, "data": None,
                   "clip": None, "playing": True}

    with server.gui.add_folder("Tile"):
        gui_family = server.gui.add_dropdown("family", tuple(families))
        gui_level = server.gui.add_dropdown("level", ("level_1.00",))
        gui_sample = server.gui.add_dropdown("sample", ("sample1",))
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

    def _key() -> str:
        return f"{gui_family.value}/{gui_level.value}"

    def _levels_of(family: str) -> tuple[str, ...]:
        return tuple(sorted(k.split("/")[1] for k in keys
                            if tiles[k]["family"] == family))

    def _status() -> None:
        mark = "🚫 DROPPED" if _key() in dropped else "✅ kept"
        gui_status.content = f"{mark} — {len(dropped)} dropped of {len(keys)}"

    def _load_tile(key: str) -> None:
        """Recompile the model — a different tile IS a different model."""
        t = tiles[key]
        model = _build_model(root, t["family"], t["level"])
        server.scene.reset()
        state["model"] = model
        state["data"] = mujoco.MjData(model)
        state["scene"] = ViserMujocoScene(server, model, num_envs=1)
        gui_sample.options = tuple(p.parent.name for p in t["clips"])
        if gui_sample.value not in gui_sample.options:
            gui_sample.value = gui_sample.options[0]
        _load_clip(key)

    def _load_clip(key: str) -> None:
        t = tiles[key]
        clip = next(p for p in t["clips"] if p.parent.name == gui_sample.value)
        with np.load(clip) as d:
            state["clip"] = {
                # joint_pos is staged in ISAACLAB order; the sim wants MuJoCo
                # order. Same permute the training loader applies — so a
                # scrambled clip is visibly scrambled here.
                "joint_pos": d["joint_pos"][:, IL2MJ].astype(np.float64),
                "root_pos": d["body_pos_w"][:, 0].astype(np.float64),
                "root_quat": d["body_quat_w"][:, 0].astype(np.float64),
            }
        n = len(state["clip"]["joint_pos"])
        gui_frame.max = n - 1
        gui_frame.value = 0

        tile_json = json.loads((clip.parent.parent / "tile.json").read_text())
        top = max(b["pos"][2] + b["half"][2] for b in tile_json["boxes"])
        foot = float(state["clip"]["root_pos"][:, 2].min())
        gui_info.content = (
            f"**{key}** · {gui_sample.value} — {n} frames @ 50 Hz\n\n"
            f"{len(tile_json['boxes'])} box(es) · top `{top:.2f} m` · "
            f"lowest pelvis `{foot:+.3f} m`")
        _status()

    def _render(frame: int) -> None:
        clip, model, data = state["clip"], state["model"], state["data"]
        if clip is None:
            return
        data.qpos[0:3] = clip["root_pos"][frame] + origin
        data.qpos[3:7] = clip["root_quat"][frame]
        data.qpos[7:] = clip["joint_pos"][frame]
        mujoco.mj_forward(model, data)
        state["scene"].update_from_mjdata(data)

    @gui_family.on_update
    def _(_evt) -> None:
        levels = _levels_of(gui_family.value)
        gui_level.options = levels
        if gui_level.value not in levels:
            gui_level.value = levels[0]
        _load_tile(_key())

    @gui_level.on_update
    def _(_evt) -> None:
        _load_tile(_key())

    @gui_sample.on_update
    def _(_evt) -> None:
        _load_clip(_key())

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
    _load_tile(_key())

    print(f"[view] http://localhost:{args.port}")
    t = 0.0
    while True:
        if state["playing"] and state["clip"] is not None:
            t += gui_speed.value
            if t >= 1.0:
                gui_frame.value = int(
                    (gui_frame.value + int(t)) % (gui_frame.max + 1))
                t -= int(t)
        _render(int(gui_frame.value))
        time.sleep(1 / 50)


if __name__ == "__main__":
    main()
