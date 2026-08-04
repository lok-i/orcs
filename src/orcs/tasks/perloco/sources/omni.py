"""OmniRetarget `robot-terrain` — the tier-1 source.

Layout (as shipped by the HF dataset, unmodified):

    OmniRetarget_Dataset/
      robot-terrain.zip                       climb_<NN>_z_scale_<L>.npz
      models/terrain/climb_<NN>/
        multi_boxes_z_scale_<L>.urdf          1-2 <mesh> links welded to world
        box_models/box<K>.obj                 8 verts / 12 faces

Two facts make this the source to build on:

1. **The terrain IS boxes.** Every `box<K>.obj` is a z-extruded rectangle at
   arbitrary yaw, recovered exactly by `terrain_spec.box_from_vertices` — no
   mesh, no convex decomposition, no heightfield bake.
2. **The pairing is in the filename.** `climb_05_z_scale_1.0.npz` pairs with
   `climb_05/multi_boxes_z_scale_1.0.urdf`. 145 clips, 29 families x 5
   z-scales, exactly one clip per tile — the dataset was built for the grid we
   put it on.

`z_scale` scales only the URDF mesh scale's z, so it is literally obstacle
height: a principled difficulty axis for the curriculum row, not an index
smuggled through a float.

Motion payload is `qpos (T, 36)` + `fps`, Drake `MultibodyPlant` order:
`[quat wxyz(4) | pos xyz(3) | 29 joints in URDF/DFS order]` — confirmed
against the dataset's own `visualize.py`. Joint NAMES are read from the shipped
URDF rather than assumed, so a dataset revision that reorders them fails loudly
in the staging permute instead of silently transposing the robot.
"""

from __future__ import annotations

import io
import re
import xml.etree.ElementTree as ET
import zipfile
from collections.abc import Iterable
from pathlib import Path

import numpy as np

from orcs.tasks.perloco.terrain_spec import (
    BoxSpec,
    ClipSpec,
    TileSpec,
    box_from_vertices,
)

__all__ = ["OmniRetargetSource"]

_CLIP_RE = re.compile(r"^(?P<family>.+?)_z_scale_(?P<level>[0-9.]+)\.npz$")
_ROBOT_URDF = "models/g1/g1_29dof.urdf"
_JOINT_RE = re.compile(r'joint name="([^"]+)" type="(?!fixed)([^"]+)"')


def _obj_verts(text: str) -> np.ndarray:
    return np.array([[float(x) for x in ln.split()[1:4]]
                     for ln in text.splitlines() if ln.startswith("v ")])


class OmniRetargetSource:
    """OmniRetarget robot-terrain, read in place (the zip is never extracted)."""

    name = "omni"
    default_root = "OmniRetarget_Dataset"

    def __init__(
        self,
        root: str | Path,
        families: tuple[str, ...] | None = None,
        levels: tuple[float, ...] | None = None,
    ) -> None:
        self.root = Path(root)
        self._zip = self.root / "robot-terrain.zip"
        self._terrain_root = self.root / "models" / "terrain"
        if not self._zip.exists():
            raise FileNotFoundError(f"missing {self._zip}")
        if not self._terrain_root.is_dir():
            raise FileNotFoundError(f"missing {self._terrain_root}")
        self.families = set(families) if families else None
        self.levels = set(levels) if levels else None
        self._joint_names = self._read_joint_names()

    # ── the shipped URDF is the joint-order truth ──

    def _read_joint_names(self) -> tuple[str, ...]:
        urdf = self.root / _ROBOT_URDF
        names = tuple(m.group(1) for m in _JOINT_RE.finditer(urdf.read_text()))
        names = tuple(n for n in names if n != "floating_base_joint")
        if len(names) != 29:
            raise ValueError(
                f"{urdf}: expected 29 actuated joints, found {len(names)}")
        return names

    def _wanted(self, family: str, level: float) -> bool:
        return ((self.families is None or family in self.families)
                and (self.levels is None or level in self.levels))

    def _pairs(self) -> list[tuple[str, str, float]]:
        """(zip member, family, level) for every clip that passes the roster."""
        out = []
        with zipfile.ZipFile(self._zip) as z:
            for member in sorted(z.namelist()):
                m = _CLIP_RE.match(Path(member).name)
                if not m:
                    continue
                family, level = m["family"], float(m["level"])
                if self._wanted(family, level):
                    out.append((member, family, level))
        return out

    # ── the protocol ──

    def _boxes_of(self, family: str, level: float) -> tuple[BoxSpec, ...]:
        """One box per URDF <link>, from its COLLISION geometry.

        Parsed as XML, per link — not regex-scraped over the file. Every link
        carries the same mesh twice (visual + collision), so a flat scan yields
        `[l1_vis, l1_col, l2_vis, l2_col, ...]`; anything that dedups by
        halving that list returns link 1 twice and drops link 2 entirely.
        """
        urdf = self._terrain_root / family / f"multi_boxes_z_scale_{level:.1f}.urdf"
        if not urdf.exists():
            raise FileNotFoundError(
                f"clip {family}_z_scale_{level} has no terrain at {urdf}")
        boxes = []
        for link in ET.parse(urdf).getroot().findall("link"):
            # collision is what physics uses; visual is the same mesh anyway
            node = link.find("collision") or link.find("visual")
            mesh = node.find("geometry/mesh") if node is not None else None
            if mesh is None:
                continue
            origin = node.find("origin")
            xyz = (0.0, 0.0, 0.0)
            if origin is not None:
                xyz = tuple(float(x) for x in origin.get("xyz", "0 0 0").split())
                rpy = [float(x) for x in origin.get("rpy", "0 0 0").split()]
                if any(abs(a) > 1e-9 for a in rpy):
                    raise NotImplementedError(
                        f"{urdf}: link '{link.get('name')}' has a rotated origin "
                        f"(rpy={rpy}); composing it with the mesh's own yaw is "
                        "unimplemented because no shipped asset needs it")
            obj = urdf.parent / mesh.get("filename")
            scale = tuple(float(x) for x in mesh.get("scale", "1 1 1").split())
            boxes.append(box_from_vertices(_obj_verts(obj.read_text()), scale, xyz))
        if not boxes:
            raise ValueError(f"{urdf}: no mesh geometry found")
        return tuple(boxes)

    def tiles(self) -> Iterable[TileSpec]:
        """One tile per (family, level) that a surviving clip refers to.

        Driven by the CLIPS, not by the URDFs on disk: a tile with no motion is
        an empty grid cell an env could be spawned onto with nothing to track.
        """
        for _, family, level in self._pairs():
            yield TileSpec(family=family, level=level,
                           boxes=self._boxes_of(family, level))

    def clips(self) -> Iterable[ClipSpec]:
        with zipfile.ZipFile(self._zip) as z:
            for member, family, level in self._pairs():
                with np.load(io.BytesIO(z.read(member))) as d:
                    qpos, fps = d["qpos"], float(d["fps"])
                if qpos.shape[1] != 36:
                    raise ValueError(
                        f"{member}: expected qpos (T, 36), got {qpos.shape}")
                yield ClipSpec(
                    name=Path(member).stem,
                    root_quat=qpos[:, 0:4].astype(np.float64),   # Drake wxyz
                    root_pos=qpos[:, 4:7].astype(np.float64),
                    joint_pos=qpos[:, 7:36].astype(np.float64),
                    joint_names=self._joint_names,
                    fps=fps,
                    family=family,
                    level=level,
                    meta={"source": "omniretarget/robot-terrain",
                          "member": member, "src_fps": fps},
                )
