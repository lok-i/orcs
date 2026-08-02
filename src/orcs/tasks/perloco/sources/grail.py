"""NVIDIA GRAIL terrain categories (curb today, stairs when they land).

    <root>/<category>/
      robot/<cat>__<terrain>__<take>.pkl     joblib: dof, root_trans_offset, root_rot
      objects/...                            terrain pose (static) + scale
      object_usd/...                         terrain geometry

Two things differ from OmniRetarget and both are converted here, so the runtime
and the staging pipeline stay identical:

  root_rot is **xyzw**, ours is wxyz   (q0 ~ (0,0,+-.7,+-.7) = upright + yaw)
  25 Hz, ours is 50                    (staging resamples; nothing to do here)

The terrain is 6-quad boxes packed one after another in the point array — 2-5
per curb, yaw-free, exact. Geometry is identical across a terrain's takes, so
one TILE holds several clips and the tile mask does real work.

`dof` is **MuJoCo actuator order**, NOT IsaacLab: GRAIL's retargeter writes
`model.actuator(i) -> qpos[i+7]` (`grail/retargeting/retarget.py`), and only 2
of 29 slots coincide with IL. The dataset ships no joint names, so the order is
spelled out below and staging permutes by name like every other source.

Do not replace `_JOINT_NAMES` with "it is already IL" — that was tried, and 27
of 29 joints were silently transposed. A bone-length audit did not catch it
(a span is mostly fixed link geometry; the joint-dependent part is centimetres)
and the arms flailed while the legs looked fine.
"""

from __future__ import annotations

from collections.abc import Iterable
from pathlib import Path

import numpy as np

from orcs.tasks.perloco.terrain_spec import (
    BoxSpec,
    ClipSpec,
    TileSpec,
    box_from_vertices,
)

__all__ = ["GrailSource"]

_VERTS_PER_BOX = 24  # 6 quads, unshared corners

_JOINT_NAMES = (
    # MuJoCo XML / actuator order, verbatim from GRAIL's own retargeter
    # (`grail/retargeting/retarget.py`, the `init_config` dict it writes into
    # `qpos[i + 7]`). Asserted against the robot spec at staging time.
    "left_hip_pitch_joint", "left_hip_roll_joint", "left_hip_yaw_joint",
    "left_knee_joint", "left_ankle_pitch_joint", "left_ankle_roll_joint",
    "right_hip_pitch_joint", "right_hip_roll_joint", "right_hip_yaw_joint",
    "right_knee_joint", "right_ankle_pitch_joint", "right_ankle_roll_joint",
    "waist_yaw_joint", "waist_roll_joint", "waist_pitch_joint",
    "left_shoulder_pitch_joint", "left_shoulder_roll_joint",
    "left_shoulder_yaw_joint", "left_elbow_joint",
    "left_wrist_roll_joint", "left_wrist_pitch_joint", "left_wrist_yaw_joint",
    "right_shoulder_pitch_joint", "right_shoulder_roll_joint",
    "right_shoulder_yaw_joint", "right_elbow_joint",
    "right_wrist_roll_joint", "right_wrist_pitch_joint", "right_wrist_yaw_joint",
)
_LEVEL = 0.0
"""GRAIL has no difficulty axis. One row, stated rather than faked — a row that
does not mean obstacle height is a curriculum that promotes nothing."""


class GrailSource:
    """GRAIL terrain categories, read in place."""

    name = "grail"
    default_root = "PhysicalAI-Robotics-Locomanipulation-GRAIL/data"

    def __init__(
        self,
        root: str | Path,
        families: tuple[str, ...] | None = None,
        levels: tuple[float, ...] | None = None,
        categories: tuple[str, ...] = ("curb", "stair1", "stair2"),
    ) -> None:
        del levels  # no difficulty axis
        self.root = Path(root)
        self.families = set(families) if families else None
        self.categories = tuple(
            c for c in categories if (self.root / c / "robot").is_dir()
        )
        if not self.categories:
            raise FileNotFoundError(f"no populated GRAIL category under {self.root}")

    def _takes(self) -> list[tuple[str, str, Path]]:
        """(family, take path stem, robot pkl) for everything the roster wants."""
        out = []
        for cat in self.categories:
            for pkl in sorted((self.root / cat / "robot").glob("*.pkl")):
                parts = pkl.stem.split("__")
                if len(parts) != 3:
                    continue
                family = parts[1]
                if self.families is None or family in self.families:
                    out.append((family, pkl.stem, pkl))
        return out

    def _boxes_of(self, stem: str) -> tuple[BoxSpec, ...]:
        from pxr import Usd, UsdGeom

        cat = stem.split("__")[0].replace("terrain_", "").rstrip("s")
        usd = next(
            (p for c in self.categories
             if (p := self.root / c / "object_usd" / f"{stem}.usd").exists()),
            None,
        )
        if usd is None:
            raise FileNotFoundError(f"no object_usd for {stem} (looked for {cat})")

        stage = Usd.Stage.Open(str(usd))  # keep alive while prims are read
        pts = [np.array(UsdGeom.Mesh(p).GetPointsAttr().Get(), float)
               for p in stage.Traverse() if p.IsA(UsdGeom.Mesh)]
        v = np.concatenate(pts)
        if len(v) % _VERTS_PER_BOX:
            raise ValueError(
                f"{usd.name}: {len(v)} verts is not a whole number of "
                f"{_VERTS_PER_BOX}-vertex boxes")
        return tuple(box_from_vertices(v[i:i + _VERTS_PER_BOX])
                     for i in range(0, len(v), _VERTS_PER_BOX))

    # ── the protocol ──

    def tiles(self) -> Iterable[TileSpec]:
        """One tile per terrain — geometry is shared across that terrain's takes."""
        seen: set[str] = set()
        for family, stem, _ in self._takes():
            if family in seen:
                continue
            seen.add(family)
            yield TileSpec(family=family, level=_LEVEL, boxes=self._boxes_of(stem))

    def clips(self) -> Iterable[ClipSpec]:
        import joblib

        for family, stem, pkl in self._takes():
            d = joblib.load(pkl)
            key = next(iter(d))
            r = d[key]
            quat_xyzw = np.asarray(r["root_rot"], float)
            yield ClipSpec(
                name=stem,
                root_pos=np.asarray(r["root_trans_offset"], float),
                root_quat=quat_xyzw[:, [3, 0, 1, 2]],  # xyzw -> wxyz
                joint_pos=np.asarray(r["dof"], float),
                joint_names=_JOINT_NAMES,
                fps=float(r["fps"]),
                family=family,
                level=_LEVEL,
                meta={"source": f"grail/{stem.split('__')[0]}",
                      "take": stem, "clip_key": key, "src_fps": float(r["fps"])},
            )
