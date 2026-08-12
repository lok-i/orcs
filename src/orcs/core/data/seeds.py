"""On-disk contract for SMPL-assisted kinematic retarget trajectories.

The seed is a simulated robot state trace used only to initialize training; it
is never a robot-space tracking reference.  Every array is indexed by the
original SMPL frame and a failed frame remains present with ``valid=False`` so
source/seed alignment can never silently shift.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

__all__ = ["SEED_SCHEMA_VERSION", "SeedMotion"]

SEED_SCHEMA_VERSION = 1


@dataclass(frozen=True)
class SeedMotion:
    """A one-to-one simulated state trace aligned with an SMPL clip."""

    fps: float
    source_frame_idx: np.ndarray
    robot_root_pos_w: np.ndarray
    robot_root_quat_w: np.ndarray
    robot_root_lin_vel_w: np.ndarray
    robot_root_ang_vel_w: np.ndarray
    joint_pos: np.ndarray
    joint_vel: np.ndarray
    last_action: np.ndarray
    body_pos_w: np.ndarray
    body_quat_w: np.ndarray
    assist_force_w: np.ndarray
    assist_torque_w: np.ndarray
    body_tracking_error: np.ndarray
    assist_saturation: np.ndarray
    valid: np.ndarray
    joint_names: tuple[str, ...]
    body_names: tuple[str, ...]
    source_path: str = ""
    capture_steps: int = 0
    morphology_scale: float = 1.0
    object_pos_w: np.ndarray | None = None
    object_quat_w: np.ndarray | None = None
    object_lin_vel_w: np.ndarray | None = None
    object_ang_vel_w: np.ndarray | None = None
    object_target_pos_w: np.ndarray | None = None
    object_assist_force_w: np.ndarray | None = None
    object_assist_torque_w: np.ndarray | None = None

    def __post_init__(self) -> None:
        self._validate()

    @property
    def num_frames(self) -> int:
        return int(self.source_frame_idx.shape[0])

    @property
    def source_time_s(self) -> np.ndarray:
        return self.source_frame_idx.astype(np.float32) / self.fps

    def _validate(self) -> None:
        if not np.isfinite(self.fps) or self.fps <= 0:
            raise ValueError(f"fps must be positive and finite, got {self.fps}")
        t = self.num_frames
        if self.source_frame_idx.shape != (t,):
            raise ValueError("source_frame_idx must be one-dimensional")
        if not np.array_equal(self.source_frame_idx, np.arange(t)):
            raise ValueError("source_frame_idx must be exactly 0..T-1")

        required = {
            "robot_root_pos_w": (t, 3),
            "robot_root_quat_w": (t, 4),
            "robot_root_lin_vel_w": (t, 3),
            "robot_root_ang_vel_w": (t, 3),
            "joint_pos": (t, len(self.joint_names)),
            "joint_vel": (t, len(self.joint_names)),
            "last_action": (t, len(self.joint_names)),
            "body_pos_w": (t, len(self.body_names), 3),
            "body_quat_w": (t, len(self.body_names), 4),
            "assist_force_w": (t, len(self.body_names), 3),
            "assist_torque_w": (t, len(self.body_names), 3),
            "body_tracking_error": (t, len(self.body_names)),
            "assist_saturation": (t,),
            "valid": (t,),
        }
        for name, shape in required.items():
            actual = getattr(self, name).shape
            if actual != shape:
                raise ValueError(f"{name} has shape {actual}, expected {shape}")

        optional = {
            "object_pos_w": (t, 3),
            "object_quat_w": (t, 4),
            "object_lin_vel_w": (t, 3),
            "object_ang_vel_w": (t, 3),
            "object_target_pos_w": (t, 3),
            "object_assist_force_w": (t, 3),
            "object_assist_torque_w": (t, 3),
        }
        present = [getattr(self, name) is not None for name in optional]
        if any(present) and not all(present):
            missing = [name for name in optional if getattr(self, name) is None]
            raise ValueError(f"object seed channel is partial; missing {missing}")
        for name, shape in optional.items():
            value = getattr(self, name)
            if value is not None and value.shape != shape:
                raise ValueError(f"{name} has shape {value.shape}, expected {shape}")
            if value is not None and not np.isfinite(value).all():
                raise ValueError(f"{name} contains NaN/Inf")

    def save(self, path: str | Path) -> Path:
        """Write a compressed, self-describing ``seed_state.npz``."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        arrays: dict[str, np.ndarray] = {
            "schema_version": np.array(SEED_SCHEMA_VERSION, dtype=np.int64),
            "fps": np.array(self.fps, dtype=np.float32),
            "source_path": np.array(self.source_path),
            "capture_steps": np.array(self.capture_steps, dtype=np.int64),
            "morphology_scale": np.array(self.morphology_scale, dtype=np.float32),
            "source_frame_idx": self.source_frame_idx.astype(np.int64),
            "source_time_s": self.source_time_s,
            "joint_names": np.asarray(self.joint_names),
            "body_names": np.asarray(self.body_names),
        }
        for name in (
            "robot_root_pos_w",
            "robot_root_quat_w",
            "robot_root_lin_vel_w",
            "robot_root_ang_vel_w",
            "joint_pos",
            "joint_vel",
            "last_action",
            "body_pos_w",
            "body_quat_w",
            "assist_force_w",
            "assist_torque_w",
            "body_tracking_error",
            "assist_saturation",
            "valid",
            "object_pos_w",
            "object_quat_w",
            "object_lin_vel_w",
            "object_ang_vel_w",
            "object_target_pos_w",
            "object_assist_force_w",
            "object_assist_torque_w",
        ):
            value = getattr(self, name)
            if value is not None:
                arrays[name] = value
        np.savez_compressed(path, **arrays)
        return path

    @classmethod
    def load(cls, path: str | Path) -> "SeedMotion":
        """Load and validate a kinematic retarget trajectory."""
        path = Path(path)
        with np.load(path) as d:
            version = int(d["schema_version"])
            if version != SEED_SCHEMA_VERSION:
                raise ValueError(
                    f"unsupported seed schema {version}; expected {SEED_SCHEMA_VERSION}"
                )
            kwargs = {
                name: d[name].copy()
                for name in (
                    "source_frame_idx",
                    "robot_root_pos_w",
                    "robot_root_quat_w",
                    "robot_root_lin_vel_w",
                    "robot_root_ang_vel_w",
                    "joint_pos",
                    "joint_vel",
                    "last_action",
                    "body_pos_w",
                    "body_quat_w",
                    "assist_force_w",
                    "assist_torque_w",
                    "body_tracking_error",
                    "assist_saturation",
                    "valid",
                )
            }
            for name in (
                "object_pos_w",
                "object_quat_w",
                "object_lin_vel_w",
                "object_ang_vel_w",
                "object_target_pos_w",
                "object_assist_force_w",
                "object_assist_torque_w",
            ):
                kwargs[name] = d[name].copy() if name in d else None
            return cls(
                fps=float(d["fps"]),
                joint_names=tuple(str(x) for x in d["joint_names"].tolist()),
                body_names=tuple(str(x) for x in d["body_names"].tolist()),
                source_path=str(d["source_path"]),
                capture_steps=int(d["capture_steps"]),
                morphology_scale=float(d["morphology_scale"]),
                **kwargs,
            )
