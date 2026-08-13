"""Normalize the in-house reconstructed SMPL-H/object corpus for UOLM.

The source repository stays immutable.  Its human and object streams are
converted into the task-blind ORCS channels under a local generated cache:

  reconstructed_motions/drcl/<collection>/<interaction>/<clip>/
      motion.npz + object_motion.npz

  smpl_motions/uolm/reconstructed/<motion-set>/<interaction>/<clip>/
      smpl_motion.npz + object_motion.npz + metadata.json [+ seed_state.npz]

Only this adapter knows the source hierarchy or field names.  Retargeting,
viewing, and training consume the normalized cache.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

import numpy as np

from orcs.core.paths import DATA_ROOT, DEPS_ROOT

__all__ = [
    "MOTION_SETS",
    "MotionSetSpec",
    "cache_motion_files",
    "cache_root",
    "is_current_cache_sample",
    "source_clips",
    "stage_motion_set",
    "stage_source_clip",
]

_TARGET_FPS = 50.0
_SMPLH_TO_SMPL24 = tuple(range(22)) + (25, 40)
_BASE_ROT_CONJ = np.array([0.5, -0.5, -0.5, -0.5], dtype=np.float32)
_CACHE_SCHEMA_VERSION = 2
_GROUND_FOOT_JOINTS = (10, 11)
_GROUND_QUANTILE = 0.05


@dataclass(frozen=True)
class MotionSetSpec:
    name: str
    source_root: str
    object_asset: str
    support: str
    object_half_extent: float


MOTION_SETS: dict[str, MotionSetSpec] = {
    "small-cube-table": MotionSetSpec(
        name="small-cube-table",
        source_root="drcl/Box",
        object_asset="custom_objects/box",
        support="table",
        object_half_extent=0.18,
    ),
    "big-cube-floor": MotionSetSpec(
        name="big-cube-floor",
        source_root="drcl/LargeCube",
        object_asset="custom_objects/cube",
        support="floor",
        object_half_extent=0.3048,
    ),
}


def source_root() -> Path:
    return DATA_ROOT / "reconstructed_motions"


def cache_root() -> Path:
    return DATA_ROOT / "smpl_motions/uolm/reconstructed"


def cache_motion_files(
    motion_set: str,
    interaction_names: tuple[str, ...] | None = None,
    *,
    root: Path | None = None,
) -> list[Path]:
    """Normalized SMPL files selected by their interaction directory."""
    motion_set_root = (root or cache_root()) / _spec(motion_set).name
    files = sorted(motion_set_root.rglob("smpl_motion.npz"))
    if interaction_names is None:
        return files
    selected = set(interaction_names)
    return [path for path in files if path.parent.parent.name in selected]


def _spec(motion_set: str) -> MotionSetSpec:
    try:
        return MOTION_SETS[motion_set]
    except KeyError as exc:
        raise ValueError(
            f"unknown UOLM motion set {motion_set!r}; choose one of "
            f"{tuple(MOTION_SETS)}"
        ) from exc


def source_clips(motion_set: str) -> list[Path]:
    """All source clips in stable interaction/name order."""
    spec = _spec(motion_set)
    root = source_root() / spec.source_root
    if not root.is_dir():
        raise FileNotFoundError(
            f"reconstructed motion set {motion_set!r} not found at {root}; "
            "sync the reconstructed_motions entry from deps.lock"
        )
    clips = sorted(
        path.parent
        for path in root.glob("*/*/motion.npz")
        if path.parent.parent.name != "object"
    )
    if not clips:
        raise FileNotFoundError(f"no reconstructed clips under {root}")
    return clips


def _cache_sample(source: Path, motion_set: str) -> Path:
    root = source_root() / _spec(motion_set).source_root
    relative = source.resolve().relative_to(root.resolve())
    if len(relative.parts) != 2:
        raise ValueError(
            f"expected <interaction>/<clip> below {root}, got {relative}"
        )
    return cache_root() / motion_set / relative


def _quat_mul(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    w1, x1, y1, z1 = np.moveaxis(a, -1, 0)
    w2, x2, y2, z2 = np.moveaxis(b, -1, 0)
    return np.stack(
        (
            w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
            w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
            w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
            w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
        ),
        axis=-1,
    )


def _quat_conjugate(q: np.ndarray) -> np.ndarray:
    out = q.copy()
    out[..., 1:] *= -1.0
    return out


def _aa_to_quat(aa: np.ndarray) -> np.ndarray:
    angle = np.linalg.norm(aa, axis=-1, keepdims=True)
    axis = np.divide(aa, angle, out=np.zeros_like(aa), where=angle > 1e-8)
    return np.concatenate((np.cos(0.5 * angle), axis * np.sin(0.5 * angle)), axis=-1)


def _quat_to_rotvec(q: np.ndarray) -> np.ndarray:
    q = q / np.linalg.norm(q, axis=-1, keepdims=True).clip(1e-8)
    q = np.where(q[..., :1] < 0.0, -q, q)
    sin_half = np.linalg.norm(q[..., 1:], axis=-1, keepdims=True)
    angle = 2.0 * np.arctan2(sin_half, q[..., :1].clip(-1.0, 1.0))
    axis = np.divide(
        q[..., 1:], sin_half, out=np.zeros_like(q[..., 1:]), where=sin_half > 1e-8
    )
    return axis * angle


def _timeline(num_frames: int, source_fps: float) -> tuple[np.ndarray, np.ndarray]:
    duration = (num_frames - 1) / source_fps
    # Keep an exact 50 Hz grid and never extrapolate past the authored clip.
    target_frames = int(np.floor(duration * _TARGET_FPS + 1e-9)) + 1
    return (
        np.arange(num_frames, dtype=np.float64) / source_fps,
        np.arange(target_frames, dtype=np.float64) / _TARGET_FPS,
    )


def _interp(values: np.ndarray, source_t: np.ndarray, target_t: np.ndarray) -> np.ndarray:
    flat = values.reshape(len(values), -1)
    out = np.stack(
        [np.interp(target_t, source_t, flat[:, column]) for column in range(flat.shape[1])],
        axis=-1,
    )
    return out.reshape((len(target_t),) + values.shape[1:])


def _slerp(quat: np.ndarray, source_t: np.ndarray, target_t: np.ndarray) -> np.ndarray:
    quat = quat.astype(np.float64, copy=True)
    quat /= np.linalg.norm(quat, axis=-1, keepdims=True).clip(1e-8)
    for index in range(1, len(quat)):
        if np.dot(quat[index - 1], quat[index]) < 0.0:
            quat[index] *= -1.0
    right = np.searchsorted(source_t, target_t, side="right").clip(1, len(source_t) - 1)
    left = right - 1
    span = (source_t[right] - source_t[left]).clip(1e-12)
    alpha = ((target_t - source_t[left]) / span)[:, None]
    q0, q1 = quat[left], quat[right]
    dot = np.sum(q0 * q1, axis=-1, keepdims=True).clip(-1.0, 1.0)
    theta = np.arccos(dot)
    sin_theta = np.sin(theta)
    linear = (1.0 - alpha) * q0 + alpha * q1
    curved = (
        np.sin((1.0 - alpha) * theta) / np.maximum(sin_theta, 1e-8) * q0
        + np.sin(alpha * theta) / np.maximum(sin_theta, 1e-8) * q1
    )
    out = np.where(np.abs(sin_theta) < 1e-6, linear, curved)
    out /= np.linalg.norm(out, axis=-1, keepdims=True).clip(1e-8)
    return out


def _velocities(pos: np.ndarray, quat: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    lin = np.zeros_like(pos, dtype=np.float32)
    ang = np.zeros((len(quat), 3), dtype=np.float32)
    if len(pos) > 1:
        lin[1:] = np.diff(pos, axis=0) * _TARGET_FPS
        lin[0] = lin[1]
        delta = _quat_mul(quat[1:], _quat_conjugate(quat[:-1]))
        ang[1:] = _quat_to_rotvec(delta) * _TARGET_FPS
        ang[0] = ang[1]
    return lin, ang


def _ground_human_to_floor(joints_w: np.ndarray) -> tuple[np.ndarray, float]:
    """Remove a clip-wide reconstruction height bias from the human only.

    Reconstructed human and object streams do not necessarily share the same
    calibrated vertical origin.  The object stream already encodes its
    physical support (floor or table), so translating the whole scene would
    break that support.  Instead, estimate the human floor from the lower of
    the two ankles over time and translate all human joints by that one scalar.

    A low quantile is used instead of a raw minimum so one noisy ankle sample
    cannot shift an otherwise valid clip.  True flight phases remain intact:
    this removes only a constant clip-level bias.
    """
    if joints_w.ndim != 3 or joints_w.shape[1] <= max(_GROUND_FOOT_JOINTS):
        raise ValueError(
            "human joints must have shape (frames, joints>=12, xyz), got "
            f"{joints_w.shape}"
        )
    ankle_floor = np.min(joints_w[:, _GROUND_FOOT_JOINTS, 2], axis=1)
    ground_offset_z = float(np.quantile(ankle_floor, _GROUND_QUANTILE))
    if not np.isfinite(ground_offset_z):
        raise ValueError("cannot infer a finite human ground height")
    grounded = joints_w.copy()
    grounded[..., 2] -= ground_offset_z
    return grounded, ground_offset_z


def _native_smplh_root() -> Path | None:
    override = os.environ.get("ORCS_SMPLH_DIR")
    candidates = [Path(override).expanduser()] if override else []
    candidates.append(DEPS_ROOT / "body_models")
    for candidate in candidates:
        if (candidate / "smplh").is_dir():
            return candidate
    return None


@lru_cache(maxsize=6)
def _body_model(gender: str):
    import smplx

    smplh_root = _native_smplh_root()
    if smplh_root is not None:
        return (
            smplx.create(
                str(smplh_root), model_type="smplh", gender=gender,
                use_pca=False, flat_hand_mean=False, batch_size=1,
            ),
            "smplh",
        )

    # The shared SMPL-X body family has the same 21 body joints, hand roots,
    # gender, and beta/pose parameterization needed by the 24-point command.
    # It is a deliberate FK-only compatibility fallback when the licensed
    # SMPL-H files are not installed; no mesh from it is persisted.
    fallback = DEPS_ROOT / "GRAIL/imports/GEM-SMPL/inputs/checkpoints/body_models"
    wanted = fallback / "smplx" / f"SMPLX_{gender.upper()}.npz"
    if not wanted.exists():
        raise FileNotFoundError(
            "SMPL-H body models are unavailable. Set ORCS_SMPLH_DIR to a body-model "
            f"root containing smplh/SMPLH_*.pkl; compatibility fallback also "
            f"missing at {wanted}."
        )
    print(
        "[reconstructed] ORCS_SMPLH_DIR not set; using the compatible SMPL-X "
        f"24-joint FK fallback for {gender} clips"
    )
    return (
        smplx.create(
            str(fallback), model_type="smplx", gender=gender,
            use_pca=False, flat_hand_mean=False, batch_size=1,
        ),
        "smplx-fallback",
    )


def _smpl_channels(
    source_file: Path,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, float, str, float]:
    import torch

    with np.load(source_file, allow_pickle=False) as data:
        source_fps = float(data["fps"])
        gender = str(data["gender"])
        arrays = {name: data[name].astype(np.float32) for name in (
            "betas", "global_orient", "body_pose", "left_hand_pose",
            "right_hand_pose", "transl",
        )}

    model, evaluator = _body_model(gender)
    kwargs = {name: torch.from_numpy(value) for name, value in arrays.items()}
    if evaluator == "smplx-fallback":
        n = len(arrays["transl"])
        kwargs["expression"] = torch.zeros(n, model.num_expression_coeffs)
        kwargs["jaw_pose"] = torch.zeros(n, 3)
        kwargs["leye_pose"] = torch.zeros(n, 3)
        kwargs["reye_pose"] = torch.zeros(n, 3)
    with torch.no_grad():
        joints_world = model(**kwargs).joints[:, _SMPLH_TO_SMPL24].cpu().numpy()

    root_quat = _aa_to_quat(arrays["global_orient"])
    root_quat = _quat_mul(root_quat, np.broadcast_to(_BASE_ROT_CONJ, root_quat.shape))
    source_t, target_t = _timeline(len(joints_world), source_fps)
    viz = _interp(joints_world, source_t, target_t).astype(np.float32)
    viz, ground_offset_z = _ground_human_to_floor(viz)
    joints = (viz - viz[:, :1]).astype(np.float32)
    root_quat = _slerp(root_quat, source_t, target_t).astype(np.float32)
    return joints, root_quat, viz, source_fps, evaluator, ground_offset_z


def is_current_cache_sample(output: Path) -> bool:
    """Whether a staged sample obeys the current adapter contract."""
    required = (output / "smpl_motion.npz", output / "object_motion.npz")
    metadata_path = output / "metadata.json"
    if not all(path.exists() for path in required) or not metadata_path.exists():
        return False
    try:
        metadata = json.loads(metadata_path.read_text())
        with np.load(required[0], allow_pickle=False) as smpl:
            has_ground_offset = "human_ground_offset_z" in smpl.files
    except (json.JSONDecodeError, KeyError, OSError, ValueError):
        return False
    return (
        metadata.get("schema_version") == _CACHE_SCHEMA_VERSION
        and has_ground_offset
    )


def stage_source_clip(
    source: Path, motion_set: str, *, overwrite: bool = False
) -> Path:
    """Normalize one reconstructed clip and return its cache sample directory."""
    spec = _spec(motion_set)
    output = _cache_sample(source, motion_set)
    if not overwrite and is_current_cache_sample(output):
        return output

    joints, root_quat, viz, source_fps, evaluator, ground_offset_z = _smpl_channels(
        source / "motion.npz"
    )
    with np.load(source / "object_motion.npz", allow_pickle=False) as obj:
        source_pos = obj["obj_pos"].astype(np.float64)
        source_quat = obj["obj_rot"].astype(np.float64)
    source_t, target_t = _timeline(len(source_pos), source_fps)
    object_pos = _interp(source_pos, source_t, target_t).astype(np.float32)
    object_quat = _slerp(source_quat, source_t, target_t).astype(np.float32)
    if len(object_pos) != len(joints):
        raise ValueError(
            f"{source}: human/object frame mismatch after resampling "
            f"({len(joints)} != {len(object_pos)})"
        )
    object_lin_vel, object_ang_vel = _velocities(object_pos, object_quat)

    output.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        output / "smpl_motion.npz",
        smpl_joints=joints,
        smpl_root_quat_w=root_quat,
        smpl_joints_viz_w=viz,
        human_ground_offset_z=np.array(ground_offset_z, dtype=np.float32),
        fps=np.array(_TARGET_FPS, dtype=np.float32),
    )
    np.savez_compressed(
        output / "object_motion.npz",
        obj_pos_w=object_pos,
        obj_quat_w=object_quat,
        obj_lin_vel_w=object_lin_vel,
        obj_ang_vel_w=object_ang_vel,
    )
    metadata = {
        "schema_version": _CACHE_SCHEMA_VERSION,
        "motion_set": motion_set,
        "source": "reconstructed_motions",
        "source_clip": str(source.resolve().relative_to(source_root().resolve())),
        "source_fps": source_fps,
        "fps": _TARGET_FPS,
        "num_frames": len(joints),
        "interaction": source.parent.name,
        "object_asset": spec.object_asset,
        "object_half_extent": spec.object_half_extent,
        "support": spec.support,
        "human_ground_offset_z": ground_offset_z,
        "body_model_evaluator": evaluator,
    }
    (output / "metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")
    return output


def stage_motion_set(motion_set: str, *, overwrite: bool = False) -> list[Path]:
    """Normalize every clip in a named set, preserving stable source order."""
    return [
        stage_source_clip(source, motion_set, overwrite=overwrite)
        for source in source_clips(motion_set)
    ]
