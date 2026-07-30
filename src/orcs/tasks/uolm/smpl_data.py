"""SMPL clip loading + dataset staging for the -Smpl command space.

Converts SONIC-format SMPL clips into the flat dataset layout the UOLM motion
command reads (``<root>/<clip>/<sampleN>/{motion,smpl_motion,object_motion,
contact_matrix}.npz``). Shared by scripts/rollout_smpl.py (single scratch clip)
and scripts/build_smpl_dataset.py (persistent dataset under data/smpl_motions).

Convention (gear_sonic split, docs/references/conventions.md):
  smpl_joints  ship ALREADY z-up, root-centered  -> encoder-exact, stay RAW.
  pose_aa root + transl  are SMPL-native y-up      -> converted to z-up here.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

_SMPL_BASE_ROT_CONJ = np.array([0.5, -0.5, -0.5, -0.5])  # conj([.5,.5,.5,.5])
_RX90 = np.array([np.cos(np.pi / 4), np.sin(np.pi / 4), 0.0, 0.0])  # +90° about X
_IL_TRACKED_BODIES = 35  # covers all IL tracked-body indices in body_pos_w


def _quat_mul(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """wxyz hamilton product, (N,4)x(N,4)->(N,4)."""
    w1, x1, y1, z1 = a[..., 0], a[..., 1], a[..., 2], a[..., 3]
    w2, x2, y2, z2 = b[..., 0], b[..., 1], b[..., 2], b[..., 3]
    return np.stack([
        w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
        w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
        w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
        w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
    ], axis=-1)


def _aa_to_quat(aa: np.ndarray) -> np.ndarray:
    """axis-angle (N,3) -> wxyz quat (N,4)."""
    angle = np.linalg.norm(aa, axis=-1, keepdims=True)
    axis = np.where(angle > 1e-8, aa / np.maximum(angle, 1e-8), 0.0)
    half = 0.5 * angle
    return np.concatenate([np.cos(half), axis * np.sin(half)], axis=-1)


def synthetic_clip(T: int = 250) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """A static ~1.7 m standing skeleton (z-up world), for smoke tests."""
    j = np.zeros((24, 3), dtype=np.float32)
    z = {0: 0.95, 1: 0.85, 2: 0.85, 3: 1.05, 4: 0.50, 5: 0.50, 6: 1.15,
         7: 0.10, 8: 0.10, 9: 1.25, 10: 0.05, 11: 0.05, 12: 1.45,
         13: 1.35, 14: 1.35, 15: 1.60, 16: 1.35, 17: 1.35, 18: 1.05,
         19: 1.05, 20: 0.80, 21: 0.80, 22: 0.72, 23: 0.72}
    y = {1: 0.10, 2: -0.10, 4: 0.11, 5: -0.11, 7: 0.12, 8: -0.12,
         10: 0.12, 11: -0.12, 13: 0.08, 14: -0.08, 16: 0.20, 17: -0.20,
         18: 0.24, 19: -0.24, 20: 0.26, 21: -0.26, 22: 0.27, 23: -0.27}
    for k, v in z.items():
        j[k, 2] = v
    for k, v in y.items():
        j[k, 1] = v
    joints = np.repeat(j[None], T, axis=0)
    quat = np.zeros((T, 4), dtype=np.float32)
    quat[:, 0] = 1.0
    return joints, quat, joints  # authored z-up world already


def load_smpl_clip(
    path: str | None, z_up: bool = False
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """-> (smpl_joints RAW (T,24,3), smpl_root_quat_w (T,4), joints_viz_w (T,24,3)).

    RAW joints are encoder-exact (SONIC never converts them). root_quat is z-up,
    wxyz, base-rot removed. joints_viz_w = (joints + transl) in z-up world —
    ghost + RSI only. ``path=None`` returns a synthetic standing clip.
    ``.npz`` inputs are passed through (must already match this contract).
    """
    if path is None:
        return synthetic_clip()

    if path.endswith(".npz"):
        d = np.load(path)
        joints = d["smpl_joints"].astype(np.float32)
        viz = (d["smpl_joints_viz_w"] if "smpl_joints_viz_w" in d
               else d["smpl_joints"]).astype(np.float32)
        return joints, d["smpl_root_quat_w"].astype(np.float32), viz

    import joblib
    d = joblib.load(path)  # SONIC smpl pkl (pose_aa, transl, smpl_joints, fps)
    joints = np.asarray(d["smpl_joints"], dtype=np.float32)  # RAW — encoder-exact
    transl = np.asarray(d.get("transl", np.zeros((len(joints), 3))), dtype=np.float32)
    root_q = _aa_to_quat(np.asarray(d["pose_aa"], dtype=np.float32)[:, :3])
    if not z_up:  # only pose_aa root + transl are y-up (joints already z-up)
        transl = np.stack([transl[..., 0], -transl[..., 2], transl[..., 1]], axis=-1)
        root_q = _quat_mul(np.broadcast_to(_RX90, root_q.shape), root_q)
    root_q = _quat_mul(root_q, np.broadcast_to(_SMPL_BASE_ROT_CONJ, root_q.shape))
    world = joints + transl[:, None, :]
    world[..., 2] -= world[..., 2].min()  # rest on the ground plane
    return joints, root_q.astype(np.float32), world.astype(np.float32)


def stage_clip(
    sample_dir: Path,
    joints: np.ndarray,
    root_quat: np.ndarray,
    joints_viz: np.ndarray,
    object_npz: str | None = None,
) -> None:
    """Write ONE sample dir: motion + smpl_motion + object_motion + contact npzs.

    motion.npz is a placeholder robot ref (IL order) — RSI seed only: root
    follows the smpl world path at standing height, joints/vels zero (wrist refs
    zero -> valid, degraded wrist orientation only). object_motion is a static
    nominal pose when ``object_npz`` is None (no smpl+object clips exist yet).
    """
    from orcs.tasks.uolm.env_cfg import _CONTACT_GRAPH_BODY_NAMES

    T = joints.shape[0]
    sample_dir.mkdir(parents=True, exist_ok=True)

    body_pos = np.zeros((T, _IL_TRACKED_BODIES, 3), dtype=np.float32)
    body_pos[:, :, :2] = joints_viz[:, :1, :2]  # all bodies at the smpl world xy
    body_pos[:, 0, 2] = 0.793  # pelvis standing height
    body_quat = np.zeros((T, _IL_TRACKED_BODIES, 4), dtype=np.float32)
    body_quat[..., 0] = 1.0
    np.savez(
        sample_dir / "motion.npz",
        joint_pos=np.zeros((T, 29), dtype=np.float32),
        joint_vel=np.zeros((T, 29), dtype=np.float32),
        body_pos_w=body_pos,
        body_quat_w=body_quat,
        body_lin_vel_w=np.zeros((T, _IL_TRACKED_BODIES, 3), dtype=np.float32),
        body_ang_vel_w=np.zeros((T, _IL_TRACKED_BODIES, 3), dtype=np.float32),
    )

    np.savez(sample_dir / "smpl_motion.npz",
             smpl_joints=joints,               # RAW (encoder-exact)
             smpl_root_quat_w=root_quat,       # z-up, wxyz, base rot removed
             smpl_joints_viz_w=joints_viz)     # z-up world (ghost only)

    if object_npz is not None:
        d = np.load(object_npz)
        np.savez(sample_dir / "object_motion.npz", **{k: d[k] for k in d.files})
    else:  # static nominal pose in front of the human (placeholder)
        obj_pos = np.zeros((T, 3), dtype=np.float32)
        obj_pos[:] = (1.2, 0.0, 0.3)
        obj_quat = np.zeros((T, 4), dtype=np.float32)
        obj_quat[:, 0] = 1.0
        np.savez(sample_dir / "object_motion.npz",
                 obj_pos_w=obj_pos, obj_quat_w=obj_quat)

    # zeros contact matrix — establishes the ContactSchedule legend
    names = list(_CONTACT_GRAPH_BODY_NAMES) + ["object", "world"]
    np.savez(sample_dir / "contact_matrix.npz",
             body_names=np.array(names),
             matrix=np.zeros((T, len(names), len(names)), dtype=np.int8))
