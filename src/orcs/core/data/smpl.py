"""The SMPL human-reference channel: what it means, how it loads, how it draws.

Two command spaces carry it (uolm's `-Smpl`, perloco's) and both feed the SAME
frozen SONIC smpl encoder, so the contract lives here once.

    smpl_joints       (T, 24, 3)  **z-up**, root-centred, root rotation
                                  APPLIED. Reaches the encoder RAW — SONIC
                                  converts the root quat, never the joints.
    smpl_root_quat_w  (T, 4)      wxyz, **z-up** world, SMPL base rot removed.
    smpl_joints_viz_w (T, 24, 3)  z-up world, translated. Ghost only.

The first two are bit-coupled to `smpl_ported.pt`: the encoder's op-chain is
`quat_apply_inverse(root_q, joints)`, so the pair must land in ONE frame. Both
z-up, and the result is a stable body frame; y-up joints against a z-up root
give one that rotates with heading. Nothing errors either way — it silently
changes what the 72 leading dims mean, and zero-shot tracking reward on 63 GRAIL
curb clips reads 4.735 (z-up) against 0.348 (y-up), g1 encoder 5.677.
`perloco/sources/smplx_fk.py` has the measurement; mocke's tokenizer docstring
and uolm's `load_smpl_clip` both still say y-up and are both wrong.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch

from orcs.core.data.scan import load_field_or_make_zeros

__all__ = ["SMPL_PARENTS", "load_smpl_channels", "draw_smpl_ghost"]

SMPL_PARENTS = (
    -1, 0, 0, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 9, 9, 12, 13, 14, 16, 17, 18, 19,
    20, 21,
)
"""SMPL-24 kinematic tree — the stick figure's bones. Root's parent is -1."""

_GHOST_COLOR = (0.2, 0.8, 0.9, 0.6)


def load_smpl_channels(
    sample_dir: Path, n_frames: int, device: str
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """`smpl_motion.npz` -> (joints, root_quat, joints_viz), zeros when absent.

    Absent is legal: a robot-space clip has no SMPL half, and the channel is
    then never read. Identity quats rather than zeros so a stray consumer gets
    a valid rotation instead of a NaN generator.
    """
    f = sample_dir / "smpl_motion.npz"
    d = np.load(f) if f.exists() else None

    joints, _ = load_field_or_make_zeros(d, "smpl_joints", (n_frames, 24, 3), device)
    quat, has_quat = load_field_or_make_zeros(d, "smpl_root_quat_w", (n_frames, 4), device)
    if not has_quat:
        quat[:, 0] = 1.0
    viz, has_viz = load_field_or_make_zeros(
        d, "smpl_joints_viz_w", (n_frames, 24, 3), device)
    return joints, quat, viz if has_viz else joints


def draw_smpl_ghost(
    visualizer,
    joints_w: np.ndarray,
    rotation_matrix: np.ndarray,
    label: str,
    color: tuple[float, float, float, float] = _GHOST_COLOR,
) -> None:
    """24-joint stick figure + root frame, at `joints_w` (z-up world).

    Spheres and bones, no body model or LBS: the reference IS joint positions,
    and a skinned mesh would render an interpolation the encoder never sees.
    """
    for j, parent in enumerate(SMPL_PARENTS):
        visualizer.add_sphere(
            center=joints_w[j], radius=0.03, color=color, label=f"{label}_j{j}")
        if parent >= 0:
            visualizer.add_cylinder(
                start=joints_w[parent], end=joints_w[j], radius=0.012,
                color=color, label=f"{label}_b{j}")
    visualizer.add_frame(
        position=joints_w[0], rotation_matrix=rotation_matrix, scale=0.25,
        label=f"{label}_root", axis_radius=0.004)
