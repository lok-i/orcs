"""GRAIL's SMPL-X recon -> the SONIC smpl channel. Offline, at staging.

GRAIL ships SMPL-X (`recon/<take>.pkl`: `poses` (T,165), `trans`, `betas`);
SONIC's encoder wants SMPL-24. Three conversions, each silently wrong-able:

  55 SMPL-X joints -> 24 SMPL   first 22 are identity; SMPL's two hand joints
                                are the SMPL-X hand ROOTS (`left_index1` 25,
                                `right_index1` 40).
  global_orient is ALREADY z-up GRAIL bakes the y-up->z-up rotation into it —
                                measured `R(global_orient) @ canonical == world`
                                to 0.0 m, and canonical sits y-up. uolm's Rx90
                                is for SONIC-native pkls and must NOT be
                                applied on top; doing so tilts every clip 90 deg
                                with no error anywhere.
  joints stay z-up              the encoder computes
                                `quat_apply_inverse(root_q, joints)`; z-up
                                joints against a z-up root give a stable body
                                frame, y-up ones give a frame that rotates with
                                heading. **Measured, not reasoned** — mocke's
                                tokenizer docstring says y-up and uolm's
                                `load_smpl_clip` implements y-up, and both are
                                wrong (neither ever ran against real SMPL data;
                                uolm's own header comment says z-up and
                                contradicts its code). 250-step zero-shot
                                tracking reward on the same 63 curb clips:

                                    y-up joints  0.348
                                    z-up joints  4.735
                                    g1 encoder   5.677  (the reference)

                                Do not "restore" the y-up rotation on the
                                strength of a docstring.

`betas` is all-zero in every GRAIL clip (`scale` 1.0), so the neutral template
is the body — `predicted_body_height` is an unused HMR estimate, not a scale.
"""

from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path

import numpy as np

from orcs.core.paths import DEPS_ROOT

__all__ = ["SMPLX_DIR", "smpl_channels", "smplx_dir"]

SMPLX_DIR = "GRAIL/imports/GEM-SMPL/inputs/checkpoints/body_models"
"""SMPL-X body-model dir, relative to DEPS_ROOT. Holds `smplx/SMPLX_*.npz`."""


def smplx_dir() -> Path:
    """Where the licensed SMPL-X models live. `ORCS_SMPLX_DIR` overrides.

    Defaults inside the GRAIL clone because that is where its own installer
    puts them — the override exists so the models are not hostage to a 7 GB
    reference checkout orcs never imports.
    """
    return Path(os.environ.get("ORCS_SMPLX_DIR") or DEPS_ROOT / SMPLX_DIR)

_SMPLX_TO_SMPL24 = tuple(range(22)) + (25, 40)
_BASE_ROT_CONJ = np.array([0.5, -0.5, -0.5, -0.5])  # conj([.5, .5, .5, .5])

# poses (165) = global_orient 3 | body 63 | jaw 3 | eyes 6 | hands 90
_SLICES = {"global_orient": (0, 3), "body_pose": (3, 66), "jaw_pose": (66, 69),
           "leye_pose": (69, 72), "reye_pose": (72, 75),
           "left_hand_pose": (75, 120), "right_hand_pose": (120, 165)}


def _quat_mul(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """wxyz Hamilton product, broadcasting."""
    w1, x1, y1, z1 = np.moveaxis(a, -1, 0)
    w2, x2, y2, z2 = np.moveaxis(b, -1, 0)
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
    return np.concatenate([np.cos(0.5 * angle), axis * np.sin(0.5 * angle)], axis=-1)


@lru_cache(maxsize=2)
def _model(model_dir: str, gender: str):
    import smplx

    # Checked here because `smplx.create` does not check: given a path that is
    # not a directory it INFERS the model type from the basename, so an absent
    # dir surfaces as `ValueError: Unknown model type body` (from
    # "body_models") several frames deep, naming neither the real problem nor
    # the path it wanted.
    want = Path(model_dir) / "smplx" / f"SMPLX_{gender.upper()}.npz"
    if not want.exists():
        raise FileNotFoundError(
            f"SMPL-X body model not found: {want}\n"
            f"  --smpl needs the licensed SMPL-X v1.1 NPZ models. Register and "
            f"download at https://smpl-x.is.tue.mpg.de, unzip so that path "
            f"exists, or point ORCS_SMPLX_DIR elsewhere.\n"
            f"  Nothing else needs them: drop --smpl and every task but "
            f"`Orcs-PerLoco-Grail-AdaptSonic-Smpl` stages and trains.")
    return smplx.create(model_dir, model_type="smplx", gender=gender,
                        use_pca=False, flat_hand_mean=True, batch_size=1)


def smpl_channels(
    recon_pkl: str | Path, model_dir: str | Path
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """-> (joints (S,24,3) root-centred, root_quat (S,4), viz (S,24,3)); all z-up.

    One forward pass: `viz` is the world skeleton the ghost draws, `joints` the
    same thing root-centred for the encoder.
    """
    import joblib
    import torch

    d = joblib.load(recon_pkl)
    rec = d[next(iter(d))] if "poses" not in d else d
    poses = np.asarray(rec["poses"], np.float32)
    trans = np.asarray(rec["trans"], np.float32)
    betas = np.asarray(rec["betas"], np.float32).reshape(1, -1)
    n = len(poses)

    model = _model(str(model_dir), str(rec.get("gender", "neutral")))
    kw = {k: torch.as_tensor(poses[:, a:b]) for k, (a, b) in _SLICES.items()}
    with torch.no_grad():
        out = model(
            betas=torch.as_tensor(np.repeat(betas[:, :model.num_betas], n, 0)),
            transl=torch.as_tensor(trans),
            expression=torch.zeros(n, model.num_expression_coeffs),
            **kw,
        )
    viz = out.joints.numpy()[:, _SMPLX_TO_SMPL24]          # (S, 24, 3) z-up world

    root_q = _aa_to_quat(poses[:, 0:3])                     # already z-up (see header)
    joints = viz - viz[:, :1]                               # z-up, root-centred
    root_q = _quat_mul(root_q, np.broadcast_to(_BASE_ROT_CONJ, root_q.shape))
    return (joints.astype(np.float32), root_q.astype(np.float32),
            viz.astype(np.float32))
