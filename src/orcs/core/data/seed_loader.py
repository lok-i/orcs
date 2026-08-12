"""Strict seed-backed timeline for SMPL retargeting tasks.

The robot channel comes exclusively from ``seed_state.npz`` and is used for
RSI compatibility with mjlab's motion-command interface.  The tracking source
is the paired ``smpl_motion.npz``; the staged retarget ``motion.npz`` is only a
sample locator supplied by the existing dataset scan.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch

from orcs.core.data.loader import ConcatMotionLoader
from orcs.core.data.seeds import SeedMotion
from orcs.core.data.smpl import load_smpl_channels

__all__ = ["SeededSmplMotionLoader"]


def _name_permutation(
    actual: tuple[str, ...], expected: tuple[str, ...], *, channel: str, path: Path
) -> list[int]:
    if len(set(actual)) != len(actual):
        raise ValueError(f"{path}: duplicate names in seed {channel}: {actual}")
    missing = [name for name in expected if name not in actual]
    extra = [name for name in actual if name not in expected]
    if missing or extra:
        raise ValueError(
            f"{path}: seed {channel} do not match the robot; "
            f"missing={missing}, extra={extra}"
        )
    return [actual.index(name) for name in expected]


def _finite(name: str, value: np.ndarray, path: Path) -> None:
    if not np.isfinite(value).all():
        raise ValueError(f"{path}: seed field {name} contains NaN/Inf")


class SeededSmplMotionLoader(ConcatMotionLoader):
    """Concatenated SMPL source plus one-to-one simulated RSI states."""

    tag = "smpl-seed"

    def __init__(
        self,
        dataset_dir: str,
        device: str | torch.device,
        *,
        motion_files: list[str],
        joint_names: tuple[str, ...],
        body_names: tuple[str, ...],
        expected_fps: float,
    ) -> None:
        self.device = device
        self.motion_files = list(motion_files)
        self.seed_files: list[str] = []

        all_jp: list[torch.Tensor] = []
        all_jv: list[torch.Tensor] = []
        all_bp: list[torch.Tensor] = []
        all_bq: list[torch.Tensor] = []
        all_blv: list[torch.Tensor] = []
        all_bav: list[torch.Tensor] = []
        all_action: list[torch.Tensor] = []
        all_scale: list[torch.Tensor] = []
        all_sj: list[torch.Tensor] = []
        all_sq: list[torch.Tensor] = []
        all_sv: list[torch.Tensor] = []
        clip_lengths: list[int] = []

        def _t(value: np.ndarray) -> torch.Tensor:
            return torch.as_tensor(value, dtype=torch.float32, device=device)

        for motion_file in self.motion_files:
            sample = Path(motion_file).parent
            seed_path = sample / "seed_state.npz"
            smpl_path = sample / "smpl_motion.npz"
            if not seed_path.exists():
                raise FileNotFoundError(f"{sample}: missing seed_state.npz")
            if not smpl_path.exists():
                raise FileNotFoundError(f"{sample}: missing smpl_motion.npz")

            seed = SeedMotion.load(seed_path)
            if not np.isclose(seed.fps, expected_fps, rtol=0.0, atol=1e-4):
                raise ValueError(
                    f"{seed_path}: seed fps {seed.fps:g} != env fps "
                    f"{expected_fps:g}"
                )
            if not seed.valid.all():
                bad = np.flatnonzero(~seed.valid)
                raise ValueError(
                    f"{seed_path}: {len(bad)} invalid RSI frames "
                    f"(first={bad[:5].tolist()})"
                )
            if not np.isfinite(seed.morphology_scale) or seed.morphology_scale <= 0:
                raise ValueError(
                    f"{seed_path}: invalid morphology_scale={seed.morphology_scale}"
                )

            joint_perm = _name_permutation(
                seed.joint_names, joint_names, channel="joint_names", path=seed_path
            )
            body_perm = _name_permutation(
                seed.body_names, body_names, channel="body_names", path=seed_path
            )
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
            ):
                _finite(name, getattr(seed, name), seed_path)

            n_frames = seed.num_frames
            sj, sq, sv = load_smpl_channels(sample, n_frames, str(device))
            if sj.shape[0] != n_frames or sq.shape[0] != n_frames or sv.shape[0] != n_frames:
                raise ValueError(
                    f"{sample}: source/seed frame mismatch (seed={n_frames}, "
                    f"smpl={sj.shape[0]})"
                )
            if not torch.isfinite(sj).all() or not torch.isfinite(sq).all() \
                    or not torch.isfinite(sv).all():
                raise ValueError(f"{smpl_path}: source contains NaN/Inf")

            body_pos = seed.body_pos_w[:, body_perm].copy()
            body_quat = seed.body_quat_w[:, body_perm].copy()
            body_lin_vel = np.zeros_like(body_pos)
            body_ang_vel = np.zeros_like(body_pos)
            pelvis = body_names.index("pelvis")
            # The free-joint state is authoritative for RSI.  Mirror it into
            # the pelvis slot consumed by MotionCommand's anchor interface.
            body_pos[:, pelvis] = seed.robot_root_pos_w
            body_quat[:, pelvis] = seed.robot_root_quat_w
            body_lin_vel[:, pelvis] = seed.robot_root_lin_vel_w
            body_ang_vel[:, pelvis] = seed.robot_root_ang_vel_w

            all_jp.append(_t(seed.joint_pos[:, joint_perm]))
            all_jv.append(_t(seed.joint_vel[:, joint_perm]))
            all_bp.append(_t(body_pos))
            all_bq.append(_t(body_quat))
            all_blv.append(_t(body_lin_vel))
            all_bav.append(_t(body_ang_vel))
            all_action.append(_t(seed.last_action[:, joint_perm]))
            all_scale.append(
                torch.full(
                    (n_frames,),
                    seed.morphology_scale,
                    dtype=torch.float32,
                    device=device,
                )
            )
            all_sj.append(sj.float())
            all_sq.append(sq.float())
            all_sv.append(sv.float())
            clip_lengths.append(n_frames)
            self.seed_files.append(str(seed_path))

        self.joint_pos = torch.cat(all_jp)
        self.joint_vel = torch.cat(all_jv)
        self.body_pos_w = torch.cat(all_bp)
        self.body_quat_w = torch.cat(all_bq)
        self.body_lin_vel_w = torch.cat(all_blv)
        self.body_ang_vel_w = torch.cat(all_bav)
        self.last_action = torch.cat(all_action)
        self.morphology_scale = torch.cat(all_scale)
        self.smpl_joints = torch.cat(all_sj)
        self.smpl_root_quat = torch.cat(all_sq)
        self.smpl_joints_viz = torch.cat(all_sv)
        self.time_step_total = int(self.joint_pos.shape[0])

        self.clip_lengths = torch.tensor(
            clip_lengths, device=device, dtype=torch.long
        )
        self.clip_offsets = torch.zeros(
            len(clip_lengths), device=device, dtype=torch.long
        )
        if len(clip_lengths) > 1:
            self.clip_offsets[1:] = self.clip_lengths[:-1].cumsum(0)
        self.clip_ends = self.clip_offsets + self.clip_lengths
        self.n_clips = len(clip_lengths)
        self.max_clip_length = int(self.clip_lengths.max().item())
        self._print_summary(dataset_dir)
