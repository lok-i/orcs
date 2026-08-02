"""ConcatMotionLoader — every clip on one timeline.

Presents mjlab's `MotionLoader` interface (`joint_pos`, `body_pos_w`, ...) so
`MotionCommand`'s properties — ghost viz, body tracking — work unchanged, and
adds the per-clip boundaries a multi-clip command needs for frame clamping.

The ROBOT half is here and is all any task needs to track a G1. Per-task
channels (object pose, SMPL joints, a contact schedule, a terrain id) ride the
same timeline via two hooks:

    _load_extra(sample_dir, npz, n_frames)   once per clip, accumulate
    _finalize_extra()                        once, concatenate

Joint columns are permuted IL (IsaacLab BFS, the dataset's order) -> MJ
(MuJoCo XML DFS, the sim's order) on load, and bodies are sliced to the 14
tracked ones, so everything downstream is in sim order.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from mocke.mdp.joint_maps import G1_TRACKED_BODIES as _G1_TRACKED_BODIES
from mocke.mdp.joint_maps import IL2MJ as _IL2MJ

from orcs.core.data.scan import last_scan, scan_flat

__all__ = ["ConcatMotionLoader"]

_IL_BODY_IDS = [idx for _, idx in _G1_TRACKED_BODIES]


class ConcatMotionLoader:
    """All clips concatenated into one timeline.

    `motion_files=None` scans `dataset_dir` flat; an explicit list is loaded
    verbatim, so clip index i corresponds to motion_files[i] — which is what
    lets a caller build an id space (object roster, terrain tile) over it.
    """

    tag: str = "orcs"
    """Prefix for the one summary line printed at env build."""

    def __init__(
        self,
        dataset_dir: str | list[str],
        device: str | torch.device,
        motion_files: list[str] | None = None,
        **kwargs,
    ) -> None:
        root = dataset_dir  # display only (for the summary line below)
        if motion_files is None:
            motion_files = scan_flat(dataset_dir)

        self.device = device
        self.motion_files = list(motion_files)
        self._init_extra(**kwargs)

        all_jp: list[torch.Tensor] = []
        all_jv: list[torch.Tensor] = []
        all_bp: list[torch.Tensor] = []
        all_bq: list[torch.Tensor] = []
        all_blv: list[torch.Tensor] = []
        all_bav: list[torch.Tensor] = []
        clip_lengths: list[int] = []

        for mf_str in motion_files:
            mf = Path(mf_str)
            d = np.load(mf)
            n_frames = d["joint_pos"].shape[0]

            def _t(a: np.ndarray) -> torch.Tensor:
                return torch.tensor(a, dtype=torch.float32, device=device)

            all_jp.append(_t(d["joint_pos"])[:, _IL2MJ])
            all_jv.append(_t(d["joint_vel"])[:, _IL2MJ])
            all_bp.append(_t(d["body_pos_w"][:, _IL_BODY_IDS]))
            all_bq.append(_t(d["body_quat_w"][:, _IL_BODY_IDS]))
            all_blv.append(_t(d["body_lin_vel_w"][:, _IL_BODY_IDS]))
            all_bav.append(_t(d["body_ang_vel_w"][:, _IL_BODY_IDS]))

            self._load_extra(mf.parent, d, n_frames)
            clip_lengths.append(n_frames)

        # ── MotionLoader-compatible tensors ──
        self.joint_pos = torch.cat(all_jp)           # (T_tot, 29)
        self.joint_vel = torch.cat(all_jv)           # (T_tot, 29)
        self.body_pos_w = torch.cat(all_bp)          # (T_tot, 14, 3)
        self.body_quat_w = torch.cat(all_bq)         # (T_tot, 14, 4)
        self.body_lin_vel_w = torch.cat(all_blv)     # (T_tot, 14, 3)
        self.body_ang_vel_w = torch.cat(all_bav)     # (T_tot, 14, 3)
        self.time_step_total: int = self.joint_pos.shape[0]

        # ── clip boundaries (for frame clamping) ──
        self.clip_lengths = torch.tensor(
            clip_lengths, device=device, dtype=torch.long
        )
        self.clip_offsets = torch.zeros(
            len(clip_lengths), device=device, dtype=torch.long
        )
        if len(clip_lengths) > 1:
            self.clip_offsets[1:] = self.clip_lengths[:-1].cumsum(0)
        self.clip_ends = self.clip_offsets + self.clip_lengths  # exclusive end
        self.n_clips: int = len(clip_lengths)
        self.max_clip_length: int = int(self.clip_lengths.max().item())

        self._finalize_extra()
        self._print_summary(root)

    # ── extension hooks ──

    def _init_extra(self, **kwargs) -> None:
        """Stash subclass config before the per-clip walk. Default: reject
        unknown kwargs rather than swallow a typo'd knob."""
        if kwargs:
            raise TypeError(f"unexpected loader kwargs: {sorted(kwargs)}")

    def _load_extra(self, sample_dir: Path, npz, n_frames: int) -> None:
        """Per clip, in timeline order — accumulate task channels."""

    def _finalize_extra(self) -> None:
        """Once, after clip bounds exist — concatenate what _load_extra gathered."""

    # ── reporting ──

    def _print_summary(self, root) -> None:
        """One line at env BUILD (not at cfg construction — see scan.last_scan).

        MOTION = one demo folder; CLIP = one `sampleN` take of it (a motion
        holds 1..200+). `n_clips` is what the timeline is made of; the
        exclusion tally is per kind because the two are not interchangeable.
        """
        n_motions = len({Path(f).parent.parent for f in self.motion_files})
        dropped = [  # scan-wide (exclusion precedes any roster filter)
            f"{len(v)} {k[len('excluded_'):].rstrip('s')}{'s' * (len(v) > 1)}"
            for k in ("excluded_motions", "excluded_clips")
            if (v := last_scan.get(k))
        ]
        print(
            f"[{self.tag}] {self.n_clips} clips from {n_motions} motions, "
            f"{self.time_step_total} frames, max_len={self.max_clip_length}"
            + (f", dropped {' + '.join(dropped)}" if dropped else "")
            + f" — {root}"
        )
