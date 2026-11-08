"""Failure-biased clip/phase sampling, following BeyondMimic and MJLab.

Each clip owns its time bins: lookback never crosses reference boundaries.
Statistics are transient EMA failure counts, not exposure-normalized rates.
See docs/hoi_suitcase.md for attribution and the training recipe.
"""

from __future__ import annotations

import math

import torch


class AdaptiveMotionSampler:
    """Joint clip/bin weights with an independent uniform full-start mixture."""

    def __init__(
        self, lengths: torch.Tensor, step_dt: float, *, kernel_size: int = 1,
        decay: float = 0.8, uniform_prior: float = 0.1, alpha: float = 0.001,
        start_fraction: float = 0.25,
    ) -> None:
        if lengths.ndim != 1 or not len(lengths) or (lengths < 1).any():
            raise ValueError("Adaptive sampling requires nonempty, positive clip lengths")
        if not math.isfinite(step_dt) or step_dt <= 0 or kernel_size < 1:
            raise ValueError("Adaptive sampling requires positive step_dt and kernel_size")
        if not 0 <= decay <= 1 or not 0 < alpha <= 1:
            raise ValueError("Adaptive decay must be in [0, 1], alpha in (0, 1]")
        if not math.isfinite(uniform_prior) or uniform_prior <= 0:
            raise ValueError("Adaptive uniform prior must be positive and finite")
        if not 0 <= start_fraction <= 1:
            raise ValueError("Adaptive start fraction must be in [0, 1]")
        self.lengths = lengths
        self.frames_per_bin = max(1, round(1 / step_dt))
        self.bin_counts = (lengths + self.frames_per_bin - 1) // self.frames_per_bin
        bins = torch.arange(int(self.bin_counts.max()), device=lengths.device)
        self.bin_starts = bins * self.frames_per_bin
        self.valid = bins[None, :] < self.bin_counts[:, None]
        self.failed = torch.zeros(self.valid.shape, device=lengths.device)
        self.pending = torch.zeros_like(self.failed)
        self.prior = uniform_prior / int(self.valid.sum())
        self.alpha = alpha
        self.start_fraction = start_fraction
        kernel = torch.tensor([decay**i for i in range(kernel_size)], device=lengths.device)
        self.kernel = kernel / kernel.sum()
        # Replicate each clip's final bin independently, matching upstream's
        # right-edge padding without leaking into another clip or padded bins.
        self.lookahead = [torch.minimum(bins[None, :] + i, self.bin_counts[:, None] - 1)
                          for i in range(kernel_size)]

    def record_failures(self, clips: torch.Tensor, local_frames: torch.Tensor) -> None:
        bins = local_frames // self.frames_per_bin
        bins = torch.minimum(bins, self.bin_counts[clips] - 1)
        flat = clips * self.failed.shape[1] + bins
        self.pending.view(-1).scatter_add_(0, flat, torch.ones_like(flat, dtype=torch.float))

    def advance(self) -> None:
        """Advance once per control step, including steps without failures."""
        self.failed.lerp_(self.pending, self.alpha)
        self.pending.zero_()

    def weights(self) -> torch.Tensor:
        scores = sum(k * self.failed.gather(1, idx)
                     for k, idx in zip(self.kernel, self.lookahead, strict=True))
        return (scores + self.prior) * self.valid

    def sample(
        self, count: int, *, allowed: torch.Tensor | None = None,
        phase_max: float = 1.0, start_from_zero: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Return clips, local frames and mean clip probabilities for this batch.

        At full phase range this factorizes one joint clip/bin distribution.
        Annealing restricts conditional phase sampling only: failures later in
        a clip still inform its difficulty when every start must be frame zero.
        """
        weights = self.weights()
        clip_weights = weights.sum(1)
        uniform = torch.ones(count, len(self.lengths), device=self.lengths.device)
        if allowed is not None:
            if allowed.shape != uniform.shape or not allowed.bool().any(1).all():
                raise ValueError("Every environment must allow at least one motion")
            uniform *= allowed.bool()
        uniform /= uniform.sum(1, keepdim=True)
        adaptive = uniform * clip_weights[None, :]
        adaptive /= adaptive.sum(1, keepdim=True)
        probabilities = ((1 - self.start_fraction) * adaptive
                         + self.start_fraction * uniform).mean(0)
        full_start = torch.rand(count, device=self.lengths.device) < self.start_fraction
        clip_probs = torch.where(full_start[:, None], uniform, adaptive)
        clips = torch.multinomial(clip_probs, 1).squeeze(1)

        if start_from_zero or phase_max <= 0:
            return clips, torch.zeros_like(clips), probabilities
        limits = (self.lengths.float() * min(phase_max, 1.0)).long().clamp(min=1)
        widths = (limits[:, None] - self.bin_starts[None, :]).clamp(0, self.frames_per_bin)
        # A partially eligible bin receives only its eligible fraction of mass.
        original_widths = (self.lengths[:, None] - self.bin_starts[None, :]).clamp(1, self.frames_per_bin)
        phase_weights = weights * widths / original_widths
        bins = torch.multinomial(phase_weights[clips], 1).squeeze(1)
        frames = self.bin_starts[bins] + (
            torch.rand(count, device=self.lengths.device) * widths[clips, bins]
        ).long()
        frames[full_start] = 0
        return clips, frames, probabilities
