"""Shared staging interpolation and velocity conventions (no source reader)."""

from __future__ import annotations

import torch
from mjlab.utils.lab_api.math import (
    axis_angle_from_quat,
    quat_conjugate,
    quat_mul,
    quat_slerp,
)


def _lerp(a: torch.Tensor, i0: torch.Tensor, i1: torch.Tensor,
          blend: torch.Tensor) -> torch.Tensor:
    b = blend.reshape(-1, *([1] * (a.dim() - 1)))
    return a[i0] * (1 - b) + a[i1] * b


def _slerp(q: torch.Tensor, i0: torch.Tensor, i1: torch.Tensor,
           blend: torch.Tensor) -> torch.Tensor:
    out = torch.zeros(len(i0), 4, device=q.device)
    for k in range(len(i0)):
        out[k] = quat_slerp(q[i0[k]], q[i1[k]], float(blend[k]))
    return out


def _grid(n_in: int, phase: torch.Tensor) -> tuple[torch.Tensor, ...]:
    """Fractional phase in [0,1] -> (i0, i1, blend) into an n_in-frame array."""
    f = phase * (n_in - 1)
    i0 = f.floor().long()
    return i0, torch.minimum(i0 + 1, torch.tensor(n_in - 1, device=f.device)), f - i0


def _velocities(state: dict, dt: float) -> dict:
    """Central differences for the linear channels, SO3 derivative for spin."""
    q = state["root_quat"]
    q_rel = quat_mul(q[2:], quat_conjugate(q[:-2]))
    omega = axis_angle_from_quat(q_rel) / (2.0 * dt)
    return {
        "root_lin_vel": torch.gradient(state["root_pos"], spacing=dt, dim=0)[0],
        "joint_vel": torch.gradient(state["joint_pos"], spacing=dt, dim=0)[0],
        "root_ang_vel": torch.cat([omega[:1], omega, omega[-1:]], dim=0),
    }
