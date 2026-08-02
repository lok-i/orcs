"""UOLM termination terms — the OBJECT tracking tubes.

The robot-side tubes (`bad_anchor_*`, `base_collapsed`) and the motion-overrun
truncation are task-blind and live in :mod:`orcs.core.mdp.terminations`; they
are re-exported so ``mdp.<name>`` resolves the whole uolm surface.
"""

from __future__ import annotations

import torch
from mjlab.envs import ManagerBasedRlEnv
from mjlab.utils.lab_api.math import quat_error_magnitude

from orcs.core.mdp.terminations import (  # noqa: F401 — robot-only, moved to core
    bad_anchor_ori,
    bad_anchor_pos,
    base_collapsed,
    exceeded_motion_by_eps,
)

__all__ = [
    "base_collapsed",
    "bad_anchor_pos", "bad_anchor_ori", "bad_object_pos", "bad_object_ori",
    "exceeded_motion_by_eps",
]


def bad_object_pos(
    env: ManagerBasedRlEnv, command_name: str, threshold: float = 0.3
) -> torch.Tensor:
    """Object position drift from motion reference > threshold. (B,) bool."""
    cmd = env.command_manager.get_term(command_name)
    err = torch.linalg.norm(cmd.object.data.root_link_pos_w - cmd.object_pos_w, dim=-1)
    return err > threshold


def bad_object_ori(
    env: ManagerBasedRlEnv, command_name: str, threshold: float = 0.8
) -> torch.Tensor:
    """Object orientation error from motion reference > threshold. (B,) bool."""
    cmd = env.command_manager.get_term(command_name)
    err = quat_error_magnitude(cmd.object.data.root_link_quat_w, cmd.object_quat_w)
    return err > threshold
