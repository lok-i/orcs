from __future__ import annotations

from types import SimpleNamespace

import torch

from orcs.core.mdp.commands import MultiClipMotionCommand


def test_motion_update_scopes_frame_advance_to_reset_env_ids() -> None:
    command = object.__new__(MultiClipMotionCommand)
    command.cfg = SimpleNamespace(init_phase_anneal_iterations=0)
    command._env = SimpleNamespace(extras={})
    command._init_phase_max = 1.0
    command.metrics = {"init_phase_max": torch.zeros(3)}
    command.motion = SimpleNamespace(clip_ends=torch.tensor([4, 7]))
    command._clip_ids = torch.tensor([0, 0, 1])
    command.time_steps = torch.tensor([1, 2, 5])
    command._steps_past_end = torch.zeros(3, dtype=torch.long)
    command._update_task = lambda: None
    command.update_relative_body_poses = lambda: None

    command._update_command(torch.tensor([1]))

    assert command.time_steps.tolist() == [1, 3, 5]
    assert command._steps_past_end.tolist() == [0, 0, 0]

    command._update_command()

    assert command.time_steps.tolist() == [2, 3, 6]
    assert command._steps_past_end.tolist() == [0, 1, 0]
