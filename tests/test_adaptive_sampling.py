from __future__ import annotations

from types import SimpleNamespace as NS

import pytest
import torch
from mjlab.managers import CommandTerm

from orcs.core.mdp.adaptive_sampling import AdaptiveMotionSampler
from orcs.core.mdp.commands import MultiClipMotionCommand


def sampler(lengths=(150, 150), **kwargs):
    return AdaptiveMotionSampler(torch.tensor(lengths), 0.02, **kwargs)


def test_failures_accumulate_and_decay_without_crossing_clip_boundaries():
    s = sampler(kernel_size=3, decay=0.5, alpha=0.5)
    baseline = s.weights()
    # Two reset groups in one step must accumulate, not overwrite one another.
    s.record_failures(torch.tensor([1]), torch.tensor([0]))
    s.record_failures(torch.tensor([1, 1]), torch.tensor([0, 0]))
    s.advance()
    assert s.failed[1, 0] == 1.5
    assert not s.pending.any()
    torch.testing.assert_close(s.weights()[0], baseline[0])
    assert s.weights()[1, 0] > baseline[1, 0]
    s.advance()
    assert s.failed[1, 0] == 0.75


def test_lookback_prioritizes_lead_in_not_later_bins():
    s = sampler((250,), kernel_size=3, decay=0.5, alpha=1)
    before = s.weights()
    s.record_failures(torch.tensor([0]), torch.tensor([100]))
    s.advance()
    delta = (s.weights() - before)[0]
    assert 0 < delta[0] < delta[1] < delta[2]
    assert torch.equal(delta[3:], torch.zeros(2))


@pytest.mark.parametrize("phase", [0.0, 0.001, 0.37, 1.0])
def test_sampling_respects_short_clips_partial_bins_and_phase_caps(phase):
    torch.manual_seed(7)
    s = sampler((1, 49, 51, 125), kernel_size=3)
    clips, frames, probs = s.sample(5000, phase_max=phase)
    limits = (s.lengths * phase).long().clamp(min=1)
    assert (frames >= 0).all() and (frames < limits[clips]).all()
    assert set(clips.tolist()) == {0, 1, 2, 3}
    assert torch.isfinite(probs).all() and probs.sum().item() == pytest.approx(1)
    assert not s.weights()[~s.valid].any()


@pytest.mark.parametrize("from_zero", [False, True])
def test_late_failures_still_reweight_clips_at_phase_zero(from_zero):
    torch.manual_seed(8)
    s = sampler(start_fraction=0.25, alpha=1)
    s.record_failures(torch.ones(100, dtype=torch.long), torch.full((100,), 149))
    s.advance()
    clips, frames, probs = s.sample(10000, phase_max=0 if not from_zero else 1,
                                    start_from_zero=from_zero)
    assert not frames.any()
    assert 0.85 < (clips == 1).float().mean() < 0.9
    assert probs[0] >= 0.25 / 2  # Uniform full-start practice survives hard-clip bias.


def test_full_start_mixture_and_live_scene_allowance():
    torch.manual_seed(9)
    s = sampler((500, 500, 500), start_fraction=0.25)
    clips, frames, _ = s.sample(20000)
    assert 0.23 < (frames == 0).float().mean() < 0.27
    assert torch.bincount(clips).min() > 6000
    allowed = torch.tensor([[1, 0, 0], [0, 1, 1]], dtype=torch.float)
    for _ in range(10):
        clips, _, _ = s.sample(2, allowed=allowed)
        assert allowed[torch.arange(2), clips].all()
    allowed[0] = torch.tensor([0, 0, 1])  # A terrain promotion changes the mask.
    assert s.sample(2, allowed=allowed)[0][0] == 2
    with pytest.raises(ValueError, match="at least one"):
        s.sample(1, allowed=torch.zeros(1, 3))


@pytest.mark.parametrize("kwargs", [dict(alpha=0), dict(alpha=2), dict(decay=-1),
                                    dict(uniform_prior=0), dict(start_fraction=1.1),
                                    dict(kernel_size=0)])
def test_invalid_configuration_is_rejected(kwargs):
    with pytest.raises(ValueError):
        sampler(**kwargs)


def fake_command():
    cmd = object.__new__(MultiClipMotionCommand)
    cmd.cfg = NS(init_phase_anneal_iterations=20000, start_from_zero=False)
    cmd._env = NS(device="cpu", num_envs=4, common_step_counter=0,
                  policy_update_count=0, extras={},
                  episode_length_buf=torch.tensor([2, 2, 0, 2]),
                  termination_manager=NS(terminated=torch.tensor([True, False, True, True])))
    cmd.motion = NS(n_clips=2, max_clip_length=150, clip_lengths=torch.tensor([150, 150]),
                    clip_offsets=torch.tensor([0, 150]), clip_ends=torch.tensor([150, 300]))
    cmd._clip_ids = torch.tensor([0, 1, 1, 1])
    cmd.time_steps = torch.tensor([60, 200, 200, 200])
    cmd._steps_past_end = torch.zeros(4, dtype=torch.long)
    cmd._adaptive_episode = torch.tensor([True, True, True, False])
    cmd._adaptive_sampler = sampler(alpha=1)
    cmd._adaptive_last_step = 0
    cmd._adaptive_probabilities = torch.tensor([0.5, 0.5])
    cmd._adaptive_log_names = ["MotionSampling/clip_0", "MotionSampling/clip_1"]
    cmd._init_phase_max = 1.0
    cmd.metrics = {key: torch.zeros(4) for key in
                   ("init_phase_max", "sampling_entropy", "sampling_top1_prob", "sampling_start_from_zero")}
    cmd._clip_allowance = lambda ids: None
    cmd._update_task = lambda: None
    cmd.update_relative_body_poses = lambda: None
    return cmd


def test_reset_records_only_actual_failed_training_episodes_before_overwrite(monkeypatch):
    cmd = fake_command()
    def base_reset(self, ids):
        self.time_steps[ids] = 0
        return {}
    monkeypatch.setattr(CommandTerm, "reset", base_reset)
    cmd.reset(torch.arange(4))
    expected = torch.zeros(2, 3)
    expected[0, 1] = 1  # timeout, initial reset, and queued benchmark episode excluded
    torch.testing.assert_close(cmd._adaptive_sampler.pending, expected)
    cmd.reset(torch.arange(4))
    torch.testing.assert_close(cmd._adaptive_sampler.pending, expected)


def test_ema_advances_once_per_control_step_and_annealing_reaches_zero():
    cmd = fake_command()
    cmd._adaptive_sampler.pending[0, 0] = 2
    cmd._update_command(torch.tensor([0]))
    assert not cmd._adaptive_sampler.failed.any()
    cmd._env.common_step_counter = 1
    cmd._env.policy_update_count = 10000
    cmd._update_command()
    assert cmd._init_phase_max == 0.5
    assert cmd._adaptive_sampler.failed[0, 0] == 2
    cmd._update_command()  # reset-time compute(dt=0) must not decay again
    assert cmd._adaptive_sampler.failed[0, 0] == 2
    cmd._env.policy_update_count = 20000
    cmd._refresh_init_phase()
    assert cmd._init_phase_max == 0
    clips, frames = cmd._sample_init_frame(torch.arange(4))
    torch.testing.assert_close(frames, cmd.motion.clip_offsets[clips])


def test_queued_benchmark_selection_overrides_adaptive_sampling_once():
    cmd = fake_command()
    cmd.queue_reset_clips(torch.tensor([0, 2]), torch.tensor([1, 0]))
    clips, frames = cmd._sample_init_frame(torch.arange(4))
    assert clips[[0, 2]].tolist() == [1, 0]
    assert frames[[0, 2]].tolist() == [150, 0]
    assert (cmd._queued_clips == -1).all()
