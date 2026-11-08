"""Quota, reference alignment, pre-reset accounting and portable report contracts."""

import json
import subprocess
import sys
from types import SimpleNamespace as NS

import pytest
import torch

from orcs.core.mdp.commands import MultiClipMotionCommand
from orcs.core.motion_benchmark.capture import TrialCapture, tracking_errors
from orcs.core.motion_benchmark.contract import BenchmarkSpec
from orcs.core.motion_benchmark.metrics import aggregate, balanced_batches, wilson


def test_balancing_with_fewer_worlds_than_motions():
    batches = list(balanced_batches(7, 3, 2))
    flat = [item for batch in batches for item in batch]
    assert len(batches) == 11
    assert [t[0] for t in flat] == list(range(21))
    assert all(sum(t[1] == m for t in flat) == 3 for m in range(7))
    assert len(batches[-1]) == 1
    with pytest.raises(ValueError):
        list(balanced_batches(0, 3, 2))


def command(n=2):
    target = torch.zeros(n, 2, 3)
    target[:, 1, 0] = 1
    actual = target + torch.tensor([2., 0., 0.])
    return NS(
        body_pos_w=target, robot_body_pos_w=actual,
        body_lin_vel_w=torch.zeros_like(target), robot_body_lin_vel_w=torch.zeros_like(target),
        anchor_pos_w=target[:, 0], robot_anchor_pos_w=actual[:, 0],
        anchor_quat_w=torch.tensor([[1., 0., 0., 0.]]).repeat(n, 1),
        robot_anchor_quat_w=torch.tensor([[1., 0., 0., 0.]]).repeat(n, 1),
        local_reference_frames=torch.ones(n, dtype=torch.long),
        hold_steps=torch.zeros(n, dtype=torch.long))


def test_metrics_use_authored_smpl_targets_and_translation_only_alignment():
    cmd = command()
    assert tracking_errors(cmd)["global_position_m"].tolist() == [2, 2]
    assert tracking_errors(cmd)["relative_position_m"].tolist() == [0, 0]
    # A seed may match the robot perfectly while the authored target does not.
    cmd.point_target_pos_w = cmd.body_pos_w + 10
    cmd.robot_point_pos_w = cmd.robot_body_pos_w
    cmd.point_target_vel_w = torch.ones_like(cmd.body_pos_w)
    cmd.robot_point_vel_w = torch.zeros_like(cmd.body_pos_w)
    cmd.body_pos_w = cmd.robot_body_pos_w
    result = tracking_errors(cmd)
    assert (result["global_position_m"] > 10).all()
    assert torch.allclose(result["point_velocity_m_s"], torch.full((2,), 3**.5))


def fake_env():
    cmd = command()
    flags = {name: torch.zeros(2, dtype=torch.bool) for name in ("fell", "exceeded_motion", "time_out")}
    tm = NS(active_terms=list(flags), get_term=flags.__getitem__,
            terminated=flags["fell"], time_outs=torch.zeros(2, dtype=torch.bool))
    env = NS(num_envs=2, device="cpu", step_dt=.02, max_episode_length=100,
             command_manager=NS(get_term=lambda _: cmd), termination_manager=tm,
             cfg=NS(terminations={"exceeded_motion": NS(params={"epsilon_steps": 2})}),
             reward_buf=torch.ones(2), episode_length_buf=torch.ones(2, dtype=torch.long))
    return env, cmd, flags


def test_terminal_capture_failure_wins_and_hold_is_separate():
    env, cmd, flags = fake_env()
    capture = TrialCapture(env, BenchmarkSpec(prepare=lambda *_: None), [{"frames": 3}]*2, 4)
    capture.step(env)
    cmd.local_reference_frames[:] = 2
    cmd.hold_steps[:] = 2
    flags["exceeded_motion"][:] = True
    flags["fell"][0] = True
    env.termination_manager.time_outs[:] = True
    env.episode_length_buf[:] = 5
    capture.step(env)
    records = capture.records([(0, 0, 0), (1, 1, 0)], 0, 42)
    assert [r["status"] for r in records] == ["failed", "completed"]
    assert records[0]["end_bin"] == 4
    assert records[0]["elapsed_s"] == .1
    # The next reset state must never contaminate the scored first episode.
    cmd.robot_body_pos_w[:] = 500
    env.episode_length_buf[:] = 1
    capture.step(env)
    assert capture.records([(0, 0, 0), (1, 1, 0)], 0, 42) == records


def test_early_timeout_nonfinite_and_episode_horizon():
    env, cmd, flags = fake_env()
    capture = TrialCapture(env, BenchmarkSpec(prepare=lambda *_: None), [{"frames": 3}]*2, 4)
    flags["time_out"][0] = True
    env.termination_manager.time_outs[0] = True
    env.reward_buf[1] = float("nan")
    capture.step(env)
    assert capture.status.tolist() == [4, 3]
    env.reward_buf[:] = 1
    env.episode_length_buf[:] = 100
    flags["time_out"][:] = True
    env.termination_manager.time_outs[:] = True
    capture = TrialCapture(env, BenchmarkSpec(prepare=lambda *_: None, horizon="episode", completion_term="time_out"), [{"frames": 900}]*2, 4)
    capture.step(env)
    assert capture.status.tolist() == [1, 1]
    assert capture.progress.tolist() == [1, 1]


def test_public_queue_checks_scene_compatibility_and_is_one_shot():
    # Avoid constructing a simulator; exercise the real command's queue methods.
    cmd = object.__new__(MultiClipMotionCommand)
    cmd._env = NS(device="cpu", num_envs=2)
    cmd._clip_ids = torch.zeros(2, dtype=torch.long)
    cmd.motion = NS(n_clips=3, clip_offsets=torch.tensor([0, 20, 50]))
    cmd._clip_allowance = lambda ids: torch.tensor([[1, 0, 1], [0, 1, 0]])[ids]
    cmd._sample_default_init_frame = lambda ids: (torch.ones(len(ids), dtype=torch.long), torch.full((len(ids),), 29))
    with pytest.raises(ValueError, match="scene context"):
        cmd.queue_reset_clips(torch.tensor([1]), torch.tensor([2]))
    cmd.queue_reset_clips(torch.tensor([0]), torch.tensor([2]))
    clips, frames = cmd._sample_init_frame(torch.tensor([0, 1]))
    assert clips.tolist() == [2, 1] and frames.tolist() == [50, 29]
    assert cmd._sample_init_frame(torch.tensor([0]))[1].item() == 29


def receipt():
    return dict(schema_version=1, motions=[dict(index=0, id="clip-a"), dict(index=1, id="clip-b")],
                trials_per_motion=2, bins=2, nominal=False, task="Fixture", state="complete",
                horizon="clip", checkpoint=None, agent="initial", seed=1, tracking_quality=True,
                point_names=["pelvis"], target_type="native robot bodies", versions={})


def trial(i, status="failed", error=2):
    return dict(trial_id=i, motion_index=i%2, status=status, progress=.5, elapsed_s=1.,
                terminations=["fell", "tracking"], metrics={"global_position_m": error},
                bin_samples=[1, 2, 0], end_bin=1)


def test_missing_trials_remain_in_denominator_and_macro_is_motion_balanced():
    r = receipt()
    summary = aggregate(r, [trial(0, "completed", 2), trial(2, "completed", 4), trial(1, error=8)])
    assert summary["macro"]["pending"] == 1
    assert summary["macro"]["completion_rate"] == .5
    assert summary["macro"]["global_position_m"] == 5.5  # mean of 3 and 8, not pooled trials
    a, b = summary["per_motion"]
    assert a["completion_ci95"] is not None and b["completion_ci95"] is None
    assert b["bin_failure_rate"] == [0., 1., None]
    assert b["failed"] == 1  # overlapping tags do not duplicate failures
    assert b["completed_global_position_m"] is None
    with pytest.raises(ValueError, match="Duplicate"):
        aggregate(r, [trial(0), trial(0)])
    r["nominal"] = True
    assert aggregate(r, [trial(i) for i in range(4)])["per_motion"][0]["completion_ci95"] is None
    assert wilson(0, 2)[0] == 0
    assert wilson(2, 2)[1] == 1


def test_task_hooks_preserve_native_failures_and_full_scene_contract():
    from orcs.core.mdp.commands import MultiClipMotionCommandCfg
    from orcs.tasks.dodge.benchmark import SPEC as dodge_spec
    from orcs.tasks.uolm.benchmark import SPEC as uolm_spec

    mc = object.__new__(MultiClipMotionCommandCfg)
    cfg = NS(commands={"motion": mc}, observations={},
             terminations={"bad_object_pos": None, "bad_object_ori": None},
             events={"virtual_object_force": None, "rand_friction": None, "perturb_robot": None})
    before = set(cfg.terminations)
    uolm_spec.prepare(cfg, False)
    assert "virtual_object_force" not in cfg.events
    assert any(k.startswith("rand_") for k in cfg.events)
    uolm_spec.prepare(cfg, True)
    assert not any(k.startswith(("rand_", "perturb_")) for k in cfg.events)
    assert set(cfg.terminations) == before
    assert mc.velocity_range == {} and mc.joint_position_range == (0., 0.)
    cfg.events["throw_ball"] = NS(params={"stand_fraction": .5})
    dodge_spec.prepare(cfg, True)
    assert "throw_ball" in cfg.events
    assert cfg.events["throw_ball"].params["stand_fraction"] == 0
    assert dodge_spec.horizon == "episode" and not dodge_spec.tracking_quality


def test_every_registered_orcs_row_has_a_benchmark_contract():
    from mjlab.tasks.registry import list_tasks

    from orcs.core.motion_benchmark.contract import SPECS

    assert {t for t in list_tasks() if t.startswith("Orcs-")} <= SPECS.keys()


def test_report_only_is_gpu_free_and_escapes_embedded_data(tmp_path):
    r = receipt()
    r["motions"][0]["id"] = "</script><script>alert(1)</script>"
    (tmp_path / "receipt.json").write_text(json.dumps(r))
    (tmp_path / "trials.jsonl").write_text("\n".join(json.dumps(trial(i)) for i in range(4)))
    code = ("import sys; from orcs.cli.bench_motion_repertoire import main; "
            f"main(['--report-only', {str(tmp_path)!r}, '--no-serve']); "
            "assert 'torch' not in sys.modules; assert 'mjlab' not in sys.modules")
    subprocess.run([sys.executable, "-c", code], check=True, capture_output=True, text=True)
    document = (tmp_path / "report.html").read_text()
    assert r["motions"][0]["id"] not in document
    for name in ("summary.json", "per_motion.csv", "trials.csv", "failures.png", "failure_time.svg", "progress.svg"):
        assert (tmp_path / name).stat().st_size > 10


def test_agent_recipe_only_accepts_primitive_yaml(tmp_path):
    import yaml

    from orcs.core.motion_benchmark.balanced_rollouts import load_agent_recipe

    path = tmp_path / "agent.yaml"
    path.write_text("actor:\n  hidden_dims: !!python/tuple [8, 8]\n")
    cfg, source = load_agent_recipe({}, None, str(path))
    assert cfg["actor"]["hidden_dims"] == (8, 8)
    assert len(source["sha256"]) == 64
    path.write_text("!!python/object:builtins.object {}")
    with pytest.raises(yaml.constructor.ConstructorError):
        load_agent_recipe({}, None, str(path))
