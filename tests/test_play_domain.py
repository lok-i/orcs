"""Play removes robot RSI perturbations without changing the training recipe."""

from __future__ import annotations

from copy import deepcopy

import pytest
from mjlab.envs import ManagerBasedRlEnvCfg
from mjlab.scene import SceneCfg

from orcs.core.mdp.commands import MultiClipMotionCommandCfg
from orcs.core.robustness import (
    ROBOT_STATE_VARIATION,
    apply_robot_robustness,
    strip_domain,
)
from orcs.tasks.perloco.mdp.commands import TerrainMotionCommandCfg
from orcs.tasks.uolm.mdp.commands import ObjectMotionCommandCfg


@pytest.mark.parametrize("command_type", [
    MultiClipMotionCommandCfg, TerrainMotionCommandCfg, ObjectMotionCommandCfg,
])
def test_play_strips_robot_rsi_for_every_motion_command(command_type, monkeypatch):
    # Exercise pose perturbations too, even though the standard range is empty.
    pose_range = {"z": (-0.01, 0.01)}
    monkeypatch.setitem(ROBOT_STATE_VARIATION, "robot_pose_range", pose_range)
    command = command_type(
        motion_file="unused.npz", resampling_time_range=(1e9, 1e9),
        init_phase_anneal_iterations=100,
    )
    train = ManagerBasedRlEnvCfg(
        decimation=4, scene=SceneCfg(num_envs=1), commands={"motion": command},
    )
    apply_robot_robustness(train, param=False)
    play = deepcopy(train)
    strip_domain(play)
    strip_domain(play)  # callers can apply play overrides more than once

    motion = play.commands["motion"]
    assert motion.pose_range == {}
    assert motion.velocity_range == {}
    assert motion.joint_position_range == (0.0, 0.0)
    assert "perturb_robot" not in play.events
    # Domain stripping owns perturbations, not phase selection or references.
    assert motion.motion_file == command.motion_file
    assert motion.start_from_zero == command.start_from_zero
    assert motion.init_phase_anneal_iterations == 100
    assert command.pose_range == pose_range
    assert command.velocity_range == ROBOT_STATE_VARIATION["robot_velocity_range"]
    assert command.joint_position_range == (-0.05, 0.05)
    assert "perturb_robot" in train.events


def test_play_domain_stripping_accepts_tasks_without_motion():
    cfg = ManagerBasedRlEnvCfg(decimation=4, scene=SceneCfg(num_envs=1))
    strip_domain(cfg)
    assert cfg.commands == {}
