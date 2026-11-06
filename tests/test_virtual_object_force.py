"""Virtual assistance must follow randomized dynamics, not construction-time mass."""

from types import SimpleNamespace

import pytest
import torch

from orcs.tasks.uolm.mdp.curriculums import VirtualObjectForceCurriculum


def _fixture(natural_frequency=2.0):
    writes = []
    data = SimpleNamespace(
        indexing=SimpleNamespace(body_ids=[1]),
        model=SimpleNamespace(
            body_mass=torch.tensor([[100.0, 3.0]]),
            body_inertia=torch.ones(1, 2, 3),
        ),
        root_link_pos_w=torch.zeros(2, 3),
        root_link_quat_w=torch.tensor([[1.0, 0.0, 0.0, 0.0]]).repeat(2, 1),
        root_link_lin_vel_w=torch.tensor([[0.25, 0.0, 0.0]]).repeat(2, 1),
        root_link_ang_vel_w=torch.tensor([[0.0, 0.0, 0.5]]).repeat(2, 1),
    )
    obj = SimpleNamespace(
        data=data,
        write_external_wrench_to_sim=lambda f, t: writes.append((f.clone(), t.clone())),
    )
    command = SimpleNamespace(
        object_pos_w=torch.tensor([[1.0, 0.0, 0.0]]).repeat(2, 1),
        object_quat_w=torch.tensor([[0.6, 0.0, 0.0, 0.8]]).repeat(2, 1),
    )
    env = SimpleNamespace(
        scene={"object": obj},
        device="cpu",
        num_envs=2,
        policy_update_count=0,
        extras={},
        command_manager=SimpleNamespace(get_term=lambda _: command),
    )
    cfg = SimpleNamespace(
        params=dict(
            natural_frequency=natural_frequency,
            terminal_scale=1e-4,
            decay_mode="exponential",
            decay_by_policy_iterations=10000,
        )
    )
    return VirtualObjectForceCurriculum(cfg, env), env, data, writes


@pytest.mark.parametrize("after_first_call", [False, True])
def test_wrench_tracks_mass_and_inertia_changed_after_construction(after_first_call):
    term, env, data, writes = _fixture()
    if after_first_call:
        term(env, None)
    # Startup DR expands/rebinds model arrays after event-term construction.
    data.model = SimpleNamespace(
        body_mass=torch.tensor([[100.0, 2.0], [200.0, 5.0]]),
        body_inertia=torch.tensor(
            [
                [[100.0, 100.0, 100.0], [1.0, 2.0, 3.0]],
                [[200.0, 200.0, 200.0], [6.0, 9.0, 12.0]],
            ]
        ),
    )
    term(env, None)
    force, torque = writes[-1]
    # wn=2: linear acceleration = 4*1 - 4*.25 = 3.
    # Rotation-vector approximation = 2*.8; angular acceleration = 4*1.6 - 4*.5 = 4.4.
    torch.testing.assert_close(
        force[:, 0], torch.tensor([[6.0, 0.0, 0.0], [15.0, 0.0, 0.0]])
    )
    torch.testing.assert_close(
        torque[:, 0], torch.tensor([[0.0, 0.0, 8.8], [0.0, 0.0, 39.6]])
    )
    # Also cover in-place parameter updates after a wrench has already been applied.
    data.model.body_mass[:, 1] *= 2
    data.model.body_inertia[:, 1] *= 3
    term(env, None)
    torch.testing.assert_close(writes[-1][0], force * 2)
    torch.testing.assert_close(writes[-1][1], torque * 3)


def test_decay_preserves_nominal_wrench_and_clears_after_the_boundary():
    term, env, _, writes = _fixture()
    term(env, None)
    force, torque = writes[-1]
    torch.testing.assert_close(
        force[:, 0], torch.tensor([[9.0, 0.0, 0.0]]).repeat(2, 1)
    )
    torch.testing.assert_close(
        torque[:, 0], torch.tensor([[0.0, 0.0, 4.4]]).repeat(2, 1)
    )
    for iteration, scale in ((5000, 0.01), (10000, 1e-4), (10001, 0.0)):
        env.policy_update_count = iteration
        term(env, None)
        torch.testing.assert_close(writes[-1][0], force * scale)
        torch.testing.assert_close(writes[-1][1], torque * scale)
        assert env.extras["log"]["VirtualObjectForce/alpha"] == pytest.approx(scale)
    count = len(writes)
    term(env, None)
    assert len(writes) == count  # The persistent wrench was cleared once.


def test_zero_natural_frequency_disables_force_and_torque_from_the_first_call():
    term, env, _, writes = _fixture(natural_frequency=0.0)
    for iteration in (0, 5000, 10000, 10001):
        env.policy_update_count = iteration
        term(env, None)
        force, torque = writes[-1]
        assert torch.count_nonzero(force) == 0
        assert torch.count_nonzero(torque) == 0
