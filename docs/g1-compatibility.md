# G1 mechanics across simulator upgrades

ORCS preserves the G1 model used by its existing frozen SONIC controllers when
upgrading the simulator. Adopting a newer simulator does not automatically adopt
new robot constraints.

## Coupled ankle stops

mjlab main at `d872bfd2b048da657345c07669172fa284b38abf` includes eight fixed
tendons that jointly limit ankle roll and pitch. Those mechanical stops were
absent from the previous candidate's mjlab revision
`11ffee9816e1d0644e538854bfb8dd8acb82ebba`.
See the [upstream model comparison](https://github.com/mujocolab/mjlab/compare/11ffee9816e1d0644e538854bfb8dd8acb82ebba...d872bfd2b048da657345c07669172fa284b38abf).

`orcs.assets.g1.flat_hand_spec()` excludes only these eight identified stops.
It matches joint names, coefficients, ranges and limit activation before any
deletion. An unfamiliar or incomplete set raises an error for review. Other
tendons and the individual ankle joint limits remain intact. This applies to
ORCS tasks and consumers that compose their robot from this factory, including
ViBe's full and reduced-visual G1 variants. mjlab's own robot factory is unchanged.

The preserved contract includes joint and actuator order, individual limits,
armature/damping/friction, actuator gains and force limits, body mass/inertia,
collision geometry, and collision filtering. Flat hand collision surgery is
unchanged. Rewards, terminations, observations and action contracts are separate
task contracts and remain unchanged by this decision.

## Regression guard

`tests/test_g1_compatibility.py` compiles the SONIC-profile robot and compares it
with `tests/fixtures/g1-contract.json`. The fixture records names, sizes and
SHA-256 digests of mechanical arrays from ORCS `a0efd45` with mjlab
`11ffee9816e1d0644e538854bfb8dd8acb82ebba` and MuJoCo core 4.0.1. It contains no
policy weights or task data. Additional tests ensure unrelated tendons survive
and unfamiliar ankle constraints fail explicitly.

Do not regenerate the fixture merely to make an upgrade pass. A mismatch needs
an explanation and review of the affected model field.

Preserving the compiled mechanical configuration does not establish identical
trajectories or checkpoint performance across simulator versions. Solver,
contact, reset and randomization fixes still require rollout validation.
Released-checkpoint provenance remains unchanged.

## Adopting the new stops later

Treat coupled ankle stops as a deliberate robot-model change. Validate the frozen
controller, joint-limit behavior, task learning and existing checkpoints before
changing this contract and its fixture. Record that change separately from a
dependency upgrade.
