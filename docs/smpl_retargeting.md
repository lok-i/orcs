# SMPL retargeting

The `Orcs-PerLoco-Grail-AdaptSonic-Smpl` pipeline has two stages:

```text
SMPL motion -> kinematic retargeting -> seed_state.npz
SMPL motion + seed_state.npz -> AdaptSonic training -> dynamic retargeting
```

The seed is used only for reset-state initialization and action history. The
training reference remains the source SMPL motion: rewards track 14 homologous
SMPL/G1 body points and their velocities, never the upstream GRAIL robot motion.

## prerequisite

First complete [GRAIL fetch and staging](perceptive_locomotion.md#fetch-and-stage-grail).
Every selected sample must contain `motion.npz` and `smpl_motion.npz`.
Kinematic retargeting uses the frozen SONIC checkpoint and, in practice, an
NVIDIA GPU.

## retarget the full GRAIL roster

```bash
orcs-pseudo-retarget --scene perloco-grail --all
```

This processes one clip per fresh worker process, writes `seed_state.npz`
beside each `smpl_motion.npz`, and prints a final completion count. The command
is resumable: an existing seed is skipped only when its schema loads, every
frame is valid, and its frame count matches the source. Failed or incomplete
clips are retried on the next run.

Regenerate every seed explicitly with:

```bash
orcs-pseudo-retarget --scene perloco-grail --all --overwrite
```

For one clip while debugging:

```bash
orcs-pseudo-retarget --scene perloco-grail \
  --source data/terrain_motions/grail/curb_000/level_0.00/sample0
```

Kinematic retargeting aligns the nominal floating base to the first SMPL pelvis
pose, settles for 100 simulation steps, then rolls frozen SONIC with mass-scaled
virtual assistance. No frame-wise pose projection or retargeted robot motion is
used. A clip is not written if the initial settling error fails to reach a
stable low-error plateau.

## inspect a result

```bash
orcs-view-seeds --scene perloco-grail \
  --source data/terrain_motions/grail/curb_000/level_0.00/sample0
# opens http://localhost:8080
```

The viewer overlays the original SMPL skeleton, the simulated G1 rollout, point
targets, virtual-force markers, and per-frame tracking diagnostics. It reads the
saved result and does not rerun the policy.

## smoke test and train

After the batch reports `complete=63/63`, verify both frame-zero and random RSI:

```bash
play Orcs-PerLoco-Grail-AdaptSonic-Smpl --agent initial --viewer native

play Orcs-PerLoco-Grail-AdaptSonic-Smpl --agent initial --viewer native \
  --env.commands.motion.start-from-zero False
```

Launch training:

```bash
train Orcs-PerLoco-Grail-AdaptSonic-Smpl --env.scene.num-envs 4096
```

The task discovers valid seed-complete clips automatically. Missing clips are
excluded only from the `-Smpl` task and do not alter
`Orcs-PerLoco-Grail-AdaptSonic`.

## global assistance gains

The task-agnostic defaults live in `DEFAULT_ASSISTANCE_GAINS` in
`src/orcs/core/assisted_retarget.py`:

| parameter | default |
|---|---:|
| `response_rate` | `8.0` rad/s |
| `damping_ratio` | `1.0` |
| `robot_force_budget_g` | `3.0` |
| `object_force_budget_g` | `5.0` |

Point gains are derived from body mass; there are no task-wise bodies or gains
to tune. If these defaults change, regenerate seeds with `--overwrite`.

## troubleshooting

- `orcs-pseudo-retarget: command not found`: reactivate the environment and run
  `uv pip install --no-deps -e .` once to refresh console entry points.
- `settling did not converge`: inspect that sample with `orcs-view-seeds` if an
  older seed exists, then rerun it alone to retain the full error output.
- The task does not register: run
  `python -c "import mjlab, orcs; print(orcs.tasks.perloco.SKIP_REASON)"` and
  confirm the batch completion count.
