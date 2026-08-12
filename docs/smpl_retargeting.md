# SMPL retargeting

The PerLoco-GRAIL and UOLM reconstructed-motion pipelines have two stages:

```text
SMPL motion -> kinematic retargeting -> seed_state.npz
SMPL motion + seed_state.npz -> AdaptSonic training -> dynamic retargeting
```

The seed is used only for reset-state initialization and action history. The
training reference remains the source data: rewards track 14 homologous
SMPL/G1 body points and their velocities, plus the source object trajectory for
UOLM. The kinematic robot/object trace is never a tracking teacher.

All persisted positions are environment-local. Corpus batching may place each
simulation world at a different layout origin, but that origin is removed on
write and applied exactly once by RSI in the destination training scene.

## prerequisite

First complete [GRAIL fetch and staging](perceptive_locomotion.md#fetch-and-stage-grail).
Every selected sample must contain `motion.npz` and `smpl_motion.npz`.
Kinematic retargeting uses the frozen SONIC checkpoint and, in practice, an
NVIDIA GPU.

## retarget the full GRAIL roster

```bash
orcs-pseudo-retarget --scene perloco-grail --all
```

This assigns one pending clip to each mjlab world and rolls the full corpus out
in one GPU batch, with one environment build and one SONIC checkpoint load.
Ragged clips stop recording at their own length while longer worlds continue.
The command writes `seed_state.npz` beside each `smpl_motion.npz` and is
resumable: an existing seed is skipped only when its schema loads, every frame
is valid, and its frame count matches the source. Only worlds that fail the
batched settling/validity checks are retried in an isolated process.

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
pose, settles for at least 100 simulation steps (continuing automatically until
a bounded low-error plateau), then rolls frozen SONIC with mass-scaled virtual
assistance. No frame-wise pose projection or retargeted robot motion is used. A
clip is not written if that plateau is not reached within the global bound.

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

## UOLM reconstructed motions

`deps.lock` pins the reconstructed corpus at `data/reconstructed_motions`.
Sync it, normalize and kinematically retarget both supported sets with one
resumable command:

```bash
bash scripts/setup/sync_dependencies.sh
orcs-pseudo-retarget --scene uolm --all
```

As with GRAIL, every pending clip becomes one mjlab world in a single rollout.
UOLM batches additionally assign the matching cube size/mass per world,
initialize each authored object stream at frame zero, and place or park its
table support according to the motion set. This also holds for a mixed
`--scene uolm --all` batch; no subprocess is launched unless one world fails
settling or output validity.

The generated cache is grouped by behavior/scene rather than flattened:

```text
data/smpl_motions/uolm/reconstructed/
├── small-cube-table/<interaction>/<clip>/
└── big-cube-floor/<interaction>/<clip>/
```

Each leaf contains `smpl_motion.npz`, `object_motion.npz`, `metadata.json`, and
`seed_state.npz`. Process only one set when iterating:

```bash
orcs-pseudo-retarget --scene uolm --motion-set small-cube-table --all
orcs-pseudo-retarget --scene uolm --motion-set big-cube-floor --all
```

The source adapter removes one robust, clip-wide vertical calibration bias
from the reconstructed human before retargeting; it does not flatten genuine
per-frame motion or move the independently authored object. Adapter schema
changes automatically invalidate stale staged samples and seeds during the
resumable command. A `*-Smpl` task refuses a partial or stale selected set
rather than silently training on it.

Inspect any cached sample after retargeting:

```bash
orcs-view-seeds --scene uolm --motion-set small-cube-table \
  --source data/smpl_motions/uolm/reconstructed/small-cube-table/carryflip/<clip>
```

The viewer overlays the original SMPL and object motions, simulated seed
rollouts, and robot/object virtual-force markers. The object controller tracks
the authored pelvis-object transform; it has no hand spring or gravity
compensation.

Train the combined corpus or a named specialization:

```bash
train Orcs-Uolm-AdaptSonic-Smpl --env.scene.num-envs 4096
train Orcs-Uolm-SmallCubeTable-AdaptSonic-Smpl --env.scene.num-envs 4096
train Orcs-Uolm-BigCubeFloor-AdaptSonic-Smpl --env.scene.num-envs 4096
```

In the combined task, object size/mass and clip selection are matched per
world. The table is placed beneath the selected clip's authored destination
for `small-cube-table` and parked below the world for `big-cube-floor`.
Reconstructed clips do not claim authored robot-contact labels, so this task
uses SMPL point rewards and standard UOLM object pose/goal rewards without the
retargeted-motion contact-consistency term.

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
- A task does not register: run
  `python -c "import mjlab, orcs; print(orcs.tasks.perloco.SKIP_REASON); print(orcs.tasks.uolm.SKIP_REASON)"`
  and confirm the relevant batch completion count.
