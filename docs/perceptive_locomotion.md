# perceptive locomotion

PerLoco pairs each privileged motion with its terrain tile. The AdaptSonic
policy receives a terrain height scan; the frozen SONIC base and motion command
remain shared with the other ORCS tasks.

## prerequisites

Complete the root [installation](../readme.md#install) with the `perloco` extra
inside an active Python 3.11 environment. You also need Git LFS and enough disk
space for the selected source data.

GRAIL SMPL staging requires the licensed **SMPL-X v1.1 NPZ** release. Register
at [smpl-x.is.tue.mpg.de](https://smpl-x.is.tue.mpg.de), download
`SMPL-X v1.1 (NPZ+PKL)`, and keep `SMPLX_NEUTRAL.npz` ready. The setup script
will print the exact destination and pause until the file exists.

## fetch and stage GRAIL

From the ORCS repository root:

```bash
bash scripts/setup/perceptive_locomotion.sh --sources grail
```

The command is idempotent and resumable. It:

1. downloads only the GRAIL curb families named by the packaged roster;
2. clones the GRAIL reference code without its large submodules;
3. waits for `SMPLX_NEUTRAL.npz` at the printed path;
4. stages terrain, robot motion, and SMPL motion under
   `$ORCS_DATA_ROOT/terrain_motions/grail`.

The current roster stages 63 clips over 8 curb tiles. Source selection lives in
`src/orcs/tasks/perloco/rosters/grail.toml`; the same roster controls download,
staging, retargeting, and runtime sampling.

To fetch both supported PerLoco datasets instead:

```bash
bash scripts/setup/perceptive_locomotion.sh
```

Use `--no-smpl` only when you do not need the `-Smpl` task. Run
`bash scripts/setup/perceptive_locomotion.sh --help` for fetch-only,
stage-only, and non-interactive options.

## verify the staged task

```bash
python -c "import mjlab, orcs; print(orcs.tasks.perloco.SKIP_REASON)"
play Orcs-PerLoco-Grail-AdaptSonic --agent initial --viewer native
```

`SKIP_REASON == {}` means every staged PerLoco task that has its required data
registered successfully. Before kinematic retargeting, a skip reason for the
`-Smpl` task is expected because `seed_state.npz` does not exist yet.

Inspect terrain-motion alignment in the browser viewer:

```bash
orcs-view-terrain --source grail
# opens http://localhost:8080
```

## train

```bash
train Orcs-PerLoco-Grail-AdaptSonic --env.scene.num-envs 4096
train Orcs-PerLoco-OmRe-AdaptSonic --env.scene.num-envs 4096
```

For SMPL point tracking and seed-backed RSI, continue with
[SMPL retargeting](smpl_retargeting.md).

## troubleshooting

- `No module named pxr` or `smplx`: install `.[perloco]`, then rerun
  `scripts/setup/sync_dependencies.sh` so the pinned `rsl_rl` fork remains the
  final install.
- An LFS pointer is being parsed as data: install Git LFS and rerun the setup
  script; do not run an unscoped `git lfs pull` in the full GRAIL dataset.
- A task is absent: read `orcs.tasks.perloco.SKIP_REASON`; registration skips
  missing datasets instead of making `import orcs` fail.
