# orcs

A package for synthesizing privileged control policies for:

1. teacher-student distillation;
2. kinodynamic retargeting of human motions.

## supported tasks

| task | tag | privileged input |
|---|---|---|
| omni-object loco-manipulation | `Orcs-Uolm-*` | object kinematic state |
| perceptive locomotion | `Orcs-PerLoco-*` | terrain height scan |
| dodgeball | `Orcs-Dodge-*` | object kinematic state |

## install

Requires Python 3.11, Git LFS, and GitHub SSH access for the pinned research
dependencies. Run from the repository root:

```bash
uv venv --python 3.11 .venv
source .venv/bin/activate

# Include perloco before syncing: staging needs USD and SMPL-X.
uv pip install -e ".[dev,perloco]"

# Sync pinned data/dependencies and generate UOLM object models.
bash scripts/setup/sync_dependencies.sh
```

Verify the installation:

```bash
python -c "import mjlab, orcs; from mjlab.tasks.registry import list_tasks; print('\n'.join(list_tasks()))"
```

Missing optional datasets suppress only their corresponding task rows. Inspect
`orcs.SKIP_REASON` for the exact reason without enabling import-time warnings.

> [!NOTE]
> Importing `mjlab` discovers and registers ORCS tasks. Bare `import orcs` is
> intentionally lightweight; accessing `orcs.SKIP_REASON` also triggers discovery.
> Import `mjlab` first before importing ORCS task, asset, or MJLab-backed core
> submodules directly.

## data workflows

- [Perceptive locomotion](docs/perceptive_locomotion.md): fetch and stage the
  OmniRetarget or GRAIL terrain-motion datasets.
- [SMPL retargeting](docs/smpl_retargeting.md): kinematically retarget the full
  GRAIL or reconstructed UOLM corpus, then launch the matching `*-Smpl` task
  for dynamics refinement.

> [!NOTE]
> The licensed SMPL-X body model is the only manual download in the GRAIL setup.
> Both workflows are resumable and follow the packaged terrain rosters.

## play and train

```bash
# `--agent initial` runs an initial policy i.e. first training iteration
play Orcs-Uolm-AdaptSonic --agent initial --viewer native
play Orcs-PerLoco-Grail-AdaptSonic --agent initial --viewer native

train Orcs-Uolm-AdaptSonic --env.scene.num-envs 4096
train Orcs-PerLoco-Grail-AdaptSonic --env.scene.num-envs 4096
```

## paths

ORCS resolves data and dependencies from the host repository when vendored.
Override them only when the default layout is unsuitable:

| variable | default |
|---|---|
| `ORCS_DATA_ROOT` | `<repo>/data` |
| `ORCS_DEPS_ROOT` | `<repo>/dependencies` |
| `ORCS_ASSETS_SOURCE` | host assets checkout, then installed `assets` package |
| `ORCS_SMPLX_DIR` | `<deps>/GRAIL/imports/GEM-SMPL/inputs/checkpoints/body_models` |

## development

```bash
ruff check src scripts tests
pytest
```

See [docs/ethos.md](docs/ethos.md) for package boundaries and the task contract.

## acknowledgements

*"Standing on the shoulders of giants":*

1. [SONIC](https://github.com/NVlabs/GR00T-WholeBodyControl)
2. [GRAIL](https://github.com/NVlabs/GRAIL)
3. [OmniRetarget](https://huggingface.co/datasets/omniretarget/OmniRetarget_Dataset)
4. [mjlab](https://github.com/mujocolab/mjlab)
5. [rsl-rl](https://github.com/leggedrobotics/rsl_rl)
