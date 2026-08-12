# orcs

Privileged oracle policies for humanoid control on
[mjlab](https://github.com/mujocolab/mjlab).

ORCS provides the task mechanics and privileged policies used to train teachers,
retarget motions, and bootstrap downstream visual policies.

| task | privileged input |
|---|---|
| `Orcs-Uolm-*` | object kinematics |
| `Orcs-PerLoco-*` | terrain height scan |
| `Orcs-Dodge-*` | projectile state |

## install

Requires Python 3.11, Git LFS, and GitHub SSH access for the pinned research
dependencies. Run from the repository root:

```bash
uv venv --python 3.11 .venv
source .venv/bin/activate

# Include perloco before syncing: staging needs USD and SMPL-X.
uv pip install -e ".[dev,perloco]"
bash scripts/setup/sync_dependencies.sh

# UOLM object models are generated locally from the synced assets.
python dependencies/assets/source/omni_objects/make_object_models.py --all
```

`sync_dependencies.sh` reads `deps.lock`, downloads the shared data, and
installs the pinned `mocke` and `rsl_rl` forks into the active environment.
Re-running it is safe.

Verify the installation:

```bash
python -c "import mjlab, orcs; from mjlab.tasks.registry import list_tasks; print('\n'.join(list_tasks()))"
```

Import `mjlab` before `orcs` in standalone scripts so task entry-point discovery
finishes before ORCS registration.

## data workflows

- [Perceptive locomotion](docs/perceptive_locomotion.md) — fetch and stage the
  OmniRetarget or GRAIL terrain-motion datasets.
- [SMPL retargeting](docs/smpl_retargeting.md) — kinematically retarget the full
  staged GRAIL roster, inspect it, and launch `PerLoco-Grail-*-Smpl` training.

The licensed SMPL-X body model is the only manual download in the GRAIL setup.
Both workflows are resumable and follow the packaged terrain rosters.

## play and train

```bash
play Orcs-Uolm-AdaptSonic --agent initial --viewer native
play Orcs-PerLoco-Grail-AdaptSonic --agent initial --viewer native

train Orcs-Uolm-AdaptSonic --env.scene.num-envs 4096
train Orcs-PerLoco-Grail-AdaptSonic --env.scene.num-envs 4096
```

`--agent initial` runs the frozen SONIC base with a zero-initialized adapter and
does not require a training checkpoint. `TaRa` task variants provide
from-scratch baselines.

## paths

ORCS resolves data and dependencies from the host repository when vendored.
Override them only when the default layout is unsuitable:

| variable | default |
|---|---|
| `ORCS_DATA_ROOT` | `<repo>/data` |
| `ORCS_DEPS_ROOT` | `<repo>/dependencies` |
| `ORCS_ASSETS_SOURCE` | `<deps>/assets/source` |
| `ORCS_SMPLX_DIR` | `<deps>/GRAIL/imports/GEM-SMPL/inputs/checkpoints/body_models` |

## development

```bash
ruff check src scripts tests
pytest
```

See [docs/ethos.md](docs/ethos.md) for package boundaries and the task contract.

## acknowledgements

*Standing on the shoulders of giants:* [SONIC](https://github.com/NVlabs/GR00T-WholeBodyControl),
[GRAIL](https://github.com/NVlabs/GRAIL),
[OmniRetarget](https://huggingface.co/datasets/omniretarget/OmniRetarget_Dataset),
[mjlab](https://github.com/mujocolab/mjlab), and
[RSL-RL](https://github.com/leggedrobotics/rsl_rl).
