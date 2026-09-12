# <img src="docs/media/logo.png" alt="" height="20"> orcs

a pkg for training privileged-observation humanoid whole-body controllers. supports

1. LoRA PEFT  (currently supports [SONIC](https://nvlabs.github.io/GEAR-SONIC/))
2. *tabula rasa* training (untested)
3. kinodynamic retargeting of human motions.

## tasks

<table>
  <tr>
    <td align="center">
      <a href="src/orcs/tasks/perloco"><img width="360" src="docs/media/perloco_omre.gif" alt="Orcs-PerLoco-OmRe-AdaptSonic"></a><br>
      <code>Orcs-PerLoco-OmRe-AdaptSonic</code>
    </td>
    <td align="center">
      <a href="src/orcs/tasks/uolm"><img width="360" src="docs/media/uolm.gif" alt="Orcs-Uolm-AdaptSonic"></a><br>
      <code>Orcs-Uolm-AdaptSonic</code>
    </td>
  </tr>
  <tr>
    <td align="center">
      <a href="src/orcs/tasks/perloco"><img width="360" src="docs/media/perloco_grail.gif" alt="Orcs-PerLoco-Grail-AdaptSonic"></a><br>
      <code>Orcs-PerLoco-Grail-AdaptSonic</code>
    </td>
    <td align="center">
      <a href="src/orcs/tasks/dodge"><img width="360" src="docs/media/dodge.gif" alt="Orcs-Dodge-AdaptSonic"></a><br>
      <code>Orcs-Dodge-AdaptSonic</code>
    </td>
  </tr>
  <tr>
    <td align="center">
      <a href="src/orcs/tasks/perloco"><img width="360" src="docs/media/perloco_grail_smpl.gif" alt="Orcs-PerLoco-Grail-AdaptSonic-Smpl"></a><br>
      <code>Orcs-PerLoco-Grail-AdaptSonic-Smpl</code>
    </td>
    <td align="center">
      <a href="src/orcs/tasks/uolm"><img width="360" src="docs/media/smallbox_table_smpl.gif" alt="Orcs-Uolm-SmallCubeTable-AdaptSonic-Smpl"></a><br>
      <code>Orcs-Uolm-SmallCubeTable-AdaptSonic-Smpl</code>
    </td>
  </tr>
</table>

## setup

Requires Python 3.11, Git LFS, and GitHub SSH access. From the repository root:

```bash
uv venv --python 3.11 .venv
source .venv/bin/activate

# Standard install.
uv pip install -e .

# PerLoco/SMPL data tooling.
# uv pip install -e ".[perloco]"

# Full contributor setup (tests, lint, and PerLoco/SMPL tooling).
# uv pip install -e ".[dev,perloco]"

# Fetch pinned dependencies/data and generate object assets.
bash scripts/setup/sync_dependencies.sh

# Optional — fetch and stage OmniRetarget + GRAIL for PerLoco.
bash scripts/setup/perceptive_locomotion.sh

# Optional — generate SMPL seed states for dynamic retargeting.
orcs-pseudo-retarget --scene perloco-grail --all
orcs-pseudo-retarget --scene uolm --all

# Optional: download and verify all public release checkpoints.
bash scripts/setup/download_released_models.sh
```

Public checkpoints: [huggingface.co/lkrajan/orcs](https://huggingface.co/lkrajan/orcs).

> [!NOTE]
> The PerLoco setup prompts for the separately licensed SMPL-X model when needed.
> See [perceptive locomotion](docs/perceptive_locomotion.md) and
> [SMPL retargeting](docs/smpl_retargeting.md) for options and dataset details.
> Missing optional data skips only the affected tasks; inspect `orcs.SKIP_REASON`.

Verify task registration:

```bash
python -c "import mjlab, orcs; from mjlab.tasks.registry import list_tasks; print('\n'.join(list_tasks()))"
```

## play

```bash
# release — download once, then load the verified public checkpoint
play Orcs-Dodge-AdaptSonic --agent release --viewer native
play Orcs-PerLoco-Grail-AdaptSonic --agent release --viewer native

# initial — construct the policy without loading a training checkpoint
play Orcs-Uolm-AdaptSonic --agent initial --viewer native

# trained — load an explicit local training checkpoint
play Orcs-PerLoco-OmRe-AdaptSonic --agent trained \
  --checkpoint-file /path/to/checkpoint.pt --viewer native

# zero — hold zero actions while inspecting the task
play Orcs-PerLoco-Grail-AdaptSonic-Smpl --agent zero --viewer native

# random — sample actions while inspecting the task
play Orcs-Uolm-SmallCubeTable-AdaptSonic-Smpl --agent random --viewer native
```

## train

```bash
train Orcs-Uolm-AdaptSonic --env.scene.num-envs 4096
train Orcs-PerLoco-Grail-AdaptSonic --env.scene.num-envs 4096
```

## paths

| variable | default |
|---|---|
| `ORCS_DATA_ROOT` | `<repo>/data` |
| `ORCS_DEPS_ROOT` | `<repo>/dependencies` |
| `ORCS_ASSETS_SOURCE` | host assets checkout, then installed `assets` package |
| `ORCS_SMPLX_DIR` | `<deps>/GRAIL/imports/GEM-SMPL/inputs/checkpoints/body_models` |
| `ORCS_RELEASE_ROOT` | `~/.cache/orcs/releases` or `$XDG_CACHE_HOME/orcs/releases` |

## license and credits

ORCS code and documentation are [BSD-3-Clause](LICENSE). Third-party code,
models, datasets, and body-model files retain their upstream terms; see
[THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md).

ORCS builds on SONIC, GRAIL, OmniRetarget, DexMachina, ResMimic, MimicKit,
mjlab, and RSL-RL. Please cite the relevant projects and datasets.
