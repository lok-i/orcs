# <img src="docs/media/logo.png" alt="" height="20"> orcs

*Optimize, Retarget, Control Suite*

training and retargeting tools for privileged humanoid control:

- LoRA adaptation of [SONIC](https://nvlabs.github.io/GEAR-SONIC/)
- kinodynamic retargeting of SMPL motions
- adapter and experimental *tabula rasa* training

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

## install

Requires [uv](https://docs.astral.sh/uv/getting-started/installation/), Git LFS,
and GitHub SSH access.

```bash
git clone https://github.com/lok-i/orcs && cd orcs
uv venv --prompt orcs
source .venv/bin/activate
```

full:

```bash
bash scripts/setup/sync_deps.sh
bash scripts/setup/sync_data.sh
```

lean:

```bash
bash scripts/setup/sync_deps.sh --no-smpl
bash scripts/setup/sync_data.sh --no-smpl
```

- `sync_data.sh`: default `all`; modes `inhouse`, `omre`, `grail`
- `--no-smpl`: non-SMPL tasks + roster-selected retargeted motions
- full setup: prompts for the three licensed SMPL-X model files
- dependency changes: rerun `sync_deps.sh` with the same mode
- missing optional data: affected tasks skip; inspect `orcs.SKIP_REASON`
- details: [PerLoco](docs/perceptive_locomotion.md) ·
  [SMPL retargeting](docs/smpl_retargeting.md)

## play

```bash
# release
play Orcs-Dodge-AdaptSonic --agent release --viewer native

# initial
play Orcs-Uolm-AdaptSonic --agent initial --viewer native

# trained
play Orcs-PerLoco-OmRe-AdaptSonic --agent trained \
  --checkpoint-file /path/to/checkpoint.pt --viewer native

# zero
play Orcs-PerLoco-Grail-AdaptSonic-Smpl --agent zero --viewer native

# random
play Orcs-Uolm-SmallCubeTable-AdaptSonic-Smpl --agent random --viewer native
```

- releases: [`lkrajan/orcs`](https://huggingface.co/lkrajan/orcs); fetched on use
- all releases: `bash scripts/setup/download_released_models.sh`
- trained checkpoints: `--checkpoint-file` or `--wandb-run-path`

## train

```bash
train Orcs-Uolm-AdaptSonic --env.scene.num-envs 4096
train Orcs-PerLoco-Grail-AdaptSonic --env.scene.num-envs 4096
```

## retarget

```bash
orcs-pseudo-retarget --scene perloco-grail --all
orcs-pseudo-retarget --scene uolm --all
```

- `--all`: every sample in the scene's default motion sets
- more sets and options: [SMPL retargeting](docs/smpl_retargeting.md)

## paths

| variable | default |
|---|---|
| `ORCS_DATA_ROOT` | `<repo>/data` |
| `ORCS_DEPS_ROOT` | `<repo>/dependencies` |
| `ORCS_ASSETS_SOURCE` | host assets checkout, then installed `assets` package |
| `ORCS_SMPLX_DIR` | `<deps>/body_models` |
| `ORCS_RELEASE_ROOT` | `~/.cache/orcs/releases` or `$XDG_CACHE_HOME/orcs/releases` |

## license and credits

ORCS code and documentation are [BSD-3-Clause](LICENSE). Third-party code,
models, datasets, and body-model files retain their upstream terms; see
[THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md).

ORCS builds on SONIC, GRAIL, OmniRetarget, DexMachina, ResMimic, MimicKit,
mjlab, and RSL-RL. Please cite the relevant projects and datasets.

## citation

`orcs` was developed as part of [ViBe](https://arxiv.org/abs/2609.09918). If you
use this repository in your research, please consider citing:

```bibtex
@misc{krishna2026vibe,
  title={ViBe: Visual Behavior Adaptation for Perceptive Humanoid Whole-Body Control},
  author={Lokesh Krishna and Sarvesh Venkatesan and An Zhang and Quan Nguyen},
  year={2026},
  eprint={2609.09918},
  archivePrefix={arXiv},
  primaryClass={cs.RO},
  url={https://arxiv.org/abs/2609.09918},
}
```
