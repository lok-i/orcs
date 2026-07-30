# orcs

**o**racle **r**obot **c**ontrol **s**ynthesis

privileged-observation oracle policies for humanoid control on [mjlab](https://github.com/mujocolab/mjlab). 

`orcs` currently supports the following tasks
1. `omni-object` locomanipualtion 
2. `perceptive locomtoion` (todo)
 
## setup

Run these from the repo root, inside your project env (conda or uv, Python 3.11):

```bash
# 1. core package (pulls mjlab)
pip install -e .

# 2. custom deps + data (assets, retargeted_motions, mocke[+ckpts], rsl_rl fork)
#    reads deps.lock; idempotent; needs git-lfs on PATH
bash scripts/setup/sync_dependencies.sh

# 3. generate object collision/visual XMLs (machine-generated, not tracked)
python dependencies/assets/source/omni_objects/make_object_models.py --all

# 4. (optional) editor + Claude config for this machine
bash scripts/setup/let_there_be_light.sh

# 5. verify — expect Orcs-Uolm-AdptSonic, -TaRa, -AdptSonic-Smpl
python -c "import orcs, mjlab.tasks; from mjlab.tasks.registry import list_tasks; print(list_tasks())"
```

Step 5 printing `[orcs.tasks.uolm] skipping task registration: ...` means step 2
or 3 is incomplete — registration degrades instead of breaking `import orcs`.

**Order matters, and re-running step 1 alone undoes step 2.** mjlab pins
`rsl-rl-lib==5.4.0`; our fork declares `5.4.1`, so any pip run that re-resolves
mjlab's dependencies replaces the editable fork with vanilla 5.4.0 from PyPI —
silently. For re-installs after step 2, use:

```bash
pip install --no-deps -e .          # re-install orcs, touch nothing else
python -c "import rsl_rl; print(rsl_rl.__file__)"   # must be dependencies/rsl_rl/
```

If it points into `site-packages`, the fork was clobbered — re-run step 2.

## play / Train — robot command space (`Orcs-Uolm-AdptSonic`)

```bash
play  Orcs-Uolm-AdptSonic --agent initial   # frozen base, no ckpt; also zero|random|trained
train Orcs-Uolm-AdptSonic --num_envs 4096

train Orcs-Uolm-TaRa --num_envs 4096       # the no-frozen-base floor, same env
```

`--agent initial` = the task's real agent (frozen SONIC base + zero-init LoRA),
no training checkpoint → rolls the frozen base bit-exact. There is no test
suite; this is how you verify a change (watch obs shapes + reward).

## SMPL command space (`Orcs-Uolm-AdptSonic-Smpl`)

Rollout-only for now — rewards and RSI are unsupported (pending a separate PR);
the env nullifies rewards and rolls the frozen base over SMPL motion.

**Quicktest** — roll one clip (self-contained, stages a scratch dataset):

```bash
# a shipped SONIC sample
python scripts/rollout_smpl.py \
  --smpl dependencies/GR00T-WholeBodyControl/sample_data/smpl_filtered/walk_forward_amateur_001__A001_M.pkl \
  --viewer native

python scripts/rollout_smpl.py            # no args -> synthetic standing clip
```

** persistent dataset** — convert a directory of SONIC smpl pkls, then play:

```bash
python scripts/build_smpl_dataset.py \
  --src dependencies/GR00T-WholeBodyControl/sample_data/smpl_filtered
# -> data/smpl_motions/<clip>/sample0/*.npz

play Orcs-Uolm-AdptSonic-Smpl --agent initial --viewer native   # multi-clip rollout
```

Object motion is a static nominal placeholder unless a matching object npz is
passed (`--object-dir <dir>`, matched by clip stem) — no smpl+object clips exist
yet, so the object stream is a placeholder to exercise the plumbing.

### SMPL data conventions

Per gear_sonic's split (see `dependencies/GR00T-WholeBodyControl/docs/source/references/conventions.md`):

| field | frame | used for |
|---|---|---|
| `smpl_joints` (T,24,3) | **z-up, root-centered, RAW** | encoder input (never converted) |
| `pose_aa` root, `transl` | SMPL-native **y-up** | converted to z-up for root quat / ghost |

`orcs.tasks.uolm.smpl_data.load_smpl_clip` handles the conversion; the staged
`smpl_motion.npz` carries `smpl_joints` (RAW) + `smpl_root_quat_w` (z-up, wxyz,
base-rot removed) + `smpl_joints_viz_w` (z-up world, ghost only). G1 wrist refs
ride `motion.npz` `joint_pos` (zeros OK — degraded wrist orientation only).

## Lint

```bash
ruff check src scripts     # config: pyproject [tool.ruff.lint]
```

## Environment overrides

Paths resolve through `orcs.core.paths` — override a root instead of moving files:

| var | default | notes |
|---|---|---|
| `ORCS_ROOT` | walk up for `pyproject.toml`+`deps.lock` | repo root |
| `ORCS_DATA_ROOT` | `<repo>/data` | datasets |
| `ORCS_DEPS_ROOT` | `<repo>/dependencies` | synced deps |
| `ORCS_ASSETS_SOURCE` | `<deps>/assets/source`, else the installed `assets` pkg | raises if set but wrong |
| `ORCS_NCCDMAX` | `64` | mujoco-warp CCD workspace rows/world; raise it if stderr shows "CCD overflow" |

## Adding a task

Drop a package under `src/orcs/tasks/` that registers its envs on import, then
add one line to `src/orcs/__init__.py`. Layer contract and full checklist:
[docs/ethos.md](docs/ethos.md).
