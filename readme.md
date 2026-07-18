# sortr — SONIC REtarget & REfine

A framework for kinematically retarget a reference, dynamically refine with a LoRA
adapter by adapting **SONIC** whole-body controller to mjlab
tasks: . 

## Setup

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

# 5. verify — expect Sortr-Uolm and Sortr-Uolm-Smpl
python -c "import sortr, mjlab.tasks; from mjlab.tasks.registry import list_tasks; print(list_tasks())"
```

## Play / Train — robot command space (`Sortr-Uolm`)

```bash
play  Sortr-Uolm --agent initial   # frozen base, no ckpt; also zero|random|trained
train Sortr-Uolm --num_envs 4096
```

`--agent initial` = the task's real agent (frozen SONIC base + zero-init LoRA),
no training checkpoint → rolls the frozen base bit-exact.

## SMPL command space (`Sortr-Uolm-Smpl`)

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

**Persistent dataset** — convert a directory of SONIC smpl pkls, then play:

```bash
python scripts/build_smpl_dataset.py \
  --src dependencies/GR00T-WholeBodyControl/sample_data/smpl_filtered
# -> data/smpl_motions/<clip>/sample0/*.npz

play Sortr-Uolm-Smpl --agent initial --viewer native   # multi-clip rollout
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

`sortr.uolm.smpl_data.load_smpl_clip` handles the conversion; the staged
`smpl_motion.npz` carries `smpl_joints` (RAW) + `smpl_root_quat_w` (z-up, wxyz,
base-rot removed) + `smpl_joints_viz_w` (z-up world, ghost only). G1 wrist refs
ride `motion.npz` `joint_pos` (zeros OK — degraded wrist orientation only).

## Adding a task

Drop a sibling package under `src/sortr/` that registers its envs on import,
then add one line to `src/sortr/__init__.py`. Framework pieces shared across
tasks (`assets.py`, `_mjlab_compat.py`) stay at the top level.
