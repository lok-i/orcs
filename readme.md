# orcs

privileged oracle policies for humanoid control on [mjlab](https://github.com/mujocolab/mjlab). 

`orcs` currently supports the following tasks

| task | ids | what the adapter reads |
|---|---|---|
| uni-object loco-manipulation | `Orcs-Uolm-*` | object kinematics |
| perceptive locomotion | `Orcs-PerLoco-*` | a terrain height scan |


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

# 5. verify — expect Orcs-Uolm-* and (once terrain is staged) Orcs-PerLoco-*
python -c "import orcs, mjlab.tasks; from mjlab.tasks.registry import list_tasks; print(list_tasks())"
```

Step 5 printing `[orcs.tasks.uolm] skipping task registration: ...` means step 2
or 3 is incomplete — registration degrades instead of breaking `import orcs`.

### perceptive locomotion (optional)

Only `Orcs-PerLoco-*` needs this. Skip it and those tasks stay unregistered;
everything else works.

```bash
bash scripts/setup/perceptive_locomotion.sh          # both sources, ~1.5 GB
bash scripts/setup/perceptive_locomotion.sh --help   # per-source / no-SMPL / resume flags
```

It sparse-clones the two source datasets, shallow-clones the GRAIL code repo,
**pauses** for you to drop in the licensed SMPL-X body models, then stages
exactly the tiles the rosters name. Idempotent and resumable — a failed
download is a re-run, not a restart.

| step | fetches | into |
|---|---|---|
| 1 | OmniRetarget `robot-terrain.zip` + `models/` (~125 MB) | `$ORCS_DATA_ROOT/OmniRetarget_Dataset` |
| 1 | GRAIL `curb/{robot,object_usd,recon}`, blobs scoped to the ROSTERED families (~70 MB of 12 GB) | `$ORCS_DATA_ROOT/PhysicalAI-Robotics-Locomanipulation-GRAIL` |
| 2 | [NVlabs/GRAIL](https://github.com/NVlabs/GRAIL) depth-1, no submodules (reference: retargeter + vendored SONIC) | `$ORCS_DEPS_ROOT/GRAIL` |
| 3 | **you**: SMPL-X v1.1 NPZ from [smpl-x.is.tue.mpg.de](https://smpl-x.is.tue.mpg.de) | `$ORCS_SMPLX_DIR/smplx/SMPLX_NEUTRAL.npz` |
| 4 | staging (`stage_terrain_motions.py`, families/levels read from the roster) | `$ORCS_DATA_ROOT/terrain_motions/<source>` |

Step 3 is the only manual one — SMPL-X is licensed, so it cannot be fetched for
you. `--no-smpl` skips it, at the cost of `Orcs-PerLoco-Grail-AdaptSonic-Smpl`.
Which tiles get staged comes from `src/orcs/tasks/perloco/rosters/<source>.toml`
— the same file the env builds its grid from, so edit the roster, re-run, done.

**Two filters, and missing either one downloads the whole dataset.**
`sparse-checkout` decides which *pointers* land in the working tree;
`git lfs pull --include` decides which *blobs* get fetched — and **`git lfs
pull` does not read sparse-checkout**, so an unscoped pull on GRAIL fetches
106k objects / 12+ GB no matter how narrow the tree is. The script scopes both,
and its LFS include is built from the roster's family list, so a roster edit
re-scopes the download too.

**Order matters, and re-running step 1 alone undoes step 2.** mjlab pins
`rsl-rl-lib==5.4.0`; our fork declares `5.4.1`, so any pip run that re-resolves
mjlab's dependencies replaces the editable fork with vanilla 5.4.0 from PyPI —
silently. For re-installs after step 2, use:

```bash
pip install --no-deps -e .          # re-install orcs, touch nothing else
python -c "import rsl_rl; print(rsl_rl.__file__)"   # must be dependencies/rsl_rl/
```

If it points into `site-packages`, the fork was clobbered — re-run step 2.

## play / Train — robot command space (`Orcs-Uolm-AdaptSonic`)

```bash
play  Orcs-Uolm-AdaptSonic --agent initial   # frozen base, no ckpt; also zero|random|trained
train Orcs-Uolm-AdaptSonic --num_envs 4096

train Orcs-Uolm-TaRa --num_envs 4096       # the no-frozen-base floor, same env
```

`--agent initial` = the task's real agent (frozen SONIC base + zero-init LoRA),
no training checkpoint → rolls the frozen base bit-exact. There is no test
suite; this is how you verify a change (watch obs shapes + reward).

## SMPL command space (`Orcs-Uolm-AdaptSonic-Smpl`)

Rollout-only for now — rewards and RSI are unsupported (pending a separate PR);
the env nullifies rewards and rolls the frozen base over SMPL motion.

**Quicktest** — roll one clip (self-contained, stages a scratch dataset):

```bash
orcs-rollout-smpl                                   # synthetic standing clip
orcs-rollout-smpl --smpl <clip>.pkl --viewer native # a SONIC smpl pkl
```

**Persistent dataset** — convert a directory of SONIC smpl pkls, then play:

```bash
orcs-build-smpl --src <dir-of-smpl-pkls>
# -> data/smpl_motions/<clip>/sample0/*.npz

play Orcs-Uolm-AdaptSonic-Smpl --agent initial --viewer native   # multi-clip rollout
```

Source pkls are **not** in `deps.lock` — bring your own (SONIC's
`sample_data/smpl_filtered`, or any pkl with `pose_aa`/`transl`/`smpl_joints`).
Every `orcs-*` command has a `python scripts/<name>.py` twin.

Object motion is a static nominal placeholder unless a matching object npz is
passed (`--object-dir <dir>`, matched by clip stem) — no smpl+object clips exist
yet, so the object stream is a placeholder to exercise the plumbing.

### SMPL data conventions

Per gear_sonic's split:

| field | frame | used for |
|---|---|---|
| `smpl_joints` (T,24,3) | **z-up, root-centered, RAW** | encoder input (never converted) |
| `pose_aa` root, `transl` | SMPL-native **y-up** | converted to z-up for root quat / ghost |

**The trap, and it costs a debugging round every time.** The encoder computes
`quat_apply_inverse(root_q, joints)`, so joints and root must land in ONE
frame — a mismatch never errors, it silently redefines the leading 72 dims as a
body frame that rotates with heading. gear_sonic's `smpl_y_up` flag converts
the **root only** (`motion_lib_base.py` loads `smpl_joints` untouched), so a pkl
with a y-up `pose_aa` still ships z-up joints. Measured on 63 GRAIL curb clips,
zero-shot tracking reward: z-up **4.735** vs y-up **0.348** (g1 encoder 5.677).
The contract lives once in `orcs.core.data.smpl`.

`orcs.tasks.uolm.smpl_data.load_smpl_clip` handles the conversion; the staged
`smpl_motion.npz` carries `smpl_joints` (RAW) + `smpl_root_quat_w` (z-up, wxyz,
base-rot removed) + `smpl_joints_viz_w` (z-up world, ghost only). G1 wrist refs
ride `motion.npz` `joint_pos` (zeros OK — degraded wrist orientation only).

## PerLoco — perceptive locomotion (`Orcs-PerLoco-*`)

Same frozen SONIC base and LoRA adapter as UOLM; the adapter reads a TERRAIN
height scan instead of object kinematics. Data first — see
[perceptive locomotion (optional)](#perceptive-locomotion-optional) above; full
usage in [tasks/perloco/readme.md](src/orcs/tasks/perloco/readme.md).

Ids read `Orcs-PerLoco-<Source>-<Agent>[-<CommandSpace>]`:

| task | source | agent / reference |
|---|---|---|
| `Orcs-PerLoco-OmRe-AdaptSonic` | OmniRetarget climb | frozen SONIC + LoRA on the height scan |
| `Orcs-PerLoco-OmRe-TaRa` | " | from-scratch floor |
| `Orcs-PerLoco-Grail-AdaptSonic` | GRAIL curb | " |
| `Orcs-PerLoco-Grail-AdaptSonic-Smpl` | " | ...encoder reads the HUMAN, not the retarget |
| `Orcs-PerLoco-Grail-TaRa` | " | from-scratch floor |

```bash
orcs-view-terrain --source omni                         # inspect + curate -> :8080

play  Orcs-PerLoco-OmRe-AdaptSonic --num-envs 10 --agent initial
train Orcs-PerLoco-Grail-AdaptSonic --num_envs 4096
```

Tasks are skipped (never raised) when the staging directory is absent — **per
task**, so one unstaged dataset costs one row:

```bash
python -c "import orcs; print(orcs.tasks.perloco.SKIP_REASON)"   # {} = all good
```

## Lint + test

```bash
ruff check src scripts tests   # config: pyproject [tool.ruff.lint]
pytest                         # packaging + registration contracts, ~30 s, no GPU
```

`pytest` does not test behavior — that is `play <task> --agent initial`. It
tests the two things a checkout cannot self-report: that a **non-editable**
install still contains everything the runtime needs, and that `import orcs`
degrades to `SKIP_REASON` instead of raising.

## Environment overrides

Paths resolve through `orcs.core.paths` — override a root instead of moving files:

| var | default | notes |
|---|---|---|
| `ORCS_ROOT` | walk up for `pyproject.toml`+`deps.lock` | repo root |
| `ORCS_DATA_ROOT` | `<repo>/data` | datasets |
| `ORCS_DEPS_ROOT` | `<repo>/dependencies` | synced deps |
| `ORCS_ASSETS_SOURCE` | `<deps>/assets/source`, else the installed `assets` pkg | raises if set but wrong |
| `ORCS_SMPLX_DIR` | `<deps>/GRAIL/imports/GEM-SMPL/inputs/checkpoints/body_models` | licensed SMPL-X models; staging only |
| `ORCS_NCCDMAX` | `64` | mujoco-warp CCD workspace rows/world; raise it if stderr shows "CCD overflow" |
| `ORCS_SKIP_DEP_CHECK` | unset | skips the shared-dep git probe at import (2 subprocesses; matters per-rank on a cluster FS) |

## Adding a task

Drop a package under `src/orcs/tasks/` that registers its envs on import, then
add one line to `src/orcs/__init__.py`. Layer contract and full checklist:
[docs/ethos.md](docs/ethos.md).

## Consuming orcs from another repo

`pip install -e dependencies/orcs` (or `git+…`). Three things a consumer gets
that are easy to miss:

| | |
|---|---|
| **paths auto-resolve** | `orcs.core.paths` walks up for the nearest repo root that HAS `data/` — a host vendoring orcs under `dependencies/` is found with no env vars and no import-order coupling |
| **your lock wins** | `mocke`/`rsl_rl`/`assets` are one editable install per env, so the CONSUMER decides their SHAs. `orcs.core.deps` prints drift at import — believe it, especially for `mocke` (bit-coupled to the ported SONIC ckpts) |
| **the data pipeline ships** | `orcs-stage-terrain` & co. are package entry points, not `scripts/` — available from a wheel |
