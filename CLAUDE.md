# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this is

`orcs` (SONIC REtarget & REfine) is a **framework** for retargeting NVIDIA's frozen
**SONIC** whole-body controller (WBC) to [mjlab](https://github.com/) tasks: kinematically
retarget a reference motion, then dynamically refine with a LoRA adapter over the frozen base
(PPO). orcs is **not itself a task** — each task is a self-registering sub-package under
`src/orcs/`. The only task today is **UOLM** (`src/orcs/tasks/uolm/`, Uni-Object Loco-Manipulation).

## Setup & commands

```bash
pip install -e .                                    # core (pulls mjlab)
bash scripts/setup/sync_dependencies.sh             # clone+install deps.lock entries (needs git-lfs)
python dependencies/assets/source/omni_objects/make_object_models.py --all   # object XMLs (untracked)
bash scripts/setup/let_there_be_light.sh            # optional: .claude/.vscode config for this machine

ruff check src scripts                              # lint (config: pyproject [tool.ruff.lint])
python -c "import orcs; from mjlab.tasks.registry import list_tasks; print(list_tasks())"  # verify registration

# mjlab CLIs (installed by mjlab): task-id is what register_mjlab_task() names
play  Orcs-Uolm --agent initial --viewer native   # frozen base, no ckpt; also zero|random|trained
train Orcs-Uolm --num_envs 4096

# SMPL command space (rollout-only; rewards/RSI unsupported)
python scripts/rollout_smpl.py --smpl <clip.pkl> --viewer native     # single-clip quicktest
python scripts/build_smpl_dataset.py --src <dir-of-pkls>             # -> data/smpl_motions/
play Orcs-Uolm-Smpl --agent initial --viewer native                # multi-clip over built dataset
```

There is **no test suite** (pytest is declared in `[dev]` but unused). Verification is done by
running the rollout scripts / `play <task> --agent initial` headless and watching obs shapes +
reward. `--agent initial` (a orcs addition, see below) rolls the freshly-constructed real agent
with NO training checkpoint — the frozen base bit-exact for adapter tasks.

## Dependency web (critical, non-obvious)

orcs is thin; the substance lives in four pinned dependencies (`deps.lock`, materialized by
`scripts/setup/sync_dependencies.sh`; both `/data` and `/dependencies` are gitignored):

| dep | role | notes |
|---|---|---|
| `mjlab` (external, PyPI/editable) | the sim + manager-based env framework | `ManagerBasedRlEnvCfg`, `register_mjlab_task`, `play`/`train` scripts |
| `mocke` (git, `pip -e`) | **frozen-WBC contract** + ported SONIC checkpoints | `mocke.sonic.profile` (obs/action/robot layout), `mocke.mdp.joint_maps` (IL↔MJ), `PRETRAINED_DIR` |
| `rsl_rl` (lok-i fork, `pip --no-deps -e`) | the models | `SonicWithAdapterModel` (LoRA over frozen SONIC), `SonicBaseModel` |
| `assets` (git, `pip -e`) | robot + object MuJoCo assets | object XMLs are **machine-generated and untracked** — must run `make_object_models.py --all` on a fresh checkout |

**`mocke` is a public contract, not vendored code.** Its `sonic.profile` obs-term order / action
order / future-window shape are coupled bit-for-bit to the ported SONIC checkpoints. Do not edit
mocke as part of a orcs change; if the SONIC I/O layout must change, it changes in mocke (+ the
port script) and gets pushed there, then `deps.lock` is repinned. The SONIC checkpoints
(`last_ported.pt` g1-mode, `smpl_ported.pt` smpl-mode) ship **tracked** in mocke's `pretrained/` —
no download/port step needed for a fresh checkout.

## Task registration flow

1. `pyproject.toml` declares entry point `[project.entry-points."mjlab.tasks"] orcs = "orcs"`.
   mjlab imports `orcs` at startup.
2. `src/orcs/__init__.py` (framework) does `import orcs.tasks.uolm` (task self-registers) then applies
   the mjlab compat shim once for all tasks.
3. `src/orcs/tasks/uolm/__init__.py` calls `register_mjlab_task(...)` for `Orcs-Uolm` and
   `Orcs-Uolm-Smpl`, wrapped in `try/except FileNotFoundError` so a checkout missing
   assets/motions still imports (registration is skipped with a warning).

**Adding a task**: drop a sibling package `src/orcs/<task>/` that self-registers on import, add one
`import orcs.<task>` line to `src/orcs/__init__.py`. Framework pieces shared across tasks
(`assets.py`, `_mjlab_compat.py`) stay at the top level; task-specific env/rl/mdp live in the package.

## SONIC frozen-base architecture

The model is a frozen `tokenizer-encoder → FSQ quantizer → action-decoder` stack; orcs straps a
zero-init LoRA adapter on the decoder (`SonicWithAdapterModel`), so at construction it reproduces
the base bit-exact, and PPO only trains the adapter (`freeze_base=True`, std frozen at the base
ckpt's converged per-dim band). Two obs streams feed it, both from `mocke.sonic.profile`:
`policy` (proprio history-10) and `tokenizer` (future reference window). orcs adds an
`augmentation` stream (task/object kinematics, the adapter's conditioning) and a privileged
`critic` stream.

**Joint order**: the retargeted dataset is IsaacLab BFS order (IL); mjlab/MuJoCo is XML DFS order
(MJ). The port script bakes the IL→MJ permutation into the ported checkpoint's first/last layers,
so the runtime consumes/emits MJ order with no runtime converters. `mocke.mdp.joint_maps.{IL2MJ,MJ2IL}`
is the single source of truth for this remap (used e.g. for the SMPL wrist-joint slots).

## The two command spaces (`command_space` on the UOLM motion command)

Same object plumbing, RSI, goals, terminations; the difference is what drives the reference motion
and which SONIC encoder runs:

| | `robot` (`Orcs-Uolm`) | `smpl` (`Orcs-Uolm-Smpl`) |
|---|---|---|
| reference | retargeted G1 clips (object-keyed, omni multi-object) | human SMPL clips (flat, single object) |
| dataset root | `data/retargeted_motions/.../unitree_g1` | `data/smpl_motions` (built by `build_smpl_dataset.py`) |
| tokenizer obs | g1 (640-d) | smpl (840-d), `mocke.sonic.sonic_smpl_tokenizer` |
| base ckpt | `last_ported.pt` | `smpl_ported.pt` |
| rewards/RSI | full MoTr rewards + robustness domain | **nullified — rollout only** (PR pending) |

`uolm_env_cfg(command_space=...)` in `src/orcs/tasks/uolm/env_cfg.py` is the single factory that branches.
`_resolve_smpl_motions()` degrades gracefully when `data/smpl_motions` is absent so registration never
requires contributor-built data.

## SMPL data convention (a real gotcha)

Per gear_sonic's split (`dependencies/GR00T-WholeBodyControl/docs/source/references/conventions.md`):
`smpl_joints` (T,24,3) ship **already z-up, root-centered, RAW** and go to the encoder unconverted —
SONIC never transforms them. Only `pose_aa`'s root and `transl` are SMPL-native **y-up** and get
converted (90° about +X, "y is the z for smpl") for the root quaternion and the ghost. Quaternions
are wxyz throughout; 6D rotations are the first two columns of the rotation matrix.
`src/orcs/tasks/uolm/smpl_data.py` (`load_smpl_clip`, `stage_clip`) owns this conversion + the flat
dataset staging (`<root>/<clip>/sample0/{motion,smpl_motion,object_motion,contact_matrix}.npz`).

## Other orcs-specific mechanics

- **`src/orcs/_mjlab_compat.py`** patches mjlab at import time (idempotent): (1) an isinstance
  sentinel so mjlab's play/train scripts don't force single-file `--motion-file` resolution on the
  multi-clip `OmniObjectMotionCommand`; (2) adds `--agent initial` to `play`; (3) caps mujoco-warp's
  CCD workspace VRAM (`ORCS_NCCDMAX`, default 64) — undershoot prints "CCD overflow" to stderr, raise
  the env var if seen; (4) muffles a cosmetic libmujoco mesh-support warning.
- **Object collision**: all objects use convex decomposition (`cvx_dcmp`), not whole hulls —
  container-shaped hulls are a narrowphase trap (~11× slower). Hull is a per-object override only.
- **mocke is edited in `dependencies/mocke/`** and pushed to its own remote; after any mocke change,
  commit+push there and repin `deps.lock`'s mocke SHA. Same for the `assets`/`rsl_rl` deps.
