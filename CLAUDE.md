# CLAUDE.md

Guidance for Claude Code (claude.ai/code) in this repository. **This file is an index, not
an encyclopedia** — read the linked docs; do not grow this file with findings.

## Library

| read this | for |
|---|---|
| [docs/ethos.md](docs/ethos.md) | what orcs is, the privileged-observation thesis, the `core`/`assets`/`tasks` layer contract + import rules, task slots, how to add a task |
| [readme.md](readme.md) | every command — setup, `play`/`train`, SMPL scripts, lint, env-var overrides |

One-line version: `orcs` (Oracle Robot Control Synthesis) trains privileged/oracle policies for
humanoid control on mjlab. It is **not** a task — tasks self-register under `src/orcs/tasks/`.
Today: **UOLM** (`src/orcs/tasks/uolm/`, Uni-Object Loco-Manipulation).

## Hard rules

1. **No test suite** (pytest is declared in `[dev]`, unused). Verify by running
   `play <task> --agent initial` headless and watching obs shapes + reward.
   `--agent initial` (an orcs addition) builds the real agent with NO checkpoint — the frozen
   base bit-exact for adapter tasks.
2. **Respect the layer contract** (docs/ethos.md §4): `core` never imports `assets`/`tasks`;
   no `__file__` depth math — paths come from `orcs.core.paths`.
3. **`mocke` is a public contract, not vendored code.** Its `sonic.profile` obs-term order /
   action order / future-window shape are coupled bit-for-bit to the ported SONIC checkpoints.
   Never edit mocke as part of an orcs change — it changes in mocke (+ its port script), gets
   pushed there, then `deps.lock` is repinned.
4. **Deps are edited in `dependencies/<dep>/`** and pushed to their own remotes; after any
   change, commit+push there and repin that dep's SHA in `deps.lock`. Applies to
   `mocke`, `assets`, `rsl_rl`.

## Dependency web (non-obvious)

orcs is thin; the substance lives in four pinned deps (`deps.lock`, materialized by
`scripts/setup/sync_dependencies.sh`; `/data` and `/dependencies` are gitignored):

| dep | role | notes |
|---|---|---|
| `mjlab` (PyPI/editable) | sim + manager-based env framework | `ManagerBasedRlEnvCfg`, `register_mjlab_task`, `play`/`train` |
| `mocke` (git, `pip -e`) | **frozen-WBC contract** + ported SONIC ckpts | `mocke.sonic.profile`, `mocke.mdp.joint_maps` (IL↔MJ), `PRETRAINED_DIR`; ckpts ship **tracked** — no port step |
| `rsl_rl` (lok-i fork, `pip --no-deps -e`) | the models | `SonicWithAdapterModel` (LoRA over frozen SONIC), `SonicBaseModel` |
| `assets` (git, `pip -e`) | robot + object MuJoCo assets | object XMLs are **machine-generated, untracked** — run `make_object_models.py --all` on a fresh checkout |

## UOLM mechanisms

- **SONIC frozen base**: frozen `tokenizer-encoder → FSQ quantizer → action-decoder`; a zero-init
  LoRA adapter rides the decoder (`SonicWithAdapterModel`) so construction reproduces the base
  bit-exact and PPO trains only the adapter (`freeze_base=True`, std frozen at the base ckpt's
  converged per-dim band). Streams: `policy` (proprio hist-10) + `tokenizer` (future ref window)
  from `mocke.sonic.profile`; orcs adds `augmentation` (task/object kinematics — the adapter's
  conditioning) and a privileged `critic`.
- **Joint order**: the retargeted dataset is IsaacLab BFS (IL); mjlab/MuJoCo is XML DFS (MJ). The
  port script bakes the IL→MJ permutation into the ported ckpt's first/last layers, so the runtime
  consumes/emits MJ order with no runtime converters. `mocke.mdp.joint_maps.{IL2MJ,MJ2IL}` is the
  single source of truth (used e.g. for the SMPL wrist-joint slots).
- **Two command spaces** — `uolm_env_cfg(command_space=...)` in
  [env_cfg.py](src/orcs/tasks/uolm/env_cfg.py) is the single branching factory. Same object
  plumbing, RSI, goals, terminations; what differs:

  | | `robot` (`Orcs-Uolm-AdptSonic`) | `smpl` (`Orcs-Uolm-AdptSonic-Smpl`) |
  |---|---|---|
  | reference | retargeted G1 clips (object-keyed, omni multi-object) | human SMPL clips (flat, single object) |
  | dataset root | `data/retargeted_motions/.../unitree_g1` | `data/smpl_motions` (built by `build_smpl_dataset.py`) |
  | tokenizer obs | g1 (640-d) | smpl (840-d), `mocke.sonic.sonic_smpl_tokenizer` |
  | base ckpt | `last_ported.pt` | `smpl_ported.pt` |
  | rewards/RSI | full MoTr rewards + robustness domain | **nullified — rollout only** (PR pending) |

- **SMPL frames** are a real gotcha — conventions table in [readme.md](readme.md#smpl-data-conventions).
  `smpl_joints` reach the encoder RAW; only `pose_aa` root + `transl` get y-up→z-up converted.
  [smpl_data.py](src/orcs/tasks/uolm/smpl_data.py) owns it.
- **Object collision**: all objects use convex decomposition (`cvx_dcmp`), never whole hulls —
  container-shaped hulls are a narrowphase trap (~11× slower). Hull is a per-object override only.
- **[core/_mjlab_compat.py](src/orcs/core/_mjlab_compat.py)** patches mjlab at import time
  (idempotent): (1) an isinstance sentinel so play/train don't force single-file `--motion-file`
  resolution on multi-clip commands — task cfgs arrive via `apply(multi_clip_cfgs=...)`, wired in
  [orcs/__init__.py](src/orcs/__init__.py); (2) adds `--agent initial` to `play`; (3) caps
  mujoco-warp's CCD workspace VRAM (`ORCS_NCCDMAX`, default 64); (4) muffles a cosmetic libmujoco
  mesh-support warning.
