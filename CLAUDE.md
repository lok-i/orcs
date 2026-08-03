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

| task | what | docs |
|---|---|---|
| **UOLM** `tasks/uolm/` | Uni-Object Loco-Manipulation — adapter reads OBJECT kinematics | this file, §UOLM mechanisms |
| **PerLoco** `tasks/perloco/` | Perceptive Locomotion over staged (terrain, motion) pairs — adapter reads a TERRAIN height scan | [docs/perceptive_locomotion.md](docs/perceptive_locomotion.md), [tasks/perloco/readme.md](src/orcs/tasks/perloco/readme.md) |

The two differ in exactly ONE obs group (`augmentation`). Everything else — frozen base,
multi-clip command, agents, critic — is literally shared code. That is the thesis, not a
coincidence; a term duplicated across the two is a bug.

**Agents live in `core/rl.py`, never in a task.** `adapt_sonic_agent_cfg` / `tara_agent_cfg` /
`sidecar_agent_cfg` — a task picks one and names its experiment. There is no `rl_cfg.py` under
any task, and adding one back is how the two definitions of "PPO" start to drift.
(`tests/test_registration.py` asserts the absence.)

**Executable code lives in `orcs/cli/`, never in `scripts/`.** `scripts/` does not ship in a
wheel, so logic there is unreachable from a `pip install`; the four `scripts/*.py` are
three-line wrappers over `[project.scripts]` entry points and `tests/test_packaging.py` keeps
them that way. Same rule for non-`.py` runtime files (`rosters/*.toml`): declare them in
`[tool.setuptools.package-data]` or they exist only in your checkout.

## Hard rules

1. **`pytest` tests CONTRACTS, not behavior** (`tests/`, ~30 s, no GPU): that a
   non-editable install still ships the rosters + CLI, and that `import orcs`
   degrades to `SKIP_REASON` instead of raising. Behavior is verified by running
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
5. **When orcs is a dependency, the CONSUMER's lock wins.** `mocke`/`rsl_rl`/`assets` are one
   editable install per venv, so a consumer (today: `vibe`) decides their SHAs and this
   lock is what a STANDALONE orcs checkout uses. Both currently agree.
   `core/deps.py` records what orcs was validated against and prints drift at import —
   believe it, especially for `mocke`. Dev on orcs alone belongs in its own venv; running
   `sync_dependencies.sh` inside a consumer's venv silently re-points the shared deps.
   **Unmerged:** the `assets` commits this lock used to pin (`6822f3b`, `46511a8` —
   optimized tire/woodchair2/largetable decompositions) are not in the pinned SHA; they
   want merging in the assets repo.

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

  | | `robot` (`Orcs-Uolm-AdaptSonic`) | `smpl` (`Orcs-Uolm-AdaptSonic-Smpl`) |
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
