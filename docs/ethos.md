# ethos

Why orcs exists, what it believes, and how the tree is shaped. Usage lives in
[readme.md](../readme.md); this file is the mental model.

## 1. name

| | |
|---|---|
| `orcs` | **O**racle **R**obot **C**ontrol **S**ynthesis |
| oracle | a policy that sees what the robot cannot — full sim state |
| synthesis | we *build* the oracle, then hand its competence down |
| why "orc" | one horde, many bodies. Tasks are members, not forks. |

## 2. the thesis — privilege first, deployability second

Learning control under onboard-sensing limits is two problems fused: *what to do*
and *what can I know*. Solve them apart.

```
       full sim state                        onboard only
  ┌────────────────────────┐            ┌────────────────────┐
  │  ORACLE POLICY         │  distill   │  STUDENT           │
  │  any  sim state        │ ─────────► │  proprio + percep  │
  │  contact forces        │  (roadmap) │  learned estimator │
  │  future reference      │            │                    │
  │  privileged critic     │            │  (critic dropped)  │
  └────────────────────────┘            └────────────────────┘
        trains fast, cannot ship            ships, must be taught
```

Consequences we accept:

| claim | consequence |
|---|---|
| privilege makes the credit-assignment problem tractable | train the oracle first, always |
| an oracle that cannot ship is still worth training | "not deployable" is not "not useful" |
| the sensing gap is a *separate* learning problem | distillation is a first-class stage, not a patch |
| a frozen competent base beats a blank slate | adapt, don't retrain (§3) |

## 3. where privilege sits today

Status is explicit — orcs is early. Nothing below is aspirational except where marked.

| stream | content | privileged? | status |
|---|---|---|---|
| `policy` | proprio history-10 | no | ✅ frozen SONIC contract |
| `tokenizer` | future reference window | yes (future) | ✅ frozen SONIC contract |
| `augmentation` | task kinematics — the adapter's conditioning | yes (needs perception on hw) | ✅ |
| `critic` | full state | yes (never deployed) | ✅ asymmetric actor-critic |
| student | proprio + perception, distilled | — | ⬜ roadmap |

So today's privilege is **asymmetric actor-critic + ground-truth task state**:
object state for UOLM, terrain scans for PerLoco, and ball state for Dodge. The
distillation stage that closes the sensing gap is the next build, not a shipped
feature.

**Adapt, don't retrain.** A task straps a zero-init LoRA adapter onto a frozen
competent base (SONIC WBC) — at construction it reproduces the base bit-exact,
and PPO only moves the adapter. Verify this claim, don't trust it:
`play <task> --agent initial` rolls the freshly-built agent with no checkpoint.

## 4. layer contract

```
src/orcs/
├── __init__.py   registry point: import each task, wire the mjlab shim
├── core/         task-blind, shared robot/control infra
│   ├── paths · deps · _mjlab_compat      no semantics at all
│   ├── data/     scan · concatenated timelines · SMPL seed/point contracts
│   ├── mdp/      commands (MultiClipMotionCommand) · observations
│   │             · terminations · events
│   ├── obs.py    group plumbing + robot-only term bundles
│   ├── rl.py     PPO runner spine + actor builders
│   ├── registry.py per-task registration that degrades, never raises
│   └── sensors.py  robot<->ground contact + kill-body vocabulary
├── assets/       reusable robot/object/support entity and scene cfgs
├── tasks/        one self-registering package per task
│   ├── dodge/    env_cfg · observation_cfgs · mdp/
│   ├── perloco/  env_cfg · terrain · terrain_spec · sensors · sources/ · mdp/
│                 · roster.py + rosters/*.toml (shipped as package-data)
│   └── uolm/     env_cfg · robustness · sensors · sources/ · mdp/
└── cli/          console entry points — the data pipeline, INSIDE the package
```

Import rules — enforced by review and `tests/test_registration.py`:

| layer | may import | must never import |
|---|---|---|
| `core` | stdlib, mjlab, mocke | `orcs.assets`, `orcs.tasks` |
| `assets` | `orcs.core` | `orcs.tasks` |
| `tasks` | `orcs.core`, `orcs.assets`, sibling-free | another task |
| `cli` | everything | — (nothing imports FROM it) |
| `__init__` | everything (the only wiring point) | — |

**`scripts/` is not a layer.** It does not ship in a wheel, so anything with
logic in it is unreachable from a `pip install`. Executable code lives in
`orcs/cli/` behind a `[project.scripts]` entry point; `scripts/*.py` are
three-line wrappers, and `tests/test_packaging.py` keeps them that way.

Two rules earn their keep:

1. **No `__file__` depth math.** Every path comes from `orcs.core.paths`. Moving
   a module can never silently orphan a dataset.
2. **`core` is task-blind shared control infrastructure.** It may know the
   common robot morphology, joints, bodies, clips, source points, and reference
   timelines. It must **not** know what an *object*, *terrain*, *table*, or
   *ball* means to a task. The mjlab compat shim needs a task's command cfg, so
   it takes the common base class as an argument —
   `apply(multi_clip_cfgs=...)`, wired in `orcs/__init__.py`. Core never reaches
   upward.

   > **Amended 2026-08-02.** Rule 2 used to read "zero semantics". That held only
   > while core carried no terms, which held only while there was one task. The
   > second task (perloco) needs the same proprio bundle, the same runner spine,
   > the same RSI/annealing/freeze machinery — and §4 forbids it importing uolm to
   > get them, which is the pressure working as intended. So core now carries
   > robot semantics, and the line moved to where it can actually be checked:
   > grep `core/` for "object" or "terrain".
   >
   > **And then it failed its own grep.** `core/sensors.py` exported
   > `terrain_contact_sensor` / `TERRAIN_CONTACT_SENSOR_NAME` — 13 hits, a rule
   > that only looked enforced. Renamed to `ground_contact_sensor` /
   > `GROUND_CONTACT_SENSOR_NAME`: what core knows is that the robot stands on
   > *something*. `"terrain"` survives there once, as mjlab's BODY name in a
   > `ContactMatch` — an mjlab fact, not our vocabulary. **A rule you can't run
   > is a preference.** If the next shared term forces the word back into
   > `core/`, the honest move is to amend this rule again, not to smuggle it.

## 5. package ownership modes

ORCS supports the same code in three ownership modes. Paths are a runtime
contract, not a checkout assumption.

| mode | code | data and dependencies |
|---|---|---|
| vendored | `<host>/dependencies/orcs` | the host's `data/` and `dependencies/` |
| standalone checkout | ORCS repository | that repository's `data/` and `dependencies/` |
| non-editable install | site-packages | `$XDG_DATA_HOME/orcs/{data,dependencies}` |

`ORCS_ROOT`, `ORCS_DATA_ROOT`, and `ORCS_DEPS_ROOT` override those defaults.
Missing optional data never makes `import orcs` fail: each task row either
registers or contributes one entry to the top-level `orcs.SKIP_REASON` mapping.
Consumers should inspect that mapping explicitly; normal optional omissions do
not produce import-time warnings.

## 6. task slots

| task | what | sources | command spaces | status |
|---|---|---|---|---|
| `uolm` | Uni-Object Loco-Manipulation | retargeted G1, reconstructed human-object motion | `robot`, `smpl` | ✅ native and seed-backed SMPL recipes train |
| `perloco` | terrain from a height scan | OmniRetarget, GRAIL | `robot`, `smpl` (GRAIL) | ✅ native and seed-backed SMPL recipes train |
| `dodge` | whole-body ball evasion | generated nominal stand | `robot` | ✅ adapter task registers and trains |
| student distillation | oracle → deployable | — | — | ⬜ the next build |

## 7. adding a task

1. `src/orcs/tasks/<name>/__init__.py` builds a `_TASKS` table and calls
   `orcs.core.registry.register_all(_TASKS)` — per-row, so one unstaged dataset
   costs one task and `SKIP_REASON[task_id]` says why. A checkout without data
   must still import.
2. Task id is `Orcs-<Name>-[<Source>-]<Agent>[-<CommandSpace>]`. The agent is
   **always** an explicit token, so no row's identity depends on knowing the
   default. The SOURCE slot appears only when provenance changes code — reader,
   file format, joint order, conventions (perloco's `OmRe`/`Grail` do; its curb
   vs stair terrains do not, so those are a roster line, not a task).
3. Add one `import orcs.tasks.<name>` line to `src/orcs/__init__.py`.
4. Draw robots/objects from `orcs.assets`, paths from `orcs.core.paths`. If you
   need a new robot, add `orcs/assets/<robot>.py` and re-export it.
5. Anything shared with a second task moves down a layer — **only once the second
   task exists.** No speculative abstraction.
6. Non-`.py` files the task needs at runtime (a roster, a schema) go in
   `[tool.setuptools.package-data]`, or they exist only in your checkout.
