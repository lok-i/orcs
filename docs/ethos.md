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
| `augmentation` | task/object kinematics — the adapter's conditioning | yes (needs perception on hw) | ✅ |
| `critic` | full state | yes (never deployed) | ✅ asymmetric actor-critic |
| student | proprio + perception, distilled | — | ⬜ roadmap |

So today's privilege is **asymmetric actor-critic + ground-truth object state**.
The distillation stage that closes the sensing gap is the next build, not a
shipped feature.

**Adapt, don't retrain.** A task straps a zero-init LoRA adapter onto a frozen
competent base (SONIC WBC) — at construction it reproduces the base bit-exact,
and PPO only moves the adapter. Verify this claim, don't trust it:
`play <task> --agent initial` rolls the freshly-built agent with no checkpoint.

## 4. layer contract

```
src/orcs/
├── __init__.py   registry point: import each task, wire the mjlab shim
├── core/         robot-generic, task-blind infra
│   ├── paths · deps · _mjlab_compat      no semantics at all
│   ├── data/     scan (clip discovery) · loader (concatenated timeline)
│   ├── mdp/      commands (MultiClipMotionCommand) · observations
│   │             · terminations · events
│   ├── obs.py    group plumbing + robot-only term bundles
│   ├── rl.py     PPO runner spine + actor builders
│   └── sensors.py  robot<->terrain contact + kill-body vocabulary
├── assets/       robots + objects as mjlab entity cfgs. g1.py, objects.py
└── tasks/        one self-registering package per task
    └── uolm/     env_cfg · rl_cfg · robustness · smpl_data · mdp/
```

Import rules — enforced by review, not tooling:

| layer | may import | must never import |
|---|---|---|
| `core` | stdlib, mjlab, mocke | `orcs.assets`, `orcs.tasks` |
| `assets` | `orcs.core` | `orcs.tasks` |
| `tasks` | `orcs.core`, `orcs.assets`, sibling-free | another task |
| `__init__` | everything (the only wiring point) | — |

Two rules earn their keep:

1. **No `__file__` depth math.** Every path comes from `orcs.core.paths`. Moving
   a module can never silently orphan a dataset.
2. **`core` is robot-generic and task-blind.** It may know what a joint, a body,
   a clip and a reference are. It must **not** know what an *object* or a
   *terrain* is — the moment a name in `core` mentions one, it belongs to the
   task that has one. The mjlab compat shim needs a task's command cfg, so it
   takes it as an argument — `apply(multi_clip_cfgs=...)`, wired in
   `orcs/__init__.py`. Core never reaches upward.

   > **Amended 2026-08-02.** Rule 2 used to read "zero semantics". That held only
   > while core carried no terms, which held only while there was one task. The
   > second task (perloco) needs the same proprio bundle, the same runner spine,
   > the same RSI/annealing/freeze machinery — and §4 forbids it importing uolm to
   > get them, which is the pressure working as intended. So core now carries
   > robot semantics, and the line moved to where it can actually be checked:
   > grep `core/` for "object" or "terrain".

## 5. task slots

| task | what | command spaces | status |
|---|---|---|---|
| `uolm` | Uni-Object Loco-Manipulation | `robot` (retargeted G1), `smpl` (human) | ✅ robot trains; smpl rollout-only |
| perceptive locomotion | terrain from onboard sensing | — | ⬜ |
| student distillation | oracle → deployable | — | ⬜ |

## 6. adding a task

1. `src/orcs/tasks/<name>/__init__.py` calls `register_mjlab_task(...)`, wrapped
   in `try/except FileNotFoundError` — a checkout without data must still import.
2. Task id is `Orcs-<Name>-<Agent>[-<CommandSpace>]` — the agent is always an
   explicit token, so no row's identity depends on knowing the default.
3. Add one `import orcs.tasks.<name>` line to `src/orcs/__init__.py`.
4. Draw robots/objects from `orcs.assets`, paths from `orcs.core.paths`. If you
   need a new robot, add `orcs/assets/<robot>.py` and re-export it.
5. Anything shared with a second task moves down a layer — **only once the second
   task exists.** No speculative abstraction.
