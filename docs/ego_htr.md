# EgoHTR motion preparation

Scene-aware G1 retargeting with an optional UMR bridge and a Viser review loop.

## Use

```bash
.venv/bin/python -m orcs.cli.ego_htr view --port 8085
```

Open `http://localhost:8085`. If another viewer occupies that port, use `8086`.

Start an agent chat with:

> Read `src/orcs/ego_htr/SKILL.md`. Review my EgoHTR frame comments and continue the retargeting workflow.

1. Select **Motion → 03** and **Version → Latest**.
2. Play, pause or step to the frame; write feedback and click **Save note**.
3. Ask the agent to revise the comments.
4. Click **Reload files → Latest**; compare with **Accepted**.
5. Tell the agent when to accept a result.

## Viewer

| Control | Use |
|---|---|
| Motion / Version | Clip, latest candidate, accepted reference and historical revisions |
| Frame / Clip time / Previous / Next | Exact pose review; source timestamp shown |
| Play / Speed / Loop | 0.1–2× playback; looping starts off |
| Scan / Collision / Overlay / Hidden | Inspect measured terrain and the convex proxy |
| Opacity / Wireframe / Show G1 / Root path | Inspect occluded contacts and motion path |
| Fit whole clip / Focus on G1 | Reset camera framing; drag/pan/scroll otherwise |
| Save note | Binds feedback to the displayed frame, version, motion hash and terrain |

- Leave the viewer running while the agent creates a revision; reload to see it.
- Only the selected clip's terrain stays resident.
- Connected viewers share playback/selection; each has its own camera.
- Remote machine: forward the chosen localhost port through SSH or the IDE.

## Included workflow

| Feature | Agent handles |
|---|---|
| Source selection | Inspect local/remote NPZ members; fetch selected arrays; choose short intervals |
| Terrain | Full-body crop, clustered scan, bounded whole/tiled CoACD proxies |
| Retarget | Exact SMPL-X source geometry, isolated UMR, saved configuration and caches |
| Corrections | Foot support, separation, native collision, root continuity, knee/elbow posture, preserved palms |
| Verification | All-frame geometric/timing/smoothness checks and comparisons |
| Optional exports | Scan/proxy/human videos, contact sheets and annotated before/after poses |
| History | Independent iterations, accepted-reference protection, note status and safe pruning |
| PerLoco export | Pin reviewed runs; copy convex files; resample to 50 Hz and generate native G1 references |

## Play and train

Export the accepted 03 once, then play the frozen base:

```bash
.venv/bin/python -m orcs.cli.ego_htr stage --clips 03
.venv/bin/play Orcs-PerLoco-EgoHtr-AdaptSonic --agent initial --viewer viser
```

- Policy viewer: `http://localhost:8080`. Use `--viewer native` for a desktop window.
- EgoHTR sets `sim.mujoco.nccdmax=128` for train and play; no environment prefix needed.
- Play fixes sliding friction to **1.0 on all foot colliders**. Hands/terrain retain
  their nominal settings; training retains its existing friction randomization.
- Train and play finish each crop with **one step of end padding** (20 ms).
  These references end in motion; this task does not require a final standing hold.
  The endpoint default is local to EgoHTR; other PerLoco tasks are unchanged.
- `initial` constructs the frozen SONIC base with an untrained adapter.
- The policy viewer runs physics; the review viewer replays the reference.
- Staged reference/proxy viewer: `orcs-view-terrain --source egohtr --port 8087`.

### 03 full training

Prepared recipe for the accepted 03 crop; W&B project:
[`lkrajan/orcs`](https://forge.coreweave.com/wandb/lkrajan/orcs).

```bash
WANDB_USERNAME=lkrajan \
  .venv/bin/train Orcs-PerLoco-EgoHtr-AdaptSonic \
  --agent.logger wandb \
  --agent.wandb-project orcs \
  --agent.experiment-name PerLoco/EgoHtr/03 \
  --env.scene.num-envs 4096 \
  --agent.max-iterations 10000 \
  --agent.save-interval 1000 \
  --env.commands.motion.init-phase-anneal-iterations 7000 \
  --env.terminations.exceeded-motion.params.epsilon-steps 1
```

| Setting | Rationale |
|---|---|
| 10,000 updates / 7,000 anneal | Current 03 experiment budget; final 3,000 updates start at frame zero |
| 4,096 environments | Completed the 03 full run on the 24 GiB GPU with CCD capacity 128 |
| CCD 128 / zero reset lift | Keep collision capacity sufficient without a vertical reference-reset offset |
| End padding: 1 step | Avoid the two-second frozen pose with moving velocity targets at this crop's end |
| `WANDB_USERNAME=lkrajan` | This checkout's RSL-RL writer passes it as the W&B entity |

- The anneal gradually shrinks the allowed reset window; it does not initially
  filter unstable intermediate poses.
- Grail, OmRe and EgoHTR share this annealing implementation. Their code default
  is 0 (disabled); the earlier OmRe commands explicitly selected 10,000 annealing
  updates for a 15,000-update run. This 03 experiment uses the shorter 10k / 7k schedule.
- Training retains the existing 0.4 m pelvis-Z / 0.8 rad orientation terminations,
  observations, actions, rewards and domain randomization.
- This command starts a fresh run. It does not change the project or schedule of
  an already-running process.
- Run names use the default timestamp. Checkpoints:
  `logs/rsl_rl/PerLoco/EgoHtr/03/<timestamp>/`.

### Multiple crops

One policy can consume a list of reviewed crops:

```bash
.venv/bin/python -m orcs.cli.ego_htr stage --clips 03 04 05 06
# Use the roster path printed by stage for both train and play:
ORCS_EGOHTR_ROSTER=/path/to/printed-roster.toml \
WANDB_USERNAME=lkrajan \
  .venv/bin/train Orcs-PerLoco-EgoHtr-AdaptSonic \
  --agent.wandb-project orcs --env.scene.num-envs 512
```

| Setting | Meaning |
|---|---|
| Packaged default | 03 only; no runtime code depends on its label |
| `stage --clips ...` | Viewer labels or full clip IDs; defaults to each accepted revision |
| `stage --run ID` | Explicit revision; the same selector is applied to every selected clip |
| `stage --out PATH` | Alternative generated dataset root |
| `ORCS_EGOHTR_ROOT` | Load that alternative root during train/play |
| `ORCS_EGOHTR_ROSTER` | Ordinary PerLoco TOML roster selecting staged crop families |

Select only 03/05 from an existing export snapshot:

```bash
ORCS_EGOHTR_ROOT=/path/to/staged-snapshot \
ORCS_EGOHTR_ROSTER=src/orcs/tasks/perloco/rosters/egohtr_03_05.toml \
  .venv/bin/play Orcs-PerLoco-EgoHtr-AdaptSonic \
  --agent trained --checkpoint-file /path/to/checkpoint.pt \
  --num-envs 2 --viewer viser
```

- The subset reuses the staged motions and terrain; it does not retarget or retrain.
- A checkpoint trained on all four clips remains a four-clip-trained checkpoint
  when evaluated on this subset. Validate tracking after changing a selection.

- Export snapshots motion/terrain hashes and copies the hull files. A repeated
  identical export reuses them; a changed reference needs a fresh `--out`.
- Training needs the staged dataset and ORCS dependencies; UMR, SMPL-X and review
  logs are unnecessary. Keep generated datasets local/untracked.
- One crop gives every independent simulation world the same terrain origin.
  Training resets sample reference phases; play starts at the beginning.
- Multiple crops share a terrain atlas. Each environment's motion is restricted
  to its assigned terrain. Robot worlds do not collide with one another.
- Geometry loads from files; no CoACD or procedural generation runs during training.
- The height scan reads the same convex surfaces used for collision. Dense scan
  meshes remain in the review viewer.

03 system checks on RTX 3090 (24 GiB), including real PPO updates:

| Worlds | PPO updates | Sampled whole-GPU peak | Seconds/update |
|---:|---:|---:|---:|
| 512 | 100 | 2.76 GiB | 1.07 |
| 4096 | 10 | 8.62 GiB | 3.11 |

- Peaks include about 1.5 GiB of pre-existing device usage; these are short runs.
- These short benchmarks used CCD capacity 64. The subsequent 03 full run
  completed at 4,096 worlds with capacity 128; its VRAM peak was not measured here.
- Checkpoint save/reload and Viser startup passed. The final 03 checkpoint
  completed 191/192 frame-zero evaluation episodes under the training domain.

### Collision settings

| Setting | EgoHTR default | Override |
|---|---|---|
| CCD workspace per world | `sim.mujoco.nccdmax=128`, train and play | `ORCS_NCCDMAX` takes precedence; training also accepts `--env.sim.mujoco.nccdmax` |
| Contact / constraint capacity | 256 / 768 | Ordinary simulation configuration |
| Play foot sliding friction | 1.0 on all 14 capsules | `play_foot_friction` startup event in `egohtr_env_cfg` |
| Training friction | Independent per-collider uniform 0.3–1.6 | Shared robustness configuration |

- The CCD bridge lives in `tasks/perloco/egohtr_sim.py` and recognizes only
  models compiled with the EgoHTR configuration. Shared simulator code is unchanged.
- `CCD overflow` drops convex collision candidates. Increasing `nconmax` alone
  does not raise the separate CCD capacity. Other tasks retain their existing
  64-row fallback unless configured explicitly.
- Uniform foot friction 1.0 passed the 03 evaluation; nominal 0.6 exposed a
  traction weakness despite being inside the training randomization range.
- The foot-friction event changes only sliding friction. Collision geometry,
  priorities, hands, terrain, and training randomization are unchanged.
- Keep the 5 ms physics timestep and zero reset height lift.

## Files and setup

- Code: `src/orcs/ego_htr/`; one CLI: `orcs.cli.ego_htr` (`orcs-egohtr` after installation).
- Local runs: `logs/ego_htr/<sequence>__<start>-<end>s/<iteration>/`.
- Each completed motion run: **`run.json`, `motion.npz`, `run.log`**.
- Source/robot/terrain/targets/correspondence are shared under `_cache/`.
- Comments: one `notes.jsonl` per clip. Optional `reports/` and `debug/` are generated on request.
- `egohtr` is an optional Python extra for viewer/report/download tools.
- UMR remains an external checkout and isolated interpreter, configured once by
  the agent. No mandatory UMR dependency or change to the training environment.
- Large assets and licensed body models stay local. A fresh checkout needs those
  inputs prepared before clips appear; the empty viewer supports **Reload files**.
- Full command/schema/feature ledger: [agent tools](../src/orcs/ego_htr/agent_tools.md).

## Current boundary

- EgoHTR, G1, stationary terrain and 30 Hz review constraints.
- Kinematic checks and playback do not establish controller stability.
- The memory figures above cover 03; the four-crop atlas needs its own PPO budget.
- 03's reviewed hunch/roughness is parked; 04/05/06 remain preserved references.
- `Orcs-PerLoco-EgoHtr-AdaptSonic` shares PerLoco's MDP. It is a trainable task,
  not a released trained checkpoint. Native contacts and 50 Hz interpolation
  need checking when admitting another crop; visual acceptance alone is insufficient.
