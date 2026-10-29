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

Normal ORCS data setup installs the shipped **03 + 05** motion/terrain pairs.
It copies verified, processed files from `retargeted_motions` and `assets`; no raw
EgoHTR download, UMR, resampling, FK or CoACD step is required.

`deps.lock` pins the merged motion and terrain revisions. The packaged
`release.json` pins the merged [HF model bundle](https://huggingface.co/lkrajan/orcs/tree/92fa052800ab5bf911b0057132f5dd47b32d7413)
by commit and verifies checkpoint size and SHA-256 on download.

```bash
bash scripts/setup/sync_deps.sh --no-smpl
bash scripts/setup/sync_data.sh --no-smpl
.venv/bin/play Orcs-PerLoco-EgoHtr-AdaptSonic --agent release --num-envs 2 --viewer viser
```

- Already installed? Run only `play`; there is no separate EgoHTR setup mode.
- `sync_data.sh inhouse --no-smpl` also installs the shipped pair, without fetching
  the independent OmRe/GRAIL datasets.
- Policy viewer: `http://localhost:8080`; use `--viewer native` for a desktop window.
- `initial` constructs the frozen SONIC base with an untrained adapter.
- Policy playback runs physics; the review viewer replays the reference.
- `release` downloads and verifies the joint 03/05 checkpoint; W&B access is unnecessary.

| Shipped crop | Reviewed revision | Reference | Terrain |
|---|---|---|---|
| 03 rocky ascent, `rocky-path_s7_bzb_01__40-50s` | `0009_natural-step-v3` | 499 frames at 50 Hz | 360 convex pieces |
| 05 window crossing, `cliff-jump_s4_bzb_01__28-38s` | `0006_native-manifold` | 499 frames at 50 Hz | 192 convex pieces |

| Owner | Files |
|---|---|
| `retargeted_motions` | `data/unitree_g1/egohtr/<clip>/sample1/{motion.npz,metadata.json}` |
| `assets` | `terrain/egohtr/<clip>/{tile.json,meshes/*.obj,provenance.json}` |
| Local runtime | `data/terrain_motions/egohtr/<clip>/level_0.00/` |
| Selection | Packaged `tasks/perloco/rosters/egohtr.toml`, default 03 + 05 |

- Motion metadata pins the matching terrain descriptor; descriptors pin mesh hashes.
- Installer verifies hashes and copies the exact training inputs. An identical
  repeat is reused; another export or modified local data is never overwritten.
  Source notices can refresh automatically when every other field and payload
  is identical. Move a different `data/terrain_motions/egohtr` export aside first.
- Runtime files are self-contained. Original review logs and raw scans are unnecessary.
- Each sample's metadata preserves the EgoHTR citation, CC-BY-4.0 license link,
  changes and original access-conditions summary through lean setup.
- The two terrains form one atlas; each world receives the reference belonging
  to its terrain. Independent robot worlds do not collide with each other.
- Height scans raycast the same convex surfaces used for collisions.
- EgoHTR's [palette](../src/orcs/ego_htr/palette.py): warm gray terrain, charcoal
  ground, pale blue reference (12% opacity), cyan height-scan points.

### Joint 03/05 training

```bash
WANDB_USERNAME=lkrajan \
  .venv/bin/train Orcs-PerLoco-EgoHtr-AdaptSonic \
  --agent.logger wandb \
  --agent.wandb-project orcs \
  --agent.experiment-name PerLoco/EgoHtr/03-05 \
  --env.scene.num-envs 4096 \
  --agent.max-iterations 10000 \
  --agent.save-interval 1000 \
  --env.commands.motion.init-phase-anneal-iterations 7000 \
  --env.terminations.exceeded-motion.params.epsilon-steps 1
```

- 10k updates, 7k phase annealing; final 3k updates start from frame zero.
- Run names use the default timestamp. W&B entity is `lkrajan`, project `orcs`.
- Existing PerLoco observations, rewards, actions and training randomization apply.
- Both clips end in motion: EgoHTR uses one step (20 ms) of endpoint padding.

### Multiple crops

Author a new reference with the review workflow, then export into a separate root:

```bash
.venv/bin/python -m orcs.cli.ego_htr stage --clips 03 04 05 06 --out /path/to/custom-export
# Use the roster path printed by stage for both train and play:
ORCS_EGOHTR_ROOT=/path/to/custom-export \
ORCS_EGOHTR_ROSTER=/path/to/printed-roster.toml \
  .venv/bin/play Orcs-PerLoco-EgoHtr-AdaptSonic --agent initial --viewer viser
```

| Option | Meaning |
|---|---|
| `stage --clips ...` | Viewer labels or full crop IDs, defaulting to accepted revisions |
| `stage --run ID` | Pin a specific revision |
| `stage --out PATH` | Separate immutable export root |
| `ORCS_EGOHTR_ROOT` | Custom runtime root; omitted for shipped references |
| `ORCS_EGOHTR_ROSTER` | Custom selection; omitted for shipped 03/05 |

Existing scripts using `egohtr_03_05.toml` remain compatible; `egohtr.toml` is the
canonical default. Changing a custom crop requires contact and policy validation.

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
- 03's reviewed hunch/roughness is parked; 04/06 remain local review references.
- `Orcs-PerLoco-EgoHtr-AdaptSonic` shares PerLoco's MDP and includes a released
  joint 03/05 policy. Native contacts and 50 Hz interpolation need checking when
  admitting another crop; visual acceptance alone is insufficient.
