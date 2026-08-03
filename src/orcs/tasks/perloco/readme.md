# PerLoco — perceptive locomotion

Design and rationale: [docs/perceptive_locomotion.md](../../../docs/perceptive_locomotion.md).
This file is how to run it.

One line: **UOLM's adapter reads object kinematics; PerLoco's reads terrain
kinematics.** Same frozen SONIC base, same LoRA adapter, same multi-clip RSI
motion command (`orcs.core.mdp.MultiClipMotionCommand`). There is no object.

Status: **five tasks register and roll.** Data first —
[`scripts/setup/perceptive_locomotion.sh`](../../../../scripts/setup/perceptive_locomotion.sh)
does the whole fetch+stage in one command.

```bash
play  Orcs-PerLoco-Grail-AdaptSonic --num-envs 10 --agent initial   # the gate
train Orcs-PerLoco-Grail-AdaptSonic --num_envs 4096
```

Ids read `Orcs-PerLoco-<Source>-<Agent>[-<CommandSpace>]`. The SOURCE is in the
id because provenance changes code (reader, format, joint order); the TERRAIN
TYPE is not — curb and stair are the same reader, so they are a roster line.

| task | grid | agent / reference |
|---|---|---|
| `Orcs-PerLoco-OmRe-AdaptSonic` | 29 climb families x 5 z-scale levels | frozen SONIC + LoRA on the height scan |
| `Orcs-PerLoco-OmRe-TaRa` | " | from-scratch MLP — the no-frozen-base floor |
| `Orcs-PerLoco-Grail-AdaptSonic` | 8 curb families x 1 level | frozen SONIC + LoRA |
| `Orcs-PerLoco-Grail-AdaptSonic-Smpl` | " | ...encoder reads the SMPL-X **human**, not the retarget |
| `Orcs-PerLoco-Grail-TaRa` | " | from-scratch floor |

Each env stands on one tile and may only sample the clips staged against it — read live from
`terrain_{types,levels}` at every reset, never cached, because a level curriculum moves that
map underneath the command.

**The `-Smpl` row is a command SPACE, not a task**: same terrain, rewards, RSI,
adapter and critic — only the frozen encoder's input changes (and the ported
ckpt with it), which makes the pair a controlled read on what retargeting costs.
GRAIL is the only source that can do this because it ships both halves of every
take. (uolm's `-Smpl` is rollout-only for the opposite reason.)

### run-1 regime

| knob | setting | why |
|---|---|---|
| rewards | 8 terms — 6 tracking + `action_rate_l2` + `joint_pos_limits` | the 6 are byte-identical to SONIC's `tracking/base`. The 4 in NVIDIA's `base_5point_local_feet_acc` (`tracking_vr_5point_local` w=2.0 being the heavy one) are deliberately **not** in yet |
| domain randomization | **off** (`env_cfg.SIM2REAL = False`) | run 1 asks whether a frozen WBC adapts to terrain at all — a behavior question. GRAIL's own terrain release nulls all five event terms too |
| obs noise | **off**, same switch | it was already inert (the group flag was True while no term carried a `.noise`); now flag and terms agree |
| anchor tubes | GRAIL 0.2 m / 0.3 rad · OmRe 0.4 / 0.8 | per source. `bad_anchor_pos` is pelvis-**z** drift, not a 3-D tube. Frozen-base \|dz\| p50: GRAIL 0.018 m, OmRe 0.030 — but OmRe's p90 is 0.562 m because **climbing means vertical excursions**, so tightening it would terminate the behaviour |

## stage

Turn a (terrain, motion) dataset into the orcs-native layout. Everything
dataset-specific dies here, so the runtime never grows a per-dataset branch.

```bash
orcs-stage-terrain --source omni          # or: python scripts/stage_terrain_motions.py
# 145 clips over 145 tiles, 91030 frames @ 50 Hz -> data/terrain_motions/omni

orcs-stage-terrain --source grail --smpl --families curb_000 curb_017 ...
# --smpl also writes smpl_motion.npz (SMPL-X forward pass; needs $ORCS_SMPLX_DIR)

# a subset, while iterating
orcs-stage-terrain --source omni --families climb_00 climb_05 --levels 1.0
```

Output — **the path IS the tile↔clip pairing**, so there is no manifest to
desync:

```
data/terrain_motions/omni/
  <family>/level_<L>/tile.json              terrain geometry, tile-local
  <family>/level_<L>/sample<N>/motion.npz   orcs-native, IL order, 50 Hz
  <family>/level_<L>/sample<N>/smpl_motion.npz   --smpl only
  <family>/level_<L>/sample<N>/metadata.json
```

`<family>` becomes a sub-terrain grid **column** and `<L>` a **row** — which is
exactly the `<dataset>/<motion>/<sampleN>` shape `orcs.core.data.scan` already
walks, so the runtime recovers the pairing from `Path.parent`.

| source | `--source` | what it reads | status |
|---|---|---|---|
| OmniRetarget `robot-terrain` | `omni` | `robot-terrain.zip` + `models/terrain/*/multi_boxes_z_scale_*.urdf`, in place (never extracted) | ✅ 145 clips / 145 tiles |
| GRAIL curb | `grail` | `robot/*.pkl` + `object_usd/*.usd` geometry + `recon/*.pkl` (SMPL-X, `--smpl`) | ✅ 63 clips / 8 tiles |
| GRAIL stair1 / stair2 | `grail` | same reader | ⬜ a roster line when they stage |
| InstinctMJ | `instinct` | `metadata.yaml` | ⬜ no reader yet |

## visualize + curate

```bash
orcs-view-terrain --source omni   # -> http://localhost:8080
```

A deliverable, not a convenience: the uolm dataset ships per-sample MP4s,
OmniRetarget ships nothing, so this is how the training roster gets chosen —
and how the one thing staging cannot check itself gets checked: **does the
motion sit ON its terrain, or through it?**

It renders the REAL MODEL, not a depiction of one — the G1 entity posed by
writing `motion.npz` into `qpos` + `mj_forward`, on terrain built by the env's
own `TileTerrainCfg`, at the same tile offset RSI uses. So a pairing, frame or
joint-order bug shows up here as a broken robot rather than as a mysterious
tracking regression 10k iterations in, and a box placed wrong here is a box
placed wrong in training.

*Keep* / *Drop* per tile, then **Write exclude file** → `exclude.txt` in the
`exclude_motions` grammar `orcs.core.data.scan` already understands. The roster
is a config line, never a deleted file.

## known data property — low z-scale sinks

Measured across all 145 staged clips, lowest ankle-link height per level:

| z_scale | clips | min (m) | mean (m) | below −0.02 m |
|---|---:|---:|---:|---:|
| 0.8 | 29 | −0.059 | 0.033 | 6 |
| 0.9 | 29 | −0.033 | 0.039 | 3 |
| 1.0 | 29 | +0.002 | 0.052 | 0 |
| 1.1 | 29 | +0.016 | 0.067 | 0 |
| 1.2 | 29 | +0.027 | 0.087 | 0 |

Monotonic in `z_scale`, so this is **systematic, not per-clip retargeting
noise** — foot clearance rises with obstacle height at roughly 40% of the box's
own growth rate. On a planted foot the ankle link sits ~3 cm above ground, so
−0.059 m is roughly 9 cm of sole penetration on the worst clip.

It is left in the data on purpose. The options are a curation call, not a
pipeline fix:

1. **drop levels 0.8/0.9** for the affected families (viewer → `exclude.txt`)
2. **accept it** — RSI writes the pose and the solver pushes the foot out on
   the first step; ≤6 cm is inside what contact resolution absorbs
3. **per-clip z lift** — rejected unless someone wants it: shifting the motion
   to hide the penetration invents data and breaks the box-top contact the
   clip was retargeted for

Do NOT "fix" it with a global z-offset. The penetration is at the GROUND end,
the contact that matters is at the BOX-TOP end, and one offset cannot serve
both.
