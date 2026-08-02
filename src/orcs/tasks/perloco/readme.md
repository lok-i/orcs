# PerLoco — perceptive locomotion

Design and rationale: [docs/perceptive_locomotion.md](../../../docs/perceptive_locomotion.md).
This file is how to run it.

One line: **UOLM's adapter reads object kinematics; PerLoco's reads terrain
kinematics.** Same frozen SONIC base, same LoRA adapter, same multi-clip RSI
motion command (`orcs.core.mdp.MultiClipMotionCommand`). There is no object.

Status: **P2** — staging + curation work; the env and tasks land in P3.

## stage

Turn a (terrain, motion) dataset into the orcs-native layout. Everything
dataset-specific dies here, so the runtime never grows a per-dataset branch.

```bash
python scripts/stage_terrain_motions.py --source omni
# 145 clips over 145 tiles, 91030 frames @ 50 Hz -> data/terrain_motions/omni

# a subset, while iterating
python scripts/stage_terrain_motions.py --source omni \
    --families climb_00 climb_05 --levels 1.0
```

Output — **the path IS the tile↔clip pairing**, so there is no manifest to
desync:

```
data/terrain_motions/omni/
  <family>/level_<L>/tile.json              terrain geometry, tile-local
  <family>/level_<L>/sample<N>/motion.npz   orcs-native, IL order, 50 Hz
  <family>/level_<L>/sample<N>/metadata.json
```

`<family>` becomes a sub-terrain grid **column** and `<L>` a **row** — which is
exactly the `<dataset>/<motion>/<sampleN>` shape `orcs.core.data.scan` already
walks, so the runtime recovers the pairing from `Path.parent`.

| source | `--source` | what it reads | status |
|---|---|---|---|
| OmniRetarget `robot-terrain` | `omni` | `robot-terrain.zip` + `models/terrain/*/multi_boxes_z_scale_*.urdf`, in place (never extracted) | ✅ 145 clips |
| GRAIL curb / stairs | `grail` | `robot/*.pkl` + `objects/*.pkl` pose + `object_usd/` | ⬜ P5 |
| InstinctMJ | `instinct` | `metadata.yaml` | ⬜ P5 |

## visualize + curate

```bash
python scripts/view_terrain_motions.py --source omni   # -> http://localhost:8080
```

A deliverable, not a convenience: the uolm dataset ships per-sample MP4s,
OmniRetarget ships nothing, so this is how the training roster gets chosen —
and how the one thing staging cannot check itself gets checked: **does the
motion sit ON its terrain, or through it?**

The skeleton is drawn from `motion.npz`'s `body_pos_w`, the same array the
training loader slices, so a pairing or frame-convention bug shows up here
rather than as a mysterious tracking regression 10k iterations in.

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
