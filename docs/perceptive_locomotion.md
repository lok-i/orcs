# perceptive locomotion (PerLoco)

Status: **built through phase 5** (both sources land; training is next). This file is the
design record — how to RUN it is [tasks/perloco/readme.md](../src/orcs/tasks/perloco/readme.md).
Read [ethos.md](ethos.md) first.

**UOLM's adapter reads object kinematics. PerLoco's reads terrain kinematics.** Same frozen
SONIC base, same LoRA adapter, same 3-stream layout, same multi-clip RSI motion command.
There is no object. The new noun is a **tile**; the new verb is **pair a clip to it**.

```
Orcs-PerLoco-OmRe-AdaptSonic          frozen SONIC + LoRA, height-scan augmentation
Orcs-PerLoco-Grail-AdaptSonic         same, GRAIL curb                  ← run-1 task
Orcs-PerLoco-Grail-AdaptSonic-Smpl    ...encoder reads the HUMAN instead
Orcs-PerLoco-{OmRe,Grail}-TaRa        tabula-rasa floors, same envs
```

The SOURCE joined the id at phase 5: provenance changes code (reader, format, joint order,
conventions), so it earns a token. Terrain TYPE does not — curb and stair share a reader, so
they are a roster line.

---

## 1. sources

| | **OmniRetarget** `robot-terrain` | **GRAIL** curb + stairs | **InstinctMJ** |
|---|---|---|---|
| nature | in-house MoCap → interaction-preserving retarget | synth video → 4D-HOI → retarget → RL-validated | AMASS/OMOMO + prepared cases |
| terrain | **1–2 oriented boxes**, z-extruded, arbitrary yaw | procedural curb / stair, binary USDC | `scene_tsdf.obj` meshes |
| clips | **145** | curb 1769 · stair 12188 | bring your own |
| tiles | 29 families × 5 z-scales = **145** | curb 200 · stair 4952 | 1 per case |
| clips/tile | **1** | ~8.8 · ~2.5 | 1 |
| payload | `qpos (T,36)` + `fps` | joblib `{key: {dof (T,29), root_trans_offset (T,3), root_rot (T,4), pose_aa, smpl_joints}}` | `base_{pos,quat}_w` + `joint_pos` + `joint_names` |
| rate | 30 Hz | 25 Hz | varies |
| pairing | **filename** `climb_05_z_scale_1.0` | filename stem across subdirs | `metadata.yaml` `terrain_id ↔ motion_file` |
| frame | **terrain-local**, z=0 ground | recon world; terrain pose in `objects/*.pkl` | terrain-local post-prepare |
| status | ✅ complete | ✅ **curb (1769)** · ❌ stair `robot/` still empty | ✅ code, no data |
| role | **tier 1 — build on this** | tier 2 — plugin now, wired when bytes land | tier 3 — interop only |

**OmniRetarget, measured.** `qpos = [quat wxyz(4) | pos(3) | 29 joints, URDF/DFS order]`,
Drake convention (confirmed against the shipped `visualize.py`). Each
`models/terrain/<family>/multi_boxes_z_scale_<L>.urdf` welds 1–2 `box<N>.obj` to world; every
`.obj` is **8 verts / 12 faces — a z-extruded rectangle at arbitrary yaw**; `z_scale` touches
the mesh scale's z only; box 2 (when present) is the ground platform. Motion is terrain-local
(`climb_05` base xy ∈ [-0.27,1.03]×[-0.50,0.45] — on the box).

⇒ an OmniRetarget terrain **is** N≤2 MuJoCo `box` geoms with a yaw quat. Not an approximation
of one. That is the whole reason it is tier 1.

**GRAIL, as of 2026-08-02** — curb is complete and usable; stairs are not. Remaining work,
in order: (1) stair `robot/`+`objects/` still empty on the mirror; (2) terrain assets are
binary USDC and `pxr` is not installed → `pip install usd-core`, then USD→mesh→(boxes|hfield)
— a curb is plausibly box-shaped, so check before baking; (3) `robot/*.pkl` carries **no
`joint_names`** — the 29-dof order is an assumption, and the viser pass (phase 2) is what
validates it; (4) 4952 stair tiles need a sampled roster (already a kwarg, §3).
The terrain pose is *not* a blocker: `objects/*.pkl` `root_pos`/`root_quat` is the
motion↔terrain transform (`root_quat[0] ≈ (-0.707, 0, 0, ·)` — the y-up→z-up conversion),
and `meta/*.pkl` adds `scene_scale`.

Out of scope: `robot-object-terrain` (`scene_*` + chair) and GRAIL `slope`. Object+terrain is
a UOLM × PerLoco cross, not this task.

---

## 2. terrain representation

| repr | Omni | GRAIL | narrowphase | overhang | verdict |
|---|---|---|---|---|---|
| **native `box` geoms** | **exact** | ✗ curves | cheapest primitive pair | ✓ | ✅ **tier 1** |
| **`hfield`, 1 geom/tile** | exact (boxes are 2.5-D) | good | cheap, one geom | ✗ | ✅ **tier 2** |
| mesh + CoACD hulls | exact | exact | expensive, hull-fill trap | ✓ | ✗ **never in orcs** |
| raw mesh geom | — | — | **broken** — MuJoCo fills concave interiors | — | ✗ |

1. orcs already ate the hull lesson on objects (CLAUDE.md: container hulls ≈ **11×** slower
   narrowphase). A box is **convex and exact** — the trap does not apply.
2. mjlab attaches the terrain spec into the shared `MjSpec` **once** (`scene.py:264`); worlds
   share geometry and differ only by `env_origins`. Terrain cost is collision-time, not
   memory-time ⇒ the only lever is **geoms per tile**, and boxes give ≤2.

**Rule: what cannot be boxes bakes to an hfield offline. orcs never learns the word CoACD.**
Boxes and hfield are *fields of one `TileSpec`*, not two classes — one `TileTerrainCfg` emits
whichever are present. Tile geoms go in **group 0** (the height scan and mjlab's terrain-bounds
termination both key on it).

---

## 3. tile ↔ clip pairing

mjlab's grid gives us both axes for free, and OmniRetarget's second axis genuinely *is*
difficulty — so unlike InstinctMJ we do not have to launder a terrain index through the
`difficulty` float:

| mjlab axis | PerLoco meaning | Omni values |
|---|---|---|
| **col** `terrain_types` — one per `sub_terrains` entry | terrain **family** | `climb_00 … climb_28` (29) |
| **row** `terrain_levels` — difficulty, runtime-promotable | **z_scale** = obstacle height | 0.8 … 1.2 (5) |

29 × 5 = **145 tiles = 145 clips = one clip per tile.** The dataset was built for this grid.
Curriculum mode pins `num_cols = len(sub_terrains)`, so it is **one `TileTerrainCfg` per
family** and `difficulty → level index` inside `function()` — exactly the API's shape.
`TerrainEntity.update_env_origins` then gives the z_scale curriculum free: clear a 0.8 box, get
promoted toward 1.2, on an axis the data actually spans.

RSI needs no new code: `_resample_command` already writes `body_pos_w[t] + env_origins[env]`,
and `env_origins[env]` **is** the tile origin. Every env on tile *t* starts on tile *t*.

**Correction to the obvious analogy — and the one real trap here.** UOLM caches
`env → object_id` in `__init__` because `sim.world_to_variant` is fixed for the run.
`terrain_levels` is **not**: the curriculum mutates it every reset. So the clip mask must be
**read from the terrain entity at `_resample_command` time**, never cached. A cached mask
would silently keep promoted envs sampling their old row's clip — the motion would then be
retargeted for a box height the env no longer stands on, and the failure looks like a tracking
regression, not a bug. This is the single thing most likely to be gotten wrong.

With one clip per tile the mask degenerates to a lookup (`clip_id = tile_id`); the general
`(n_tiles, n_clips)` bool stays because GRAIL is ~2.5–8.8 clips/tile.

Sizing: `size=(8,8)` over the full grid is 232 m × 40 m with **≤290 box geoms total**.
`families=` / `levels=` are factory kwargs — a roster decision changes world extent, **not the
design**. Decide it from the phase-2 viewer.

---

## 4. the unified interface

**The runtime loads only orcs-native files. Every dataset quirk dies in the offline stage.**
That one rule is what keeps three sources from becoming three code paths.

```
   OFFLINE — scripts/stage_terrain_motions.py                RUNTIME
 ┌──────────┐                                          ┌──────────────────────┐
 │ Omni zip │─┐   TerrainMotionSource (Protocol)       │ TerrainMotionCommand │
 │ GRAIL pkl│─┼─►   .tiles() → TileSpec                │  = MultiClipMotionCmd│
 │ Instinct │─┘     .clips() → ClipSpec                │    + tile mask       │
 └──────────┘             │  resample→50Hz             └──────────────────────┘
                          │  joints → IL order         ┌──────────────────────┐
                          │  FK → body_*_w             │ TileTerrainCfg       │
                          ▼                            │  reads tile.json     │
        data/terrain_motions/<source>/<family>/<level>/ └──────────────────────┘
              tile.json  [+ hfield.npy]
              sample0/motion.npz    ← orcs-native schema, unchanged
```

```python
# tasks/perloco/terrain_spec.py — tile-local frame, metres, z=0 at ground
@dataclass(frozen=True)
class BoxSpec:    pos: Vec3; quat: Vec4; half: Vec3

@dataclass(frozen=True)
class HFieldSpec: heights: np.ndarray          # (H,W) f32, tile-local z
                  size: tuple[float, float]    # xy extent
                  base: float                  # solid thickness below min(z)

@dataclass(frozen=True)
class TileSpec:   family: str                  # → grid COLUMN
                  level: float                 # → grid ROW
                  boxes: tuple[BoxSpec, ...] = ()
                  hfield: HFieldSpec | None = None

@dataclass(frozen=True)
class ClipSpec:   qpos: np.ndarray             # (T, 7+29) quat wxyz | pos | joints
                  fps: float
                  joint_names: tuple[str, ...] # source order; staging permutes to IL
                  family: str; level: float    # ← the pairing, and nothing else

class TerrainMotionSource(Protocol):
    name: str
    def tiles(self) -> Iterable[TileSpec]: ...
    def clips(self) -> Iterable[ClipSpec]: ...
```

| source | tiles from | clips from |
|---|---|---|
| `OmniRetargetSource` | 8-vert `.obj` × URDF `<mesh scale>` → AABB in the yaw-aligned frame | `qpos` npz; family/level from the filename |
| `GrailSource` | USD → trimesh → ray-cast bake → `HFieldSpec` | `robot/*.pkl` |
| `InstinctYamlSource` | `metadata.yaml` `terrains[]` → hfield | `metadata.yaml` `motion_files[]` |

**The path IS the pairing** — `<family>/<level>/` → `(col, row)`. No manifest, no parser,
nothing to desync, and `motion_dirs()` is already depth-invariant so the existing scan walks it
unchanged. (`InstinctYamlSource` still ships, so an InstinctMJ-prepared dataset drops into the
same pipe.)

**Staging transform** — target is the *unchanged* orcs `motion.npz` (`joint_{pos,vel}` (T,29)
IsaacLab BFS order, `body_{pos,quat,lin_vel,ang_vel}_w` (T,37,·), `fps=50`):

| field | Omni | GRAIL | staging does |
|---|---|---|---|
| `joint_pos` | `qpos[:,7:]` URDF order | pkl 29 dof | name-keyed permute → IL |
| `joint_vel` | ✗ | ✗ | central diff, post-resample |
| `body_*_w` | ✗ | ✗ | **FK** |
| `fps` | 30 | 25 | lerp + slerp → 50 |

**Do not write an FK tool.** `mjlab/scripts/csv_to_npz.py` already is one — `Simulation`,
qpos write, kinematics step, read back all four body arrays, plus the lerp/slerp resampler and
finite-difference velocities. Staging is that file's `MotionLoader` with a new front-end.

---

## 5. the env — a diff, not a design

| group | `Orcs-Uolm-AdaptSonic` | `Orcs-PerLoco-AdaptSonic` |
|---|---|---|
| `policy` / `tokenizer` | mocke contract | **identical** |
| `augmentation` → LoRA | object state + object_id + root state + goal + motion cmd | **`height_scan` (187)** + root state + root-twist cmd |
| `critic` | + object ref future, reward_vec | − object, + `tile_id` one-hot, + clean scan, + `foot_height`, + reward_vec |

```python
RayCastSensorCfg(name="height_scan",
    frame=ObjRef(type="body", name=scan_frame, entity="robot"),  # kwarg, default "pelvis"
    pattern=GridPatternCfg(size=(1.6, 1.0), resolution=0.1),     # 17 × 11 = 187
    ray_alignment="yaw", max_distance=5.0, include_geom_groups=(0,))
```

`ray_alignment="yaw"` is **not** optional — a base-aligned scan under climbing pitch stops
being a height map. `scan_frame` is a kwarg: `pelvis` (SONIC anchor, so scan and tokenizer
share a frame) vs `torso_link` (higher vantage) gets settled by playing it. Raw 187 into the
adapter, no encoder — the honest floor, and it matches how UOLM feeds object kinematics.

Privilege ledger (ethos §3) stays honest: a height map is what an onboard lidar/depth *would*
produce, so `augmentation` remains "needs perception on hw", not "needs a simulator".

| | keep from uolm | drop | add |
|---|---|---|---|
| rewards | `root_pos/ori`, `body_pos/ori`, `body_lin/ang_vel`, `action_rate_l2`, `joint_pos_limits` | every `object_*`, `contact_consistency` | — |
| terminations | `time_out`, `illegal_contact`, `exceeded_motion` | `bad_object_*` | **`bad_anchor_pos/ori` ON** (no object drags the root here); loosened to 0.4 m for the vertical excursions climbing produces |
| events | `reset_default`, `policy_update_counter`, robustness | `virtual_object_force` | — |

Kill set: climbing legitimately puts hands and knees on the box → `ROOT_KILL_BODIES` (pelvis
only), not `STRICT_KILL_BODIES`.

`terrain_contact_sensor` needed ONE change, and it is worth recording because the old code was
silently plane-only: the secondary match was `mode="geom", pattern="terrain"`. A secondary
`ContactMatch` with no `entity` is a **literal MuJoCo name, not a regex** — the plane's geom
happens to be named `terrain`, but the generator renames every tile geom to `terrain_<n>`, so
the first grid build died with `unrecognized name 'terrain'`. BODY names agree across both
flavours (one body named `terrain` either way), so the secondary is now `mode="body"`.

**Tile frame — the correction.** An earlier draft of this doc said sub-terrain
`origin = (0,0,0)`. That is wrong and would have put half of every tile inside its neighbour:
mjlab hands `function()` a frame whose origin is the tile's **corner**
(`_get_sub_terrain_position` returns the corner; every stock terrain therefore builds around
`(size/2, size/2)`). Staged tiles are centred on their own origin, so they are placed at that
centre and it is returned as `origin` — which is then exactly the `env_origins` RSI adds, so
OmniRetarget's z-frame (ground at 0) still survives verbatim. Each tile also emits its own
`size × size × 0.5` floor box with its TOP at `z = 0`, so tiles abut seamlessly and there are
no scan-miss gaps.

### 5.1 frames + conventions — verified, not assumed

Every place a number crosses from OmniRetarget into MuJoCo. The right-hand column is how it
was checked, because "both are z-up" is the kind of claim that is true four times and wrong
once.

| quantity | OmniRetarget / Drake | MuJoCo / mjlab | how verified |
|---|---|---|---|
| quaternion | wxyz | wxyz | Drake `MultibodyPlant` qpos is `[quat wxyz(4) \| pos(3) \| joints]`, confirmed against the dataset's own `visualize.py` |
| up axis | +z | +z | — |
| box extents | recovered from the 8 `.obj` verts | `geom.size` = **half**-extents | `BoxSpec.half` stores halves; orthogonality of the recovered frame asserted at ~1e-6 |
| box base | z = 0 | tile floor top at z = 0 | tile floor box is `pos_z = -depth/2` |
| terrain ↔ motion share a frame | *undocumented by the dataset* | — | **empirical**: min ankle height is +0.002 m at z_scale 1.0 and rises monotonically with z_scale. Feet land on box tops |
| joint order | URDF order (read from the shipped file) | MJ / XML DFS order | permuted **by name**, never by position; `[il_names[i] for i in IL2MJ] == mj_names` asserted at FK construction |
| tile origin | tile-local, boxes at ±half | tile-local frame's origin is the **corner** | placed at `(size/2, size/2)`; the viewer's `mj_forward` reproduces staged `body_pos_w` to 0.000000 m |
| viser box | — | `add_box` takes **full** extents | viewer passes `2 × half` |

The joint-order row is the one that already cost a rebuild: all 145 clips were once FK'd on
scrambled joints, and it survived every shape/quat/foot-height check because IL and MJ slot 0
are both `left_hip_pitch`. What caught it was a bone-length audit against a known-good dataset
— **a limb cannot be longer than its own kinematics allow.** That check is now a name-based
assertion at staging time, which is cheaper and cannot be argued with.

---

## 6. layout

**PerLoco never imports `orcs.tasks.uolm`.** Ethos §4 ("`tasks` may not import another task")
is exactly the pressure that forces the shared half down a layer — and there are two real
consumers now, so §6.5's "only once the second task exists" is satisfied.

```
src/orcs/core/
├── paths.py · deps.py · _mjlab_compat.py          unchanged
├── data/
│   ├── scan.py          motion_dirs · matches_exclude · scan_grouped(root)
│   └── loader.py        ConcatMotionLoader — robot timeline + clip bounds;
│                        `_load_extra(sample_dir, npz, T)` hook for task channels
├── mdp/
│   ├── commands.py      MultiClipMotionCommand — RSI + rand, phase anneal,
│   │                    last-frame freeze, N-step future, masked clip sampling, ghost
│   ├── terminations.py  exceeded_motion_by_eps · bad_anchor_pos · bad_anchor_ori
│   └── events.py        PolicyUpdateCounter
├── obs.py               _T · _grp · proprio_terms · robot_root_state_terms
│                        · robot_root_twist_cmd_terms
├── rl.py                THE agent zoo — runner spine + adapt_sonic/tara/sidecar
│                        agent cfgs. A task picks one; it never declares PPO.
└── sensors.py           terrain_contact_sensor · {UOLM,LOCOMANIP,STRICT}_KILL_BODIES

src/orcs/tasks/perloco/
├── __init__.py          register both tasks, try/except FileNotFoundError
├── env_cfg.py           perloco_env_cfg(agent, source, families, levels, scan_frame, play)
├── observation_cfgs.py  augmentation + critic, over core.obs atoms
├── sensors.py           terrain_scan (height scan) · terrain contact
├── terrain.py           TileTerrainCfg(SubTerrainCfg) — tile.json → boxes [+ hfield]
│                        · staged_roster · terrain_generator_cfg
├── terrain_spec.py      Box/HField/Tile/ClipSpec · TerrainMotionSource
├── sources/{omni,grail,instinct}.py   pure readers, no sim/torch/mjlab
├── mdp/commands.py      TerrainMotionCommand(MultiClipMotionCommand)
└── readme.md            stage → visualize → curate → play → train

scripts/
├── stage_terrain_motions.py   --source omni|grail|instinct → data/terrain_motions/
└── view_terrain_motions.py    viser: tiles + clip playback + keep/drop curation
```

The test for "does it belong in core": **does it mention an object or a terrain?**

| stays in `uolm/` | moves to `core/` | new in `perloco/` |
|---|---|---|
| `ObjectMotionCommand` = core cmd **+ object** | `MultiClipMotionCommand` | `TerrainMotionCommand` = core cmd **+ tile mask** |
| `contact_schedule`, object rewards/terms/obs | loader + scan helpers | `TileTerrainCfg`, height-scan sensors |
| `bodywise_contact_cmd` (object contact) | `robot_root_{lin,ang}_vel_cmd`, `proprio_terms` | `height_scan` obs, `tile_id` one-hot |
| `robustness.py`, `smpl_data.py`, VOF | `_runner`, dist bands, kill-body sets | — |

**Ethos §4 amendment this forces.** Core is declared "functional only, no semantics"; `core/mdp`
and `core/obs` break that. The replacement, which is the rule actually worth enforcing:

> `core` is **robot-generic and task-blind**. It may know what a joint, a body, a clip and a
> reference are. It must not know what an *object* or a *terrain* is. It still may not import
> `orcs.assets` or `orcs.tasks`.

Grep-testable, which the old line never was.

Budget: ~450 LOC **moved** into core, ~500 new in `perloco/`, ~580 new offline (staging +
3 sources + viewer). Versus ~3200 if InstinctMJ's terrain stack were inherited.

---

## 7. phases

| # | deliverable | gate |
|---|---|---|
| **1** | core promotion; uolm refactored onto it; ethos §4 amended | `play Orcs-Uolm-AdaptSonic --agent initial` **bit-identical to today** — it is a pure move, and rolling it is the only way to know |
| **2** | `OmniRetargetSource` + `stage_terrain_motions.py` + `view_terrain_motions.py` + `perloco/readme.md` | 145 clips staged; viser shows each clip **on its box, not through it**; you curate the roster, it writes `exclude_motions` in orcs's existing grammar |
| **3** ✅ | `TileTerrainCfg`, `TerrainMotionCommand`, sensors, obs, rewards/terminations; agents promoted to `core.rl` | ✅ `play Orcs-PerLoco-AdaptSonic --num-envs 10 --agent initial` — 5×29 grid, 483 geoms, `augmentation` 187+15, every env's clip verified to belong to its own tile |
| **4** | `train --num_envs 4096` + z_scale curriculum | beats `-TaRa` on reward vs `_runtime`; promotion fires |
| **5** ✅ | `GrailSource` (curb, 63 clips / 8 tiles) + the `-Smpl` command space | ✅ zero-shot tracking reward 6.039 (robot) / 4.660 (smpl) vs OmRe's 4.718; skeleton renders on its curb |

**Phase 2's viewer is a deliverable, not a convenience** — the UOLM dataset ships a per-sample
`retargeted_motion.mp4`; OmniRetarget ships nothing, so this is *how the roster gets chosen*.
It renders the **real model**: the G1 entity posed by writing `motion.npz` into `qpos` +
`mj_forward`, on terrain built by the env's own `TileTerrainCfg`. The first version drew a
14-node stick figure instead, and its limbs stretched — which turned out to be a real
joint-order bug in staging, but took a bone-length audit against a known-good dataset to tell
apart from "the skeleton is a bad drawing". **A viewer built from an abstraction of the data
can only disagree with the data decoratively.** Cross-check that the current one cannot: FK
from its `qpos` reproduces the staged `body_pos_w` to 0.000000 m.

---

## 8. settled

| | |
|---|---|
| name | `PerLoco` — `Orcs-PerLoco-<Source>-<Agent>[-Smpl]` |
| terrain | boxes wherever possible, hfield for the rest, CoACD never |
| data | OmniRetarget `robot-terrain` (145 clips); GRAIL curb (8 tiles rostered of 1769 takes); stairs when they stage; no slope, no `robot-object-terrain` |
| sharing | via `core/`, never task→task; costs an ethos §4 amendment |
| tile mask | **read at reset from `terrain_levels/types`, never cached** — the curriculum mutates it |
| anchor tubes | `bad_anchor_pos/ori` ON, **width per source** — see §9 |
| scan | raw 187, no encoder; frame a kwarg, default `pelvis` |
| roster size | a roster line; chosen from the phase-2 viewer; affects world extent only |

---

## 9. run-1 regime (2026-08-02)

The first training run asks ONE question: **can a frozen WBC be adapted to terrain
constraints at all?** That is a behavior question, so nothing that makes it a sim2real
question is on.

| knob | setting | rationale |
|---|---|---|
| rewards | **8 terms** | the 6 tracking terms are byte-identical to SONIC's `tracking/base`, plus `action_rate_l2` (−0.1) and `joint_pos_limits` (−1.0) |
| DR + obs noise | **off**, one switch (`env_cfg.SIM2REAL`) | see below |
| anchor tubes | GRAIL 0.2 m / 0.3 rad · OmRe 0.4 / 0.8 | see below |

**What NVIDIA runs, for calibration.** GRAIL ships no reward code — `grail/` is a data
pipeline and all its RL is vendored SONIC (`imports/SONIC`). Its terrain entry point,
`config/exp/manager/universal_token/scene/terrain_tracking.yaml`, picks the stock
`rewards: tracking/base_5point_local_feet_acc`. So the four terms we do not have are:

| term | weight / std | why it is not in yet |
|---|---|---|
| `tracking_vr_5point_local` | **+2.0 / 0.1** | their heaviest term; anchor-local 5-point over torso+0.5z, 2 wrists, 2 ankles. The one worth adding if run 1 underfits limb placement |
| `undesired_contacts` | −0.1, 1 N | we use a hard root-only `illegal_contact` termination instead |
| `anti_shake_ang_vel` | −5e-3, wrists+head, deadzone 1.5 rad/s | regularizer |
| `feet_acc` (`joint_acc_l2` on `.*ankle.*`) | −2.5e-7 | regularizer |

Their `joint_limit` is −10.0 against our −1.0. Their terminations are much tighter
(`anchor_pos` 0.15, `anchor_ori` 0.2, plus `ee_body_pos_adaptive` and `foot_pos_xyz`).

**Their height map is ours**: root origin, heading-aligned (yaw only), downward,
`size 1.5` @ `res 0.15` → 11×11 = 121. Ours is 1.6×1.0 @ 0.1 → 187 at the pelvis. Same
convention, finer grid. The real divergence is the CONSUMER — theirs feeds the tokenizer
encoder through a conv2d projector, ours feeds the adapter. That difference IS the orcs
thesis, so it is deliberate.

**Why no domain randomization.** PerLoco never had one: `events` is reset + the update
counter, and uolm's `apply_robustness` is object-specific. Obs noise was likewise inert —
the groups set `enable_corruption=True` while mocke's `policy_obs_terms()` defaults
`noisy=False`, so no term ever carried a `.noise`. `SIM2REAL = False` makes that state
legible and gives it one name; flipping it turns on proprio noise. **NVIDIA agrees**: their
terrain config nulls all five event terms and sets `enable_corruption: false`. A push/mass
domain for perloco does not exist yet and belongs behind the same switch when it lands.

**Why the anchor tubes differ by source.** `bad_anchor_pos` is pelvis-**z** drift, not a
3-D tube. Frozen base, 250 steps × 32 envs:

| source | \|dz\| p50 | p90 | ori p50 | p90 |
|---|---|---|---|---|
| GRAIL curb | 0.018 m | 0.281 | 0.101 rad | 0.433 |
| OmniRetarget climb | 0.030 m | **0.562** | 0.149 rad | **1.491** |

0.2 / 0.3 sits at roughly GRAIL's p90 — it kills divergence. On OmRe it would kill the
**behaviour**: climbing means large vertical excursions, which is what the original 0.4
comment said and what the p90 confirms. So the thresholds are `_core` parameters with
per-source defaults, not a global constant.
