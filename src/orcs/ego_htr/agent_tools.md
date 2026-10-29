# EgoHTR agent tools

Read [SKILL.md](SKILL.md) first. Commands below use
`.venv/bin/python -m orcs.cli.ego_htr`; the table abbreviates it as `ego`.
All non-view commands return JSON; logs from external programs go to `run.log`.
Errors exit nonzero. `--workspace PATH` precedes the command.

## Commands

| Intent | Command |
|---|---|
| Dependency/patch/clip inventory | `ego doctor` |
| Configure local paths once | `ego configure --data-root PATH --umr-root PATH --umr-python PATH --smplx-model FILE` |
| Apply the checked integration patch | `ego doctor --apply-patch` |
| Discover clips / inspect a revision list | `ego inspect` / `ego inspect --clip 03` |
| Inspect local NPZ headers, shapes and dtypes | `ego inspect --sequence rocky-path_s7_bzb_01` |
| Bounded remote archive inventory | `ego remote --sequence SEQUENCE --revision REVISION` |
| Fetch specific array members | Add `--extract scene_vertices.npy scene_faces.npy egomc_footlock_LeftFoot.npy egomc_footlock_RightFoot.npy` |
| Read only member headers | Add `--prefix MEMBER.npy` (writes a `.prefix`, not a usable array) |
| Prepare source/proxy/correspondence | `ego prepare --recipe FILE --stage all` |
| Start from a reviewed recipe | `ego prepare --preset 03 --stage source` |
| Continue an explicit preparation revision | `ego prepare --clip LABEL --run ID --stage collision` (also `build`, `train`, `all`) |
| Solve a new complete candidate | `ego retarget --clip 03 --parent accepted --label candidate --overrides FILE` |
| Short solver probe | Add `--max-frames 8` (check frame count before presenting as a full clip) |
| Audit / compare all frames | `ego audit --clip 03 --run ID --compare accepted` |
| Detailed array dump | Add `--debug` to `audit` |
| Optional images/gallery | `ego report --clip 03 --run ID --compare accepted` |
| Scan/proxy and human videos | Add `--videos --human` to `report` |
| Launch Viser | `ego view --port 8085` (optional `--clip 03 --time 4.5`) |
| Export reviewed clips for PerLoco | `ego stage --clips 03 --run accepted` (accepts a list; returns a roster path) |
| Review pending comments | `ego notes --clip 03 --pending` |
| Record a comment without the browser | `ego comment --clip 03 --run ID --frame 137 --text 'left heel'` |
| Mark work awaiting review | `ego notes --clip 03 --resolve NOTE_ID --status addressed --run ID --text 'change/evidence'` |
| Close an accepted comment | Same command with `--status resolved`; use `dismissed` only with a recorded reason |
| Record user acceptance | `ego accept --clip 03 --run ID` |
| Preview selected cleanup | `ego prune --clip 03 --runs ID ID --dry-run` |
| Apply requested cleanup | Replace `--dry-run` with `--apply` |
| Import this research prototype once | `ego import-legacy --root logs/ego_htr/retarget_v1` |
| Optional synthetic memory experiment | `ego memory-probe --worlds 128 --terrain hfield --ccd 50 --nccd 64` |

Use `--help` on a command for its complete flags. Internal `_prepare`, `_targets`
and `_surface-audit` workers run through the bridge; do not ask the human to run them.

## Optional environment

| Component | Contract |
|---|---|
| ORCS environment | Viewer: `viser`, `mjviser`, `trimesh`; reports: Pillow, ffmpeg/ffprobe, EGL; remote: requests, huggingface-hub. Optional `egohtr` extra declares Python packages. |
| External UMR | Checkout `d1cdaedaf67a827198da71c5fa831f48b2e51f91` of `hanyang9/UMR`; `patches/umr_egohtr.patch` must apply cleanly or already be applied. |
| Isolated interpreter | UMR requirements plus `smplx`, `coacd`, `trimesh`, `embreex`, `rtree`, SciPy, Torch and MuJoCo. Migration used Python 3.11, NumPy 1.26.4, MuJoCo 3.3.7. |
| Body model | Explicit licensed SMPL-X model file matching the source gender; tested clips use `SMPLX_MALE.npz`. |
| Dataset | Local optimized motion and matching scene arrays; HF credentials are optional and read in memory for selective downloads. |

The bridge does not install UMR, vendor its environment or change ORCS's simulator
pins. It records the interpreter, commit and patch hash. A temporary package-only
import path makes ORCS workers available to the isolated interpreter and its
spawned CoACD workers without mixing site-packages. Ordinary import/help and
playback of cached runs do not need UMR.

## Workspace contract

```text
logs/ego_htr/
  workspace.json                   # local dependency paths
  _cache/<kind>/<content-key>/      # source, models, proxy, targets, correspondence
  <sequence>__<start>-<end>s/
    clip.json                      # label, interval, latest + accepted pointers
    notes.jsonl                    # append-only comments/dispositions/acceptance
    0000_accepted/
      run.json                     # recipe, assets, parent, provenance, checks, hash
      motion.npz                   # one normalized trajectory
      run.log                      # consolidated external-process output
    0001_<label>/...
```

- `prepared` revisions contain inputs and two files; no trajectory or Latest
  pointer. Use the returned preparation ID explicitly as `retarget --parent`.
- Imported references use `imported`; successful new solves use `ready`.
  `failed` / `failed_checks` keep evidence without replacing Latest.
- Latest is the most recent successful allocation, independent of completion
  order. Accepted changes only after review. Neither is rewritten by a solver.
- IDs increase monotonically, including failed and pruned runs. Pruning leaves
  a `run.json` tombstone and notes; it never deletes shared caches.
- Cache identities include inputs/configuration; models copy only referenced
  meshes. No live scan/model/correspondence dependency on historical log folders.
- Move the workspace as a unit. Internal assets use relative references;
  `configure` updates external dataset, body-model, UMR and interpreter paths.
- Completed inputs/results are immutable; `run.log` may append subsequent audit
  subprocess output. Reports/debug are optional, regenerable directories.

## Coordinates and comments

| Field | Meaning |
|---|---|
| `source.start_frame`, `end_frame` | Original 30 Hz indices; half-open interval |
| `motion.frame_ids` | Indices relative to the prepared source interval |
| `motion.source_frames` | `start_frame + frame_ids` |
| Viewer `frame` | Output-array row, zero based |
| `clip_time_s` | Output row / output fps |
| `source_time_s` | Original source frame / 30 |
| World transform | `(source_world - origin_source_m) * smpl_scale` for human **and** scene |
| Root rotation | Normalized WXYZ quaternion; 29 joint columns in named MuJoCo order |
| Terrain | Fixed identity pose in every frame after recentering; scale applied once |

Comments store immutable IDs, the loaded run/hash, source/output frame, terrain
assets, surface mode, text and timestamp. Save snapshots the displayed pose at
callback time; later playback or dropdown changes cannot rebind the comment.
Annotations in comparisons are matched by source frame. `addressed` stays pending
until `resolved` or `dismissed`. Inspect all events when recovering review history.

## Recipes and changes

`presets/base.json` plus `03.json`–`06.json` encode the reviewed source interval,
camera, solver and collision choices. `recipe_for(preset='03')` returns the merged
recipe. Save that dictionary to a local JSON file when selecting a new interval;
choose a new `id`/`label` to distinguish it from existing reviewed clips.

| Recipe key | Use |
|---|---|
| `source` | Sequence, start/end frames, fps=30 |
| `gender` | Must match the supplied SMPL-X model |
| `optimized`, `scene_vertices`, `scene_faces` | Optional explicit input files; defaults inside `data_root/sequences/<sequence>` |
| `crop.margin` | XY margin around all 22 source body joints, default 1.2 m; vertices cluster at 2.5 cm |
| `collision.mode` | `tiled` or `whole`; presets preserve 03/04/05 tiled, 06 whole |
| `collision.tile_size`, `corridor`, `workers` | Defaults 1.2 m / 0.3 m / 4 |
| `collision.coacd` | CoACD settings; defaults cap tiled hulls to 24/tile × 32 vertices, whole to 128 × 64 vertices |
| `config` | UMR configuration; runtime asset/output paths are rebound by the bridge |
| `target_options` | Reviewed contact corrections to use in a later target-planning iteration |

New source preparation verifies exact `SMPLXLayer` joints against supplied joints
within 5 mm. Keep the original matrices: they are not perfectly orthogonal.
The axis-angle archive is only the upstream container; the patch consumes the
exact cached geometry. Sparse footlock flags are diagnostics, not complete stance
ground truth. Use matched scene arrays; never assume two rocky clips share a scan.

For a new clip, prepare and run a bootstrap retarget first. That yields the human
scale and full G1 trajectory needed for foot/hand target planning. Then use the
recipe's `target_options` in a separate reviewed iteration. Never claim a two-epoch
correspondence smoke test or a frame-limited probe is a quality result.

Example revision override (ranges are half-open **prepared-source frame IDs**):

```json
{
  "summary": "Extend left support and preserve the accepted palms",
  "note_ids": ["COMMENT_ID"],
  "targets": {
    "stance_overrides": [{"side": "left", "start": 122, "end": 163, "stance": true}],
    "weight_boosts": []
  },
  "hand_reference": "0000_accepted",
  "config": {"solver": {"egohtr_review_foot_cost": 800000.0}}
}
```

- Omitting `targets` reuses the parent's target cache; `{}` regenerates feet.
  Regenerating feet preserves existing palms unless `hand_reference` replaces them.
- `support_surface`: `"scan"` (existing default) or `"proxy"` for stance planes
  fitted to top-facing convex collision surfaces. Proxy mode rejects underside
  hits, which can otherwise attract soles into thin slabs. `sole_clearance`
  sets the nonnegative plane clearance in metres (default `0.004`). Native
  full-sole contacts and swing-foot clearance still need an independent audit.
- `weight_boosts`: `{side,start,end,gain}` gives a sine-squared gain over that range.
- `foot_heading_overrides`: `{side,start,end,direction:"forward",weight,fade}`
  rotates only that foot's target footprint toward the pelvis-forward direction;
  use it on diagnosed swing intervals and taper interior boundaries.
  Add `preserve_tilt:true` to retain the fitted sole-plane normal when adjusting
  heading on sloped support. The default preserves earlier recipe behavior.
- `joint_posture_overrides`: `{joint,start,end,target,cost,fade}` adds a smooth,
  frame-local hinge preference. `heading_cost_overrides` similarly scales the
  pelvis heading/upright cost over `{start,end,scale,fade}`. Keep these local to
  reviewed intervals so climbing posture remains available elsewhere.
- Root continuity/velocity costs, root/joint speed bounds, native collision and
  separation penalties are explicit solver keys; defaults in `review_defaults.json`.
- `egohtr_review_orcs_terrain_cost` (default `0`, disabled) enables additional
  robot–terrain clearance constraints using ORCS's native robot geometry and FK.
  `egohtr_review_orcs_terrain_margin` sets their clearance (default `0.005` m).
  This complements the native **self**-collision check: UMR's terrain constraints
  alone use a different robot. Validate all frames and the staged 50 Hz motion
  with the task's native colliders before training.
- `egohtr_review_orcs_terrain_substeps` (default `1`) adds clearance constraints
  between source poses. Use `5` to cover the staged 30→50 Hz interpolation grid;
  it costs more solver time. `egohtr_review_exact_root_rotation_limit=true`
  uses the quaternion-difference Jacobian for the root speed constraint instead
  of the older angular box approximation (default `false` for recipe parity).
- `egohtr_review_orcs_self_substeps` (default `1`) likewise checks native self
  clearance between source poses. Use `5` when wrists/hips or limbs overlap only
  after 50 Hz resampling; increasing endpoint margins alone can miss these cases.
- `egohtr_review_initial_qpos` optionally seeds the first sequential solve with
  a complete G1 pose (root XYZ, WXYZ, native joint order). Record its source run
  and any initial clearance in the override summary. This is an optimization
  seed, not an exported trajectory offset; use it when zero-joint initialization
  starts inside a scan or chooses the wrong side of a thin surface.
- Other review keys: foot cost/separation, knee separation/flexion target/cost,
  elbow posture target/cost, heading cost, balance cost, native margin and hand
  preservation cost. Use existing `egohtr_review_*` keys from presets/patch.
- G1 elbows use their XML coordinate convention: a larger numerical coordinate
  is not automatically a more relaxed elbow. Inspect FK before changing targets.
- Assembly exclusions are the reviewed 147 body pairs in
  `assembly_exclusions.json`, tied to this G1 model. They do not disable the
  separate native-ORCS self-collision audit.

## Checks and interpretation

| Check | Interpretation |
|---|---|
| Shape/names/quaternions/finite values | Reject malformed or incompatible G1 arrays |
| Frame selection and 30 Hz | Reject truncated/misaligned outputs |
| Joint ranges | Physical and requested configuration limits |
| Native self collision / crossed feet | Review-run gates: <2 mm / zero crossings |
| Native terrain penetration | All-frame ORCS colliders; <2 mm gate when native terrain constraints are enabled |
| Joint/root speed | Recorded solver bounds with numerical tolerance |
| Proxy penetration | Body/frame outliers; proxy may fill holes in an open scan |
| Sole scan distance | Five sole probes; unsigned nearest distance alone cannot prove support |
| Vertical scan rays / proxy-minus-scan | Local layer/contact diagnosis; inspect scan holes visually |
| Source stance ankle speed | Drift indicator; includes foot rotation |
| Foot/hand target errors | Preserve contact choices and accepted palm placements |
| Root/joint derivatives; knee/elbow ranges | Detect roughness and posture tradeoffs, not stability proofs |
| Reports | Full 30 Hz videos, ffprobe frame-count/duration checks, link validation, exact-source-frame before/after images |

Read `checks.gates`, `review_required` and `limitations`; a passing kinematic audit
still needs human terrain review. The optional memory probe uses synthetic terrain,
bounded CCD allocation and passive steps; PPO, sensors, randomization and real
scene tracking are outside that measurement.

## Migration feature ledger

All 30 capabilities in the approved plan have a destination below. Contracts are
covered by `tests/test_ego_htr.py` / `tests/test_packaging.py`; actual data/solver/
viewer evidence is saved locally in `logs/ego_htr/migration_validation.json`.

| # | Preserved capability | Implementation / verification |
|---|---|---|
| 1 | Local schema, source/robot compatibility | `source.inspect_source`, `robot.load_motion`; source smoke + all four replays |
| 2 | Selective remote ZIP reads | `remote.py`; range/size refusal contract (live HF fetch needs credentials) |
| 3 | Intervals, body crop, clustering, timing | `source.py`, `terrain.crop_mesh`; fresh source preparation |
| 4 | Human reference/gallery | `report --human`; rendered 03 |
| 5 | Exact non-orthogonal source matrices | `prepare_worker` + UMR patch; alignment gate |
| 6 | Shared recenter/scale, fixed scene, frame maps | Manifests + all-frame audit |
| 7 | Whole/tiled CoACD and bounded hulls | `terrain.decompose`; preparation smoke + imported proxy counts |
| 8 | Scan/proxy rays and geometry bytes | `audit.py`, cached collision statistics |
| 9 | Isolated prepare/build/train/solve and probes | `source.py`, `umr.py`; full fresh 03, preparation smoke |
| 10 | Exact source adapter, bounded scene sampling | Version-checked UMR patch; real solver run |
| 11 | Sole/stance/widening/plane/blend planner | `targets.build`; target regeneration comparison |
| 12 | Frame-range fixes and weight boosts | JSON `targets` overrides; reviewed 03/04 presets |
| 13 | Both collision models, joint order, exclusions | `robot.export_orcs`, patch, audit; native export smoke |
| 14 | Foot/knee separation, heading, COM preference | UMR patch + solver config; fresh 03 audit |
| 15 | Root/joint continuity and normalized QP | Recorded review defaults + patch; fresh 03 parity |
| 16 | Relaxed knees/elbows and preserved palms | Preset 03, target cache, posture/hand diagnostics |
| 17 | Real G1 FK/mesh replay | `viewer.py`, `robot.py`; four-clip system test |
| 18 | Clip/version/reload; preserve frame/camera | Viewer system test; manifests replace fixed paths |
| 19 | Playback/speed/loop/seek/step | `Viewer.tick`; system test |
| 20 | Scan/proxy/overlay/hidden; opacity/wireframe/path | Viewer controls; surface-switch system test |
| 21 | Orbit/pan/zoom/fit/focus/host/port | Viser controls + launch flags |
| 22 | One terrain resident; shared play, per-client camera | Scene-handle replacement; four-clip reload test |
| 23 | Exact version-bound frame comments | Callback snapshot system test + note contracts |
| 24 | Scan/proxy videos, sheets, posters/gallery | `report.py`; full 03 videos verified |
| 25 | Annotated before/after poses and measurements | `note_images`, annotated audit; five historical 03 notes |
| 26 | Finite/limits/self/crossing checks | `audit`; full fresh 03 |
| 27 | Terrain outliers/support/target/slip diagnostics | `audit`, isolated `surface_audit` |
| 28 | Derivatives and runtime/geometry provenance | Checks/run manifests, patch hash |
| 29 | Accepted protection/history/SHA/media validation | Workspace contracts + system-test hashes |
| 30 | Synthetic simulator-memory experiment | `memory-probe`; optional diagnostic, not required for retarget review |

Historical inventory prose, discarded trials and virtual environments stay local
research artifacts. No runtime code imports the legacy log scripts. Source assets,
licensed models, learned weights, generated motions and videos remain ignored.

## PerLoco handoff

- End-to-end directory map and commands:
  [reviewed revisions to training](../../../docs/ego_htr.md#multiple-crops).
- `stage --clips LABEL...` snapshots each selected accepted run; `--run ID`
  explicitly overrides that selector. It does not accept or alter review results.
- Source adapter: `tasks/perloco/sources/egohtr.py`. Shared native staging handles
  timestamp interpolation, named joint permutation, velocities and ORCS robot FK.
- Output defaults to `data/terrain_motions/egohtr`. Each crop has `export.json`
  with source identity and content hashes, `level_0.00/tile.json`, copied hulls,
  `sample0/motion.npz` and metadata. Identical staging is reusable; changed inputs
  require a fresh `--out`, with `ORCS_EGOHTR_ROOT` pointing the task there.
  Use `data/terrain_motions/egohtr_exports/<selection>-vNN` for custom snapshots;
  leave the default root for the installed release pair. This is a naming
  convention, not an automatically allocated export directory.
- Select the returned roster with `ORCS_EGOHTR_ROSTER` for **both train and play**.
  Record both root and roster in the handoff; the roster selects families while
  each export manifest pins its reviewed motion and terrain. Later acceptance
  does not update this export. Do not hand-edit staged arrays to revise a motion.
  Without overrides, the packaged roster selects shipped 03 + 05. One source/agent task
  accepts any staged list; crop families are columns with one row, not difficulty.
- The training task is `Orcs-PerLoco-EgoHtr-AdaptSonic`. It reads staged files only:
  optional UMR, body models and review workspaces are not runtime dependencies.
- Review remains 30 Hz. Training export is 50 Hz; output row `k` maps to source
  frame `source_start_frame + k * source_fps / fps` from metadata. Check intermediate
  native-G1 contacts as well as the original frame poses; interpolation can add collisions.
- Preserve the reviewed scale: qpos is already in robot space; only proxy vertices
  receive `smpl_scale`. The terrain wrapper loads convex files into a fixed body,
  uses a catch floor below the crop, and gives sensors the same contact surfaces.
- Visual acceptance is not a physical-readiness gate. Before a long run, inspect
  native terrain/self contacts, worst-frame resets, sensor hits, actuation and PPO
  memory/overflow under the actual training domain. Create a new review revision
  for a needed motion fix; do not patch the staged joint arrays in place.
