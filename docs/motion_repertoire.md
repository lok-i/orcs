# Motion repertoire benchmark

`orcs-bench-motion-repertoire` measures one policy across every distinct playable
clip in a registered task. Each clip receives exactly K trials. Objects and
terrains determine valid scene assignments, but never change the per-clip quota.
This measures coverage of the evaluated library; calling it held-out
generalization requires a separately identified evaluation split.

```bash
orcs-bench-motion-repertoire Orcs-Uolm-Suitcase-AdaptSonic \
  --wandb-run-path lkrajan/orcs/zih7cd4i \
  --num-envs 4096 --trials-per-motion 65

# A local checkpoint; no W&B access or writes.
orcs-bench-motion-repertoire Orcs-Uolm-Suitcase-AdaptSonic \
  --checkpoint-file logs/rsl_rl/orcs_uolm_suitcase/2026-11-06_18-00-37/model_20998.pt \
  --num-envs 4096 --trials-per-motion 65

# Clean reference check with the released terrain policy, including serial batches.
orcs-bench-motion-repertoire Orcs-PerLoco-EgoHtr-AdaptSonic \
  --agent release --nominal --trials-per-motion 1 --num-envs 1

# Initial actor, with no training or optimizer steps.
orcs-bench-motion-repertoire Orcs-Dodge-AdaptSonic \
  --agent initial --num-envs 16 --trials-per-motion 16 --no-serve
```

Install/update the editable package after pulling this feature so its console
entry point is available: `uv pip install --no-deps -e .`. The equivalent module
command is `.venv/bin/python -m orcs.cli.bench_motion_repertoire`.

## Evaluation conditions

| Setting | Behavior |
|---|---|
| Domain | Retain the registered training task's randomization |
| `--nominal` | Remove robustness randomization, RSI jitter and observation noise; retain task events |
| Assistance | Off; UOLM's virtual object force is removed |
| Start | Frame zero; native reset/observation timing is preserved |
| Actor | Deterministic inference through the registered runner; strict checkpoint loading |
| Completion | Native clip + configured hold, or native episode horizon for Dodge |
| Failures | Native task failures, including those normally disabled for interactive play |
| `--num-envs` | Concurrency cap, not sample count; default 4096 |
| `--trials-per-motion` | K; default 65 |
| `--seed` | Startup/reset: seed + batch index; runtime events/noise: batch seed + 1,000,000 |

The collector scores only the first episode in each world. It builds successive
batches when M×K exceeds capacity, preserving the full motion roster and policy
observation widths. Each batch gets fresh startup randomization and the correct
object variant or terrain tile. Automatic resets after a scored termination do
not add trials. Very small batches cost extra scene/runner construction time.
Changing concurrency also changes random draws; repeat comparisons with the
same seed **and** batch configuration.

For a local checkpoint, adjacent `params/agent.yaml` restores its actor/critic,
observation groups and action clipping. Otherwise the registered recipe is used.
`--agent-config /path/to/agent.yaml` explicitly selects an inference recipe,
including architecture settings that cannot be inferred from tensor shapes.
Referenced base-controller paths must exist on this machine. The report records
the selected recipe. The environment always comes from the current registered
task, not a historical training snapshot; its resolved configuration is saved.

## Report and saved records

One invocation collects, aggregates, saves, and serves. By default results live
under `logs/eval/<task>/<source>/<checkpoint-label>/<timestamp>/`. W&B's exact
resolved checkpoint name and checksum are in `receipt.json`; the folder label
is the requested name, or `trained` when selecting the latest remote checkpoint.
`--output <new-directory>` chooses an explicit destination and refuses overwrite.

The terminal prints a clickable `http://localhost:<port>/report.html` link.
The simulator runs in a child process and exits before the local CPU server
starts, releasing its CUDA context. Ctrl-C stops serving and keeps the files.
`--no-serve` saves and exits. Use `--port` for a specific port; otherwise a free
one is chosen. The server binds only to loopback and serves the report directory.
For a remote machine, forward that port through SSH or VS Code.

The portable output folder also opens directly from `report.html` without a
server or external CDN. Layout: overview cards → failure bars / timing heatmap /
progress tabs → sortable, searchable motion table → exports and protocol.

| Artifact | Contents |
|---|---|
| `receipt.json`, `config.json` | Conditions, repertoire paths/hashes, checkpoint/recipe hashes, source/version receipt, resolved configuration |
| `trials.jsonl` | Versioned per-trial outcomes, generic errors, compact time-bin diagnostics; written after each batch |
| `summary.json`, `per_motion.csv`, `trials.csv` | Derived aggregates and export tables |
| `report.html`, `*.png`, `*.svg` | Portable report and shareable figures |
| `collection.log`, `source.patch` | Collection output and tracked checkout diff, when available |

```bash
# Recompute figures, tables and HTML from recorded facts; no simulator/GPU import.
orcs-bench-motion-repertoire --report-only logs/eval/<saved-result-directory>
```

This can repair aggregation or presentation without rerunning expensive physics.
It cannot reconstruct unrecorded signals or repair collection bugs. Interrupted
collection reports completed batches and, on graceful shutdown, the active batch;
unscheduled trials remain pending. Resuming collection is outside this version.

## Reading the metrics

- **Completion/failure:** native outcomes; simultaneous failure and completion
  counts as failure. Unexpected truncation is censored. Nonfinite actions,
  rewards or measured states are invalid. Invalid, censored and pending trials
  remain explicit and in the scheduled completion denominator.
- **Progress:** normalized reference time reached, with the final hold shown
  separately. Reaching the last frame does not establish successful stabilization.
  Dodge uses normalized episode time. Mean progress uses observed valid trials.
- **Failure timing:** failures in each bin divided by trials entering that bin;
  gray means unvisited. This is post-training failure timing, not the number of
  training iterations until success shown in adaptive-sampling papers.
- **Global/root-relative point error:** mean Euclidean distance over matched
  named bodies or source points. Root-relative error subtracts each trajectory's
  own root position; it does not rotate/yaw-align or perform Procrustes alignment.
- **Root pose / point velocity error:** root translation in metres, quaternion
  angular distance in radians, mean matched-point velocity error in m/s.
  Native commands use their body references; seed-backed SMPL commands use
  authored/scaled point targets, not robot reset seeds. Root means the configured
  anchor (pelvis in the current tasks).

Errors average samples within each trial, trials within each motion, and then
motions equally. The table toggles observed-prefix and completed-only errors;
neither should be interpreted without completion and progress. Dodge's nominal
pose deviations are diagnostics, not a failure criterion. Missing measurements
are displayed as unavailable, not zero. Termination tags can overlap; their sum
need not equal the number of failed trials.

Per-motion 95% Wilson intervals appear only for complete, valid randomized
coverage. Nominal repetitions are not independent statistical evidence. No
pooled interval claims uncertainty about a different population of motions.

## Extending a task

`core.motion_benchmark` owns scheduling, capture, aggregation and reporting. It
does not import task or asset packages. A task family supplies a `BenchmarkSpec`
to `register_all(..., benchmark=SPEC)`:

- `prepare(cfg, nominal)`: remove assistance; optionally remove robustness noise.
- `configure_batch(cfg, clips)`: assign immutable scene variants before construction.
- `bind_batch(env, clips)`: assign runtime terrain contexts before reset.
- `horizon`, `completion_term`, `tracking_quality`: declare native outcome meaning.

`MultiClipMotionCommand` exposes the resolved catalog, scene-context IDs, valid
motion masks, one-shot `queue_reset_clips`, and phase/hold accessors. Its ordinary
training sampler is unchanged when no reset selection is queued. Task subclasses
provide context IDs. Reference error capture uses the command's public native
body or authored-point accessors. Every registered ORCS task family supplies a
contract; optional missing datasets continue to skip registration normally.

The collector adds an ordinary MJLab metrics term to observe termination flags
and matched states before automatic reset. It neither replaces `env.step` nor
patches private manager methods. Registered runner construction and actor-only
loading are shared with play; no learning loop or logger upload is invoked.
