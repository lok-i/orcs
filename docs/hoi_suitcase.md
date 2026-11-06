# HOI-Retarget suitcase

`Orcs-Uolm-Suitcase-AdaptSonic` runs one suitcase with 63 OMOMO G1 trajectories
from [HOI-Retarget](https://huggingface.co/datasets/leggedrobotics/hoi-retarget).
The environment and agent are the native UOLM recipe: frozen SONIC, LoRA rank
28/alpha 28, std multiplier 1.3, and the existing rewards, contacts, resets,
robustness and virtual-object-force curriculum.

## Setup and play

After creating and activating the repository's `.venv`:

```bash
bash scripts/setup/sync_deps.sh --no-smpl
bash scripts/setup/sync_data.sh hoiretarget --no-smpl
bash scripts/setup/sync_data.sh hoiretarget --check
play Orcs-Uolm-Suitcase-AdaptSonic --agent initial --num-envs 1 --viewer viser
```

Default `sync_data.sh` (`all`) includes this task. Named modes can be combined,
for example `sync_data.sh hoiretarget grail --no-smpl`; `--no-smpl` skips
reconstructed/GRAIL SMPL data and does not alter the suitcase roster. `--check` for HOI
verifies all installed checksums without downloading or generating files.

The initial policy is the frozen SONIC controller with its zero-initialized
adapter. Use the viewer's motion selector to inspect individual clips. A new
task does not yet have a released adapter checkpoint.

To train with the UOLM recipe and a 20,000-update phase-annealing schedule:

```bash
train Orcs-Uolm-Suitcase-AdaptSonic --env.scene.num-envs 4096 \
  --env.commands.motion.init-phase-anneal-iterations 20000 \
  --agent.max-iterations 20000
```

The existing force multiplier is 1 initially, 0.1 at update 2,500, 0.01 at
5,000, 0.0001 at 10,000, and exactly zero after 10,000. This schedule and the
agent settings are unchanged. Track `VirtualObjectForce/alpha`, object position
and orientation errors, contact consistency, and object/fall terminations.
Training returns while assistance is active are not evidence of independent
manipulation. Use trained-checkpoint playback (which removes the virtual force)
to review pickup, carry, placement and the final hold from frame zero.

For a no-VOF ablation, append
`--env.events.virtual-object-force.params.natural-frequency 0.0`.
This disables both force and torque from the first step while preserving the
phase-annealing schedule and other task settings. The logged
`VirtualObjectForce/alpha` still follows its schedule; with zero natural
frequency the applied wrench is zero regardless of that multiplier.

## Data ownership

The packaged [roster](../src/orcs/tasks/uolm/rosters/suitcase.toml) pins the
upstream revision and file checksums and lists every selected sample. It is
the runtime and staging selection, including the control `sub1_suitcase_010`
and the 14 short net-displacement clips. The selection is QC pass, complete
pickup/carry/place with carried path at least 0.5 m, and pelvis below 0.5 m
while holding.

The source pin lives with the packaged roster because setup installs a native
export rather than retaining a Git checkout. `deps.lock` continues to pin
retained dependency checkouts. No HOI-Retarget code dependency is installed;
the `hoiretarget` Python extra supplies only the staging-time Parquet reader.

The upstream OMOMO G1 shard contains other objects too (~453 MiB). Setup
downloads it and the suitcase mesh into an owned temporary directory, selects
the 63 rows, validates the complete native output, then installs it atomically.
Temporary source files are removed on success and failure. A subsequent sync
checks native checksums and does not fetch the source again. The native output
contains no raw Parquet, video data or Git/LFS object cache.

```text
data/hoi_motions/unitree_g1/
    manifest.json
    omomo/<clip>/sample1/
        motion.npz
        object_motion.npz
        contact_matrix.npz
        metadata.json
```

`ORCS_DATA_ROOT` relocates this tree. It is isolated from legacy UOLM's
`retargeted_motions` scan, so installing it does not expand the existing task.
The factory accepts `roster="suitcase"` or a TOML path and an optional
`dataset_dir` override. A different roster needs its own matching staged export.
Missing data skips only the new task; stale or invalid data raises a diagnostic.
To rebuild a stale export, move its directory aside and sync again.

For a local source already on disk:

```bash
orcs-stage-hoi --source-parquet /path/to/source.parquet --source-mesh /path/to/suitcase.obj
```

Both files must match the pinned checksums. Explicit local inputs are borrowed
and never deleted. The CLI also works as `python -m orcs.cli.stage_hoi_motions`.

## Conversion and interpretation

All channels share one 50 Hz timeline. Named joints map into native IL order;
quaternions are wxyz. Object geometry establishes the origin conversion, and
velocities are recomputed after that conversion. Robot body output comes from
the existing native FK writer. MuJoCo replay builds symmetric contact matrices
against Orcs's flat-hand G1 and `cvx_dcmp` suitcase at a 2 cm proximity threshold.
The scalar object-contact flag is derived from this matrix.

The current source and Orcs suitcase have matching vertex topology but slightly
different scale (~0.8167 native versus 0.83 source). Staging preserves the
source robot/world heights and does not independently lower the object. The
kinematic references can contain floating objects and foot-floor penetration;
source contacts can disagree with the native geometry. `manifest.json` records
per-clip contact recall and floor distances; `contact_matrix.npz` also preserves
the source labels and measured distances (object, left foot, right foot).
These diagnostics are evidence for playback and later control evaluation,
not a guarantee that every source clip is dynamically executable.

In the current export, `sub5_suitcase_046` is the contact outlier: only about
15% of its source right-palm contact frames have native right-wrist/object
contact within 2 cm. During these source-labeled contact frames, the nearest
native hand/object surface gap has median 2.1 cm and 90th percentile 2.8 cm;
the binary threshold makes a small visual gap look like a large recall drop.
This is a reference-geometry diagnostic, not a policy success rate. Training
uses the replayed native contact matrix rather than the upstream palm labels.
The clip remains in the 63-clip roster for explicit review.

When assessing a trained adapter, evaluate all 63 clips from frame zero through
placement and hold with virtual assistance disabled. Keep unsuccessful clips
in the denominator. See [third-party notices](../THIRD_PARTY_NOTICES.md) for
attribution and upstream data terms.
