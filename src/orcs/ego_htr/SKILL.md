---
name: egohtr-review
description: Prepare EgoHTR terrain clips, retarget G1 with the optional external UMR bridge, and revise motions from frame comments in the ORCS Viser viewer. Use for EgoHTR clip preparation, contact/posture fixes, review comparisons, acceptance, or pruning these local retarget runs.
---

# EgoHTR review

Keep the human interface to one viewer command and this chat. Own the preparation,
configuration, diagnostics and iteration bookkeeping as the agent.

## Start

1. Read [agent_tools.md](agent_tools.md) for commands, manifests and coordinates.
2. Run `.venv/bin/python -m orcs.cli.ego_htr doctor` and `inspect --clip <label>`.
3. Run `notes --clip <label> --pending`. Resolve every comment to its recorded
   revision, motion hash, source frame and terrain before interpreting it.
4. Read the requested run's `run.json`, its checks and the accepted baseline.
   Use exact run IDs in follow-up commands; `Latest` can change during review.

The workspace defaults to `logs/ego_htr`; `--workspace PATH` goes **before** the
subcommand. Use the current environment's Python if `.venv/bin/python` is absent.
An installed package exposes these same tools as `orcs-egohtr`.

## Revise

1. Diagnose the annotated frames and neighboring transitions. Distinguish scan
   holes/proxy overfill, source stance errors, leg crossings, posture and pacing.
2. Make explicit JSON overrides for the requested change. Link note IDs and
   explain the intended effect in `summary`. Keep source interval/scene fixed
   unless the user requests a different crop.
3. Use `retarget --parent <reviewed-run-id> --label <short-name> --overrides FILE`.
   This creates a new revision, solves in the isolated UMR interpreter and audits
   every output frame. A failed run keeps the previous Latest pointer.
4. Compare against the parent and accepted reference with `audit` or `report`.
   Check sole gaps and support, crossings, both robot collision models, hands,
   knees/elbows, root/joint velocities, acceleration and jerk. Do not optimize
   one scalar while silently degrading the other reviewed properties.
5. Mark linked notes `addressed` with the new run ID and concise evidence.
   Return the candidate ID and ask the user to **Reload files → Latest**.
6. Mark notes `resolved` and use `accept` after the user accepts that candidate.
   Do not infer acceptance from geometric checks or a timeout.

## Preserve

- Accepted motion, source, terrain and completed run metadata are immutable.
  Never rewrite an NPZ or shared cache to revise a motion; allocate a new run.
- Preserve existing hand targets when changing feet. An explicit
  `hand_reference` replaces them using that revision's G1 palms.
- Use original optimized rotation matrices through `SMPLXLayer`. Axis-angle
  round trips change this dataset's contact geometry.
- Keep source frames at 30 Hz and stride 1 when using review constraints.
  Apply the same recentering and scale to the human and stationary terrain.
- Avoid blind trajectory smoothing: it can undo foot placement and separation.
- Defaults are lean: `run.json`, `motion.npz`, `run.log`. Generate `reports/`
  or `debug/` only when useful. Preview `prune` before applying requested cleanup;
  it protects latest/accepted/running revisions and leaves shared caches alone.
- UMR remains optional and external. Inspect its commit/patch status; do not
  upgrade its dependencies into the ORCS training environment.
- Do not treat playback or kinematic checks as a policy-stability/OOM test.

## Migration review state

- 03: reviewed contact/posture result; forward hunch and residual roughness parked.
- 04: reviewed contact fixes; preserve unless new feedback requests a change.
- 05 and 06: accepted references; preserve.
- Historical imported notes retain their original before-feedback revisions.
  Resolved historical notes are evidence, not new requests to repeat the fixes.
- A fresh 03 migration solve should reproduce the parked reference; no new
  posture tuning is part of migration validation.
