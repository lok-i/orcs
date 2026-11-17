# Adaptive motion sampling

Adaptive sampling changes training resets in the shared `MultiClipMotionCommand`.
It chooses a compatible clip and an eligible start phase using recent failures;
it does not change the task, agent, observations, rewards or reference data.

```bash
train Orcs-Uolm-Suitcase-AdaptSonic \
  --env.commands.motion.sampling-mode adaptive \
  --env.commands.motion.adaptive-kernel-size 3 \
  --env.commands.motion.adaptive-start-from-zero-ratio 0.25
```

The same flags apply to the current UOLM, PerLoco and seed-backed SMPL command
families. The [suitcase recipe](hoi_suitcase.md#adaptive-motion-sampling) adds
the 21k training horizon, phase annealing and no-VOF setting.

## Scene constraints

Every reset reads the world's current allowed-clip mask. Both adaptive selection
and the uniform full-start mixture obey that mask. The sampler does not move
worlds between object variants or terrain tiles, or replace a task's curriculum.

| Repertoire | Available adaptation |
|---|---|
| Suitcase: one scene, 63 clips | Clip selection across all 63 and start phase |
| GRAIL: several clips per terrain tile | Clip selection within the current tile and start phase |
| Legacy native UOLM / current OmRe: one clip per context | Start phase; scene assignment determines the clip |
| Seed-backed SMPL | The same clip/phase rules, with authored targets and seed-based resets |
| Dodge: one nominal pose, starts at zero | No meaningful clip/phase choice |

These are properties of the current rosters, not task-name branches in the
sampler. A changed terrain assignment takes effect at the next reset.

## Distribution and phase limits

- Each clip owns roughly one-second time bins. A native failure contributes
  once, even if several termination terms fire. Completion/timeouts alone do
  not contribute. Failures during the final hold map to that clip's last bin.
- Failure counts use an exponential moving average (`adaptive_alpha=0.001`),
  advanced once per control step. They are not exposure-normalized failure rates.
- `adaptive_kernel_size` defaults to 1. A size of 3 with
  `adaptive_lambda=0.8` biases starts toward a failure bin and its two preceding
  bins. Lookback never crosses clip boundaries.
- Summed bin weights select clips; conditional bin weights select phases.
  Annealing restricts eligible phases, while clip difficulty uses the whole
  motion. Clip prioritization therefore continues after the start phase reaches
  zero.
- `adaptive_start_from_zero_ratio=0.25` selects an allowed clip uniformly and
  starts at zero for a quarter of resets. The remaining resets use adaptive
  weights with `adaptive_uniform_ratio=0.1` as an additive prior. The latter is
  not a guaranteed 10% uniform mixture. With no failures, its clip mass scales
  with the number of valid bins, rather than assigning equal mass to every clip.
- `start_from_zero=True` forces all phases to zero, retaining clip adaptation.
  `init_phase_anneal_iterations=0` disables annealing; it does not force phase zero.

The default `sampling_mode="start"` and the legacy `"uniform"` mode preserve
ORCS's existing RSI behavior: eligible-length-weighted clips and uniform frames,
becoming uniform over allowed clips at phase zero. `start_from_zero=True` is
their explicit frame-zero override. These legacy mode names do not have the
same semantics as MJLab's single-file command.

## Logging, resume and evaluation

`Metrics/motion/sampling_entropy`, `sampling_top1_prob` and
`sampling_start_from_zero` describe clip concentration and realized full starts.
`MotionSampling/clip_<index>:<identity>` records mean clip probabilities for the
latest reset batch, including scene masks and the uniform mixture. This is not
whole-run exposure or the fraction of gradients contributed by a clip. The
legacy `sampling_top1_bin` metric is unused by the multi-clip sampler.

Sampler histories are runtime-only and start fresh on resume. Policy/optimizer
checkpoint handling is unchanged; a resume is not an exact continuation of the
sampling distribution.

The [motion repertoire benchmark](motion_repertoire.md) queues balanced,
frame-zero assignments. Those episodes bypass adaptive selection and failure
statistics, preserving the same evaluation protocol for both training recipes.

The implementation follows the failure histogram/EMA and lookback design in
[BeyondMimic](https://github.com/HybridRobotics/whole_body_tracking/blob/main/source/whole_body_tracking/whole_body_tracking/tasks/tracking/mdp/commands.py)
and [MJLab](https://github.com/mujocolab/mjlab/blob/main/src/mjlab/tasks/tracking/mdp/commands.py).
ORCS adds per-clip boundaries, scene compatibility, full-start practice and its
existing phase limits.
