# post-release implementation todo

Implementation and infrastructure work only; none of these block the current
release.

## high leverage

- [ ] Replace the global MJLab monkey patches with upstream extension hooks for
  multi-clip commands, `--agent initial`, and simulator configuration.
- [ ] Expose an explicit, idempotent `orcs.register()` API returning a structured
  registration report; keep the MJLab entry point as a thin caller.
- [ ] Register task config factories lazily instead of constructing every config
  and probing datasets during plugin discovery.
- [ ] Version generated asset and motion-data contracts. Record schema version,
  generator SHA, source provenance, coordinate convention, and inputs in each
  generated dataset; reject stale caches with a regeneration command.

## hardening

- [ ] Replace import-time path constants with an explicit `RuntimeLayout` object
  covering vendored, standalone, and wheel installs.
- [ ] Add one small headless reset/step smoke test for UOLM, PerLoco, Dodge, and
  SMPL tasks to catch config, sensor, observation, and model-loading breakage.
- [ ] Add CI for Ruff, contract tests, clean wheel installation, console scripts,
  task registration, and both standalone and vendored layouts.
- [ ] Give `deps.lock` a validated schema and one command that reports code,
  generated-data, and asset-generator drift.

## polish

- [ ] Replace scattered diagnostic prints with structured, consistently
  suppressible logging.
- [ ] Define a small stable public API for registration, runtime layout, assets,
  agents, and dataset contracts; treat task implementation modules as internal.
- [ ] Move historical implementation narratives from source comments into short
  decision records, leaving active invariants beside the code.
