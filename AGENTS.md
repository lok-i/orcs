# Working on ORCS

ORCS (Optimize, Retarget, Control Suite) owns privileged humanoid control and
retargeting. Tasks compose shared mechanics, assets and agent factories.
ViBe consumes those mechanics for vision adaptation; downstream applications
own their task selection, language interfaces and evaluation worlds.

## Read first

- [readme.md](readme.md): setup, task IDs, train/play and retargeting commands.
- [docs/ethos.md](docs/ethos.md): ownership, layers and task-extension rules.
- [CONTRIBUTING.md](CONTRIBUTING.md): validation and review requirements.
- [CLAUDE.md](CLAUDE.md): additional UOLM and frozen-controller rationale.
- [docs/smpl_retargeting.md](docs/smpl_retargeting.md): seed-backed SMPL tasks.
- [docs/ego_htr.md](docs/ego_htr.md): EgoHTR staging and assisted retargeting.
- [docs/hoi_suitcase.md](docs/hoi_suitcase.md): native suitcase roster and staging.
- [docs/motion_repertoire.md](docs/motion_repertoire.md): balanced policy benchmark and task hooks.

## Extension rules

- Reuse the task's environment factory for a new scene or motion collection.
  A registered task ID does not require another task package or copied MDP.
- Keep shared agents in `core/rl.py`. Task registration selects an agent and
  its recipe; tasks do not define their own PPO implementation.
- `core` must not import `assets` or `tasks`; `assets` must not import tasks;
  task packages must not import sibling tasks. Use `orcs.core.paths` for data
  and dependency paths.
- Source conventions belong in staging adapters. Runtime consumes normalized
  ORCS data. CLI implementation lives in `src/orcs/cli`, with entry points in
  `pyproject.toml`; package runtime rosters and other non-Python resources.
- Preserve existing default rosters when adding a dataset. Native UOLM scans
  `retargeted_motions/data/unitree_g1` by metadata object identity: adding files
  there can change an existing task without changing its registration row.
- Registration is per task row. Missing optional data populates `SKIP_REASON`;
  invalid configuration must not be disguised as missing data. Bare
  `import orcs` stays lightweight; import `mjlab` first in standalone scripts
  that need task discovery.
- Registered UOLM SMPL recipes use `uolm_smpl_env_cfg`. The older
  `uolm_env_cfg(command_space="smpl")` path is for rollout/debugging.
- Motion-repertoire benchmarks use the training task plus its `BenchmarkSpec`,
  preserving native failures and full observation widths. New task families
  declare scene assignment and horizon hooks; core never branches on task names.

## Data and controller contracts

- `motion.npz` uses IsaacLab joint order; runtime maps it to MuJoCo order.
  Map source joints by name and reuse `cli/_motion_npz.py` for native FK/body
  output. Its 37-row body arrays populate the 14 tracked body rows only.
- Object and robot trajectories must share timing and world transforms.
  `ContactSchedule` uses its own body-name legend, not motion-body row order.
  Missing contact files are not evidence of no contact: partial absence falls
  back to zeros, while absence for every clip raises.
- Preserve SONIC observation/action order and tokenizer contracts in `mocke`.
  Dependency changes belong upstream, followed by a reviewed lock update.
- The consumer's dependency lock wins. Compare actual pins: standalone ORCS
  and ViBe can use different assets/data revisions even with the same ORCS
  source. Do not run ORCS setup inside a consumer's environment.
- Apply play overrides after the training domain is assembled. Configuration
  tests cannot establish rollout, rendering or checkpoint compatibility.
  A changed object roster can also change object-identity observation widths.

## Validation and artifacts

Follow `CONTRIBUTING.md`: diff checks for every change, focused lint and tests
for Python, the full suite for packaging/shared infrastructure, and relevant
initial/trained rollouts for runtime behavior. Inspect registration skips when
interpreting test results. Documentation-only changes need link/command checks,
not simulation. State which checks ran and what they do not establish.

Keep raw data, generated caches, checkpoints and experiment receipts out of
source commits. Preserve source identity, revisions, conversion provenance and
license notices when staging third-party data. Research notes belong in
`logs/`; durable usage and architecture belong in `docs/`.
