# Contributing

- Setup and commands: [README](readme.md).
- Architecture and layer rules: [ethos](docs/ethos.md#4-layer-contract).
- PR structure: [template](.github/PULL_REQUEST_TEMPLATE.md).

## Write for review

- **Points over paragraphs. Tables for comparisons. Links for detail.**
- Lead with the problem and resulting behavior; describe the final change.
- Use short phrases, concrete names and direct links to code, docs or upstream PRs.
- State release impact and any required migration; use **None** for no breaking changes.
- Delete irrelevant template rows and sections; keep small PRs small.
- Report commands and observed results, including skipped checks and known limits.
- Keep development history and local debug output out of the description.

## Changes and dependencies

| Change | Include |
|---|---|
| Task or runtime behavior | Reproduction command; relevant rollout evidence |
| Public API or configuration | Before/after behavior; migration if needed |
| Git dependency | Upstream PR; old → new SHA in `deps.lock`; setup verification |
| Released checkpoint | HF review; immutable revision and checksum in `src/orcs/release.json` |
| Third-party code or data | Attribution, license and access conditions; [notices](THIRD_PARTY_NOTICES.md) |

- Change dependencies in their own repositories; repin ORCS after upstream review.
- Keep temporary PR-head pins explicit; use merged commits for release.
- Keep raw data, checkpoints, recordings and debug logs out of source commits.

## Validate the change

| Scope | Check |
|---|---|
| Every PR | `git diff --check`; review the final diff |
| Python | `.venv/bin/ruff check <changed-python-files>`; relevant `.venv/bin/pytest -q tests/...` |
| Packaging or shared infrastructure | `.venv/bin/pytest -q` |
| Runtime behavior | `play <task> --agent initial`; relevant trained/release rollout |
| Dependency pins | `bash scripts/setup/sync_deps.sh --check`; `bash scripts/setup/sync_data.sh --check` with the installed data mode |
| Documentation only | Verify links, commands and examples; no simulation run needed |

- Add tests for meaningful contracts; use rollouts to verify simulation behavior.
- State what was not checked and why. Link larger evidence instead of pasting logs.
