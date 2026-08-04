"""Shared-dependency SHA guard.

`orcs` and its consumers (today: `vibe`) both pin `mocke` / `rsl_rl` / `assets` in
their own ``deps.lock``, but each is a single editable install — one SHA per env,
whichever sync ran last. When orcs is the dependency, the CONSUMER's lock wins and
orcs's own lock becomes advisory.

That is fine as long as the drift is visible. `mocke` in particular carries the
frozen-WBC obs/action contract that the ported SONIC checkpoints are bit-coupled
to: a silent bump loads cleanly and produces garbage. This module turns that into
a printed line at import.

Warn, never raise — a checkout without git, or a consumer deliberately running
ahead, must still import.
"""

from __future__ import annotations

import importlib.util
import os
import subprocess
from pathlib import Path

VALIDATED: dict[str, str] = {
    "mocke": "004d4c02bb1be65c67688f25f95197c865f83d6f",
    "rsl_rl": "4f8b9b3b657f27397a3337dabafb7ef7398387b4",
}
"""SHAs orcs was last validated against. Bump only after `play Orcs-Uolm-AdaptSonic
--agent initial` still rolls the frozen base bit-exact."""


def _head(pkg: str) -> str | None:
    """Installed package's git HEAD, or None if it is not a git checkout."""
    try:
        spec = importlib.util.find_spec(pkg)
    except (ImportError, ValueError):
        return None
    if spec is None or not (spec.origin or spec.submodule_search_locations):
        return None
    root = Path(spec.origin).parent if spec.origin else Path(
        list(spec.submodule_search_locations)[0])
    try:
        return subprocess.run(
            ["git", "-C", str(root), "rev-parse", "HEAD"],
            capture_output=True, text=True, timeout=5,
        ).stdout.strip() or None
    except (OSError, subprocess.SubprocessError):
        return None


def check(strict: bool = False) -> dict[str, tuple[str, str]]:
    """Report {pkg: (validated, live)} for every shared dep that drifted.

    `ORCS_SKIP_DEP_CHECK=1` short-circuits it: this shells out to `git` once per
    dep on every `import orcs`, which is free on a laptop and not free on a
    cluster filesystem where a 4096-env job pays it per rank.
    """
    if os.environ.get("ORCS_SKIP_DEP_CHECK"):
        return {}
    drift = {
        pkg: (want, live)
        for pkg, want in VALIDATED.items()
        if (live := _head(pkg)) is not None and live != want
    }
    if drift:
        print("[orcs.core.deps] shared deps differ from orcs's validated set "
              "(consumer lock wins — heads-up, not an error):")
        for pkg, (want, live) in drift.items():
            print(f"  {pkg:8s} validated {want[:7]}  live {live[:7]}")
        if "mocke" in drift:
            print("  ^ mocke carries the frozen-WBC obs/action contract the ported "
                  "SONIC ckpts are bit-coupled to. Re-verify with "
                  "`play Orcs-Uolm-AdaptSonic --agent initial`.")
    if strict and drift:
        raise RuntimeError(f"shared dependency drift: {sorted(drift)}")
    return drift


__all__ = ["VALIDATED", "check"]
