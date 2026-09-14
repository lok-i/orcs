"""Shared-dependency SHA guard.

`orcs` and its consumers (today: `vibe`) both pin `mocke` and `rsl_rl` in their
own ``deps.lock``, but each is a single editable install — one SHA per env,
whichever sync ran last. When orcs is the dependency, the CONSUMER's lock wins
and orcs's own lock becomes advisory.

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
    "mocke": "b50fb6d11e7616598dec8b444e88c55b559ec87b",
    "rsl_rl": "215b6f4ceda569d2ca39df64dd679b9af15375cb",
}
"""SHAs orcs was last validated against. Bump only after `play Orcs-Uolm-AdaptSonic
--agent initial` still rolls the frozen base bit-exact."""


NOT_GIT = "not-a-git-checkout"
"""`_head` result for a dep that imports from a published wheel instead of its
pinned editable fork — a later `pip install` resolved it off PyPI and uninstalled
the fork. The replacement is silent and total (e.g. `rsl_rl` loses
`SonicWithAdapterModel`), so it is drift, not an absence."""


def _head(pkg: str) -> str | None:
    """Installed package's git HEAD, `NOT_GIT` if it imports from somewhere that
    is not a git checkout, or None if it is absent / undecidable."""
    try:
        spec = importlib.util.find_spec(pkg)
    except (ImportError, ValueError):
        return None
    if spec is None or not (spec.origin or spec.submodule_search_locations):
        return None
    root = Path(spec.origin).parent if spec.origin else Path(
        list(spec.submodule_search_locations)[0])
    try:
        proc = subprocess.run(
            ["git", "-C", str(root), "rev-parse", "HEAD"],
            capture_output=True, text=True, timeout=5,
        )
    except (OSError, subprocess.SubprocessError):
        return None  # no git binary or it hung: cannot tell, so stay quiet
    if proc.returncode != 0:
        return NOT_GIT
    return proc.stdout.strip() or NOT_GIT


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
            print(f"  {pkg:8s} validated {want[:7]}  "
                  f"live {live if live == NOT_GIT else live[:7]}")
        if any(live == NOT_GIT for _, live in drift.values()):
            print("  ^ a pinned editable fork was replaced by a published wheel. "
                  "Re-run scripts/setup/sync_dependencies.sh — it must be the "
                  "last install in the env.")
        if "mocke" in drift:
            print("  ^ mocke carries the frozen-WBC obs/action contract the ported "
                  "SONIC ckpts are bit-coupled to. Re-verify with "
                  "`play Orcs-Uolm-AdaptSonic --agent initial`.")
    if strict and drift:
        raise RuntimeError(f"shared dependency drift: {sorted(drift)}")
    return drift


__all__ = ["VALIDATED", "check"]
