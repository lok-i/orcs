"""Filesystem roots — the single source of path truth. No caller does ``__file__``
depth math, so moving a module never silently orphans a dataset.

Resolution order for every root: env override -> auto-find -> marker walk-up.

  ORCS_ROOT         repo root                (default: walk up for _MARKERS)
  ORCS_DATA_ROOT    datasets                 (default: <repo>/data)
  ORCS_DEPS_ROOT    synced dependencies      (default: <repo>/dependencies)
  ORCS_ASSETS_SOURCE  robot/object asset tree (default: installed ``assets`` pkg,
                      else <deps>/assets/source)

``REPO_ROOT``/``DATA_ROOT``/``DEPS_ROOT`` are import-time constants and never
assert existence — absent data must degrade to a skipped task registration, not
a broken ``import orcs``. :func:`assets_source` validates and is therefore lazy:
it raises only when an asset is actually requested.
"""

from __future__ import annotations

import importlib.util
import os
from functools import lru_cache
from pathlib import Path

_MARKERS = ("pyproject.toml", "deps.lock")
"""Files that mark the repo root — both must be present."""

_ASSETS_SENTINEL = "g1/meshes"
"""Child that must exist for an assets-source candidate to be accepted."""


def _env(var: str) -> Path | None:
    val = os.environ.get(var)
    return Path(val).expanduser().resolve() if val else None


def _repo_roots() -> list[Path]:
    """Every repo root above this file, nearest first.

    Nearest is orcs's own checkout. When orcs is INSTALLED AS A DEPENDENCY it
    sits at ``<host>/dependencies/orcs/``, so the next entry is the host repo —
    which is where the data actually lives, because a dependency checkout is
    code only (its own ``data/`` and ``dependencies/`` are gitignored and never
    synced).
    """
    here = Path(__file__).resolve()
    return [d for d in here.parents if all((d / m).exists() for m in _MARKERS)]


_ROOTS = _repo_roots()
if not _ROOTS:
    raise FileNotFoundError(
        f"orcs repo root not found above {Path(__file__).resolve()} — no "
        f"ancestor holds all of {_MARKERS}. Set ORCS_ROOT to override."
    )


def _resolve(var: str, child: str) -> Path:
    """Env override, else the nearest repo root whose ``child`` exists, else ours.

    The fallback is what makes orcs work unmodified as a library: a host project
    that vendors orcs under ``dependencies/`` keeps its datasets at its OWN root,
    and orcs finds them without the host having to set anything. Falling back to
    ``_ROOTS[0]`` when nothing exists keeps the path well-defined for error
    messages instead of raising at import.
    """
    override = _env(var)
    if override:
        return override
    for root in _ROOTS:
        if (root / child).is_dir():
            return root / child
    return _ROOTS[0] / child


REPO_ROOT: Path = _env("ORCS_ROOT") or _ROOTS[0]
DATA_ROOT: Path = _resolve("ORCS_DATA_ROOT", "data")
DEPS_ROOT: Path = _resolve("ORCS_DEPS_ROOT", "dependencies")


def _assets_candidates() -> list[Path]:
    """Ordered guesses for the ``assets`` source tree, best first.

    The in-repo checkout wins: ``deps.lock`` pins its SHA for *this* repo, so it
    outranks whatever ``assets`` happens to be pip-installed (a shared editable
    install may belong to a sibling project at a different SHA). ``find_spec``
    is the fallback that makes an out-of-repo install work.
    """
    out: list[Path] = [DEPS_ROOT / "assets/source"]
    try:
        spec = importlib.util.find_spec("assets")
    except (ImportError, ValueError):
        spec = None
    if spec is not None:
        origin = spec.origin if spec.origin not in (None, "namespace") else None
        locs = list(spec.submodule_search_locations or [])
        pkg = Path(origin).parent if origin else (Path(locs[0]) if locs else None)
        if pkg is not None:
            pkg = pkg.resolve()
            out += [pkg, pkg / "source", pkg.parent, pkg.parent / "source"]
    return out


@lru_cache(maxsize=1)
def assets_source() -> Path:
    """The ``assets`` dep's source tree (robot meshes + generated object XMLs).

    Lazy + validated: a candidate counts only if it holds ``_ASSETS_SENTINEL``,
    so a same-named PyPI package can't shadow the real tree. An explicit
    ``ORCS_ASSETS_SOURCE`` is never silently ignored — a wrong one raises.
    """
    override = _env("ORCS_ASSETS_SOURCE")
    if override:
        if (override / _ASSETS_SENTINEL).is_dir():
            return override
        raise FileNotFoundError(
            f"ORCS_ASSETS_SOURCE={override} holds no '{_ASSETS_SENTINEL}' — "
            "unset it to fall back to auto-discovery."
        )
    tried = _assets_candidates()
    for cand in tried:
        if (cand / _ASSETS_SENTINEL).is_dir():
            return cand
    listed = "\n  ".join(str(p) for p in tried)
    raise FileNotFoundError(
        f"assets source tree not found (no candidate holds '{_ASSETS_SENTINEL}'):"
        f"\n  {listed}\nSync it with scripts/setup/sync_dependencies.sh, or set "
        "ORCS_ASSETS_SOURCE."
    )
