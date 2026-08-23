"""Oracle Robot Control Synthesis: shared infrastructure for privileged policies.

The package root is deliberately lightweight. Importing :mod:`orcs` exposes
the library without importing MJLab or registering tasks; MJLab discovers
``orcs.registration`` through the ``mjlab.tasks`` entry-point group.

``SKIP_REASON`` and ``MULTI_CLIP_CFGS`` remain convenient public diagnostics.
Accessing either explicitly asks for task discovery and loads MJLab when it has
not already been imported.
"""

from __future__ import annotations

from importlib import import_module
from typing import Any

__all__ = ["MULTI_CLIP_CFGS", "SKIP_REASON"]

_REGISTRATION_EXPORTS = frozenset(__all__)


def __getattr__(name: str) -> Any:
    """Load registration-only public state on first explicit access."""
    if name not in _REGISTRATION_EXPORTS:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

    # Importing mjlab runs its task entry-point discovery. The ORCS entry point
    # loads orcs.registration; importing that module again below is then a
    # cheap, idempotent lookup. Keeping this out of package import is what lets
    # orcs.cli entry points start without recursively loading sibling packages.
    import_module("mjlab")
    registration = import_module("orcs.registration")
    value = getattr(registration, name)
    globals()[name] = value
    return value


def __dir__() -> list[str]:
    return sorted((*globals(), *_REGISTRATION_EXPORTS))
