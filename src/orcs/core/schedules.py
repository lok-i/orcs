"""THE annealing ramp every orcs curriculum is written against.

One primitive, one naming convention — obey it or the annealers drift apart:

    <what>_anneal_start : int    cfg — policy update where the ramp begins
    <what>_anneal_end   : int    cfg — where it saturates; end <= start = OFF
    alpha_<what>_init   : float  cfg — value held before `start`, and forever
                                 when the window is empty
    alpha_<what>        : float  live value on the term
    <What>Annealing/alpha : str  the wandb key

Two scalars, not a `(start, end)` tuple: mjlab's `train` parses argv in two
stages (`return_unknown_args=True`), and NO multi-value flag survives that — a
tuple field would be settable in code and silently unreachable from the CLI.

DIRECTION IS PER-QUANTITY and lives in the caller's `final`, not in the name:
`alpha_phase` shrinks 1 -> 0 (the RSI window closes toward frame 0), `alpha_goal`
grows 0 -> 1 (the task-domain goal fraction). Read the call site.
"""

from __future__ import annotations

__all__ = ["anneal_alpha"]


def anneal_alpha(
    k: int, start: int, end: int, init: float, final: float
) -> float:
    """Linear ramp `init` -> `final` over [start, end] policy updates.

    An EMPTY window (`end <= start`, the default) holds `init` forever: every
    annealer off by default, and `alpha_<what>_init` alone is a fixed mixture.
    """
    if end <= start:
        return init
    t = min(max((k - start) / (end - start), 0.0), 1.0)
    return init + (final - init) * t
