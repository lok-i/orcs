"""Task registration that survives an incomplete checkout.

`import orcs` must never raise. A consumer that vendors orcs for ONE task
should not lose it because another task's dataset is unstaged — so every row
registers independently and a failure becomes an entry in the returned dict.

    SKIP_REASON = register_all(_TASKS)   # {} when everything registered

A missing task is the signal; `SKIP_REASON[task_id]` is the explanation.
"""

from __future__ import annotations

from typing import Callable, Iterable

from mjlab.tasks.registry import register_mjlab_task

__all__ = ["TaskRow", "register_all", "MISSING_DATA"]

MISSING_DATA = (FileNotFoundError, NotADirectoryError, OSError)
"""What "the data is not here yet" looks like. Anything else is a real bug and
propagates — a typo in a cfg must not masquerade as a missing dataset."""

TaskRow = tuple[str, Callable[..., object], Callable[[], object]]
"""(task_id, env_cfg factory taking `play=`, rl_cfg factory taking nothing)."""


def register_all(rows: Iterable[TaskRow]) -> dict[str, str]:
    """Register each row; return {task_id: reason} for the ones that could not."""
    skipped: dict[str, str] = {}
    for task_id, env_cfg, rl_cfg in rows:
        try:
            register_mjlab_task(
                task_id=task_id,
                env_cfg=env_cfg(),
                play_env_cfg=env_cfg(play=True),
                rl_cfg=rl_cfg(),
            )
        except MISSING_DATA as e:
            skipped[task_id] = f"{type(e).__name__}: {e}"
    return skipped
