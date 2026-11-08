"""Task-owned evaluation semantics passed down to a task-blind runner."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Literal


def _noop(*args) -> None:
    pass


@dataclass(frozen=True)
class BenchmarkSpec:
    prepare: Callable[[Any, bool], None]
    configure_batch: Callable[[Any, list[dict]], None] = _noop
    bind_batch: Callable[[Any, list[dict]], None] = _noop
    command_name: str = "motion"
    horizon: Literal["clip", "episode"] = "clip"
    completion_term: str = "exceeded_motion"
    tracking_quality: bool = True


SPECS: dict[str, BenchmarkSpec] = {}


def load_spec(task_id: str) -> BenchmarkSpec:
    try:
        return SPECS[task_id]
    except KeyError as exc:
        raise ValueError(f"Task {task_id!r} has no motion-repertoire benchmark contract") from exc
