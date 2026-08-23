"""ORCS discovery is a contract: never raises, and says why a task is missing.

The one property every downstream consumer depends on. Needs no GPU; needs no
data either — a checkout with nothing staged must pass this, with SKIP_REASON
carrying the explanation instead of a traceback.
"""

from __future__ import annotations

import pytest

orcs = pytest.importorskip("orcs")

from mjlab.tasks.registry import list_tasks  # noqa: E402

TASK_MODULES = (orcs.tasks.dodge, orcs.tasks.uolm, orcs.tasks.perloco)


def test_import_is_silent_about_failure():
    """Whatever is or is not staged, importing worked. (Reaching this line at
    all is the assertion; the import happened at module load.)"""
    assert orcs.__doc__


@pytest.mark.parametrize("mod", TASK_MODULES, ids=lambda m: m.__name__)
def test_skip_reason_is_a_mapping(mod):
    """A dict, not a last-failure string — one unstaged dataset must not hide
    the other three tasks' reasons."""
    assert isinstance(mod.SKIP_REASON, dict)
    assert all(isinstance(k, str) and isinstance(v, str)
               for k, v in mod.SKIP_REASON.items())


@pytest.mark.parametrize("mod", TASK_MODULES, ids=lambda m: m.__name__)
def test_registered_and_skipped_partition_the_table(mod):
    """Every declared task either registered or has a reason. No third state."""
    declared = {row[0] for row in mod._TASKS}
    registered = set(list_tasks())
    for task_id in declared:
        assert (task_id in registered) ^ (task_id in mod.SKIP_REASON), task_id


def test_top_level_skip_reasons_aggregate_every_task_module():
    expected = {
        task_id: reason
        for module in TASK_MODULES
        for task_id, reason in module.SKIP_REASON.items()
    }
    assert orcs.SKIP_REASON == expected


def test_missing_optional_data_skips_only_its_task_row(monkeypatch):
    from orcs.core import registry

    registered: list[str] = []
    monkeypatch.setattr(
        registry,
        "register_mjlab_task",
        lambda **kwargs: registered.append(kwargs["task_id"]),
    )

    def missing(*, play: bool = False):
        del play
        raise FileNotFoundError("optional reconstructed motions are absent")

    def ready(*, play: bool = False):
        return {"play": play}

    skipped = registry.register_all(
        (
            ("Orcs-Missing", missing, lambda: object()),
            ("Orcs-Ready", ready, lambda: object()),
        )
    )

    assert registered == ["Orcs-Ready"]
    assert skipped == {
        "Orcs-Missing": (
            "FileNotFoundError: optional reconstructed motions are absent"
        )
    }


def test_agents_live_only_in_core():
    """docs/ethos.md: a task picks an agent, it never declares PPO. A second
    definition of PPO is the drift this repo keeps re-litigating."""
    from pathlib import Path
    tasks = Path(orcs.__file__).parent / "tasks"
    assert not list(tasks.rglob("rl_cfg.py"))
