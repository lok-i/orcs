"""What a NON-editable install must still contain. Runs without a GPU or data.

These exist because an editable dev loop cannot see packaging bugs: it imports
from the source tree, so a file missing from the wheel and a `requires-python`
that undersells the real floor both look fine locally and break the first
`pip install git+...`. Both had shipped.
"""

from __future__ import annotations

import subprocess
import sys
import sysconfig
import tomllib
import zipfile
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
PYPROJECT = tomllib.loads((REPO / "pyproject.toml").read_text())


@pytest.fixture(scope="module")
def wheel(tmp_path_factory) -> zipfile.ZipFile:
    out = tmp_path_factory.mktemp("wheel")
    # --no-cache-dir is load-bearing: pip happily returns a CACHED wheel built
    # before the packaging bug was introduced, so the test passes on a broken
    # tree. (Verified by breaking it on purpose.)
    subprocess.run(
        [sys.executable, "-m", "pip", "wheel", "--no-deps", "--no-cache-dir",
         "--no-build-isolation", "-w", str(out), str(REPO)],
        check=True, capture_output=True,
    )
    built = list(out.glob("orcs-*.whl"))
    assert len(built) == 1, f"expected one wheel, got {built}"
    return zipfile.ZipFile(built[0])


def test_rosters_ship(wheel):
    """setuptools drops non-.py by default; PerLoco cannot build a grid without
    these, and the failure is a SILENT skipped registration."""
    rosters = sorted(REPO.glob("src/orcs/tasks/perloco/rosters/*.toml"))
    assert rosters, "no rosters in the source tree — this test is checking nothing"
    names = set(wheel.namelist())
    for r in rosters:
        assert f"orcs/tasks/perloco/rosters/{r.name}" in names


def test_cli_ships(wheel):
    """`scripts/` is not packaged, so the data pipeline has to live in the
    package or a pip-installed orcs can consume data it cannot produce."""
    names = set(wheel.namelist())
    for entry in PYPROJECT["project"]["scripts"].values():
        module = entry.split(":")[0].replace(".", "/") + ".py"
        assert module in names, f"{module} missing (entry point {entry})"


def test_scripts_are_thin_wrappers():
    """Each `scripts/*.py` must delegate to its `orcs.cli` twin — a second copy
    of the logic is how the two drift."""
    for path in sorted((REPO / "scripts").glob("*.py")):
        body = path.read_text()
        assert f"from orcs.cli.{path.stem} import main" in body, path
        assert len(body.splitlines()) < 15, f"{path} grew a body"


def test_requires_python_covers_stdlib_used():
    """`tomllib` is 3.11+; the floor said 3.10 and `import orcs` died there."""
    floor = PYPROJECT["project"]["requires-python"]
    assert floor.startswith(">=3.11"), floor


def test_entry_points_resolve():
    """Declared console scripts must import — a renamed module is otherwise
    only discovered by a user typing the command."""
    for entry in PYPROJECT["project"]["scripts"].values():
        module, func = entry.split(":")
        subprocess.run(
            [sys.executable, "-c", f"import {module}; assert callable({module}.{func})"],
            check=True, capture_output=True,
        )


def test_console_scripts_installed():
    """Present in the active env (i.e. `pip install -e .` was re-run after the
    entry points were added)."""
    bindir = Path(sysconfig.get_path("scripts"))
    for name in PYPROJECT["project"]["scripts"]:
        assert (bindir / name).exists(), f"{name} not installed — re-run pip install -e ."
