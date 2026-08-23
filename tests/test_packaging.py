"""What a NON-editable install must still contain. Runs without a GPU or data.

These exist because an editable dev loop cannot see packaging bugs: it imports
from the source tree, so a file missing from the wheel and a `requires-python`
that undersells the real floor both look fine locally and break the first
`pip install git+...`. Both had shipped.
"""

from __future__ import annotations

import configparser
import os
import subprocess
import sys
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


def test_package_import_is_lightweight():
    """The library root must not recursively start MJLab discovery."""
    code = "import sys; import orcs; assert 'mjlab' not in sys.modules"
    subprocess.run(
        [sys.executable, "-c", code],
        check=True,
        capture_output=True,
        text=True,
    )


def test_mjlab_entry_point_targets_registration_module():
    assert PYPROJECT["project"]["entry-points"]["mjlab.tasks"] == {
        "orcs": "orcs.registration"
    }


def test_cli_help_does_not_break_sibling_discovery(tmp_path):
    """A direct ORCS CLI import used to expose ORCS half-initialized to Vibe."""
    env = os.environ.copy()
    env["MPLCONFIGDIR"] = str(tmp_path / "matplotlib")
    result = subprocess.run(
        [sys.executable, "-m", "orcs.cli.pseudo_retarget", "--help"],
        check=True,
        capture_output=True,
        text=True,
        env=env,
    )
    assert "Failed to load task package" not in result.stderr


def test_console_scripts_ship_in_wheel(wheel):
    """The artifact owns its script metadata; the active dev env is irrelevant."""
    candidates = [
        name for name in wheel.namelist()
        if name.endswith(".dist-info/entry_points.txt")
    ]
    assert len(candidates) == 1, candidates
    parser = configparser.ConfigParser()
    parser.read_string(wheel.read(candidates[0]).decode())
    assert dict(parser["console_scripts"]) == PYPROJECT["project"]["scripts"]


def test_wheel_imports_without_checkout_markers(wheel, tmp_path):
    """A wheel has neither pyproject.toml nor deps.lock beside its package."""
    site = tmp_path / "site-packages"
    wheel.extractall(site)
    runtime = tmp_path / "runtime"
    env = os.environ.copy()
    env.update(
        {
            "MPLCONFIGDIR": str(tmp_path / "matplotlib"),
            "ORCS_SKIP_DEP_CHECK": "1",
            "PYTHONPATH": os.pathsep.join(
                value
                for value in (str(site), env.get("PYTHONPATH"))
                if value
            ),
            "XDG_DATA_HOME": str(runtime),
        }
    )
    code = (
        "from pathlib import Path; import mjlab, orcs; "
        "from orcs.core.paths import REPO_ROOT; "
        f"assert Path(orcs.__file__).is_relative_to(Path({str(site)!r})); "
        f"assert REPO_ROOT == Path({str(runtime / 'orcs')!r})"
    )
    subprocess.run(
        [sys.executable, "-c", code],
        check=True,
        capture_output=True,
        text=True,
        env=env,
    )
