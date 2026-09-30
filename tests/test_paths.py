"""Filesystem ownership contracts for checkout and installed-package modes."""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
PATHS_MODULE = REPO / "src/orcs/core/paths.py"


def _copy_module(path: Path) -> Path:
    path.parent.mkdir(parents=True)
    shutil.copyfile(PATHS_MODULE, path)
    return path


def _mark_repo(root: Path) -> None:
    root.mkdir(parents=True, exist_ok=True)
    for marker in ("pyproject.toml", "deps.lock"):
        (root / marker).touch()


def _load_roots(module: Path, **overrides: str) -> dict[str, str]:
    env = os.environ.copy()
    for name in (
        "ORCS_ROOT",
        "ORCS_DATA_ROOT",
        "ORCS_DEPS_ROOT",
        "ORCS_ASSETS_SOURCE",
        "XDG_DATA_HOME",
    ):
        env.pop(name, None)
    env.update(overrides)
    code = (
        "import json, runpy; "
        f"m = runpy.run_path({str(module)!r}); "
        "print(json.dumps({k: str(m[k]) for k in "
        "('REPO_ROOT', 'DATA_ROOT', 'DEPS_ROOT')}))"
    )
    result = subprocess.run(
        [sys.executable, "-c", code],
        check=True,
        capture_output=True,
        text=True,
        env=env,
    )
    return json.loads(result.stdout)


def _load_smplx_dir(module: Path, **overrides: str) -> str:
    env = os.environ.copy()
    for name in (
        "ORCS_ROOT",
        "ORCS_DATA_ROOT",
        "ORCS_DEPS_ROOT",
        "ORCS_SMPLX_DIR",
        "XDG_DATA_HOME",
    ):
        env.pop(name, None)
    env.update(overrides)
    code = (
        "import runpy; "
        f"m = runpy.run_path({str(module)!r}); "
        "print(m['smplx_dir']())"
    )
    result = subprocess.run(
        [sys.executable, "-c", code],
        check=True,
        capture_output=True,
        text=True,
        env=env,
    )
    return result.stdout.strip()


def test_standalone_checkout_owns_its_runtime_data(tmp_path: Path) -> None:
    checkout = tmp_path / "orcs"
    _mark_repo(checkout)
    module = _copy_module(checkout / "src/orcs/core/paths.py")

    roots = _load_roots(module)

    assert roots == {
        "REPO_ROOT": str(checkout),
        "DATA_ROOT": str(checkout / "data"),
        "DEPS_ROOT": str(checkout / "dependencies"),
    }


def test_vendored_checkout_uses_host_data_and_dependencies(tmp_path: Path) -> None:
    host = tmp_path / "host"
    checkout = host / "dependencies/orcs"
    _mark_repo(host)
    _mark_repo(checkout)
    (host / "data").mkdir()
    module = _copy_module(checkout / "src/orcs/core/paths.py")

    roots = _load_roots(module)

    assert roots == {
        "REPO_ROOT": str(checkout),
        "DATA_ROOT": str(host / "data"),
        "DEPS_ROOT": str(host / "dependencies"),
    }


def test_noneditable_install_uses_per_user_runtime_root(tmp_path: Path) -> None:
    module = _copy_module(tmp_path / "site-packages/orcs/core/paths.py")
    xdg_data = tmp_path / "xdg-data"

    roots = _load_roots(module, XDG_DATA_HOME=str(xdg_data))

    runtime = xdg_data / "orcs"
    assert roots == {
        "REPO_ROOT": str(runtime),
        "DATA_ROOT": str(runtime / "data"),
        "DEPS_ROOT": str(runtime / "dependencies"),
    }


def test_orcs_root_overrides_missing_checkout(tmp_path: Path) -> None:
    module = _copy_module(tmp_path / "site-packages/orcs/core/paths.py")
    runtime = tmp_path / "runtime"

    roots = _load_roots(module, ORCS_ROOT=str(runtime))

    assert roots == {
        "REPO_ROOT": str(runtime),
        "DATA_ROOT": str(runtime / "data"),
        "DEPS_ROOT": str(runtime / "dependencies"),
    }


def test_body_models_default_outside_grail_checkout(tmp_path: Path) -> None:
    checkout = tmp_path / "orcs"
    _mark_repo(checkout)
    module = _copy_module(checkout / "src/orcs/core/paths.py")

    assert _load_smplx_dir(module) == str(checkout / "dependencies/body_models")


def test_body_models_honor_explicit_override(tmp_path: Path) -> None:
    checkout = tmp_path / "orcs"
    _mark_repo(checkout)
    module = _copy_module(checkout / "src/orcs/core/paths.py")
    models = tmp_path / "licensed-models"

    assert _load_smplx_dir(module, ORCS_SMPLX_DIR=str(models)) == str(models)


def test_populated_legacy_body_models_remain_readable(tmp_path: Path) -> None:
    checkout = tmp_path / "orcs"
    _mark_repo(checkout)
    module = _copy_module(checkout / "src/orcs/core/paths.py")
    legacy = (
        checkout
        / "dependencies/GRAIL/imports/GEM-SMPL/inputs/checkpoints/body_models"
    )
    (legacy / "smplx").mkdir(parents=True)
    (legacy / "smplx/SMPLX_NEUTRAL.npz").touch()

    assert _load_smplx_dir(module) == str(legacy)
