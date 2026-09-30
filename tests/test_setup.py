"""Static setup-script contracts; no network, environment mutation, or GPU."""

from __future__ import annotations

import subprocess
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
SETUP = REPO / "scripts/setup"


def test_setup_scripts_parse() -> None:
    for name in (
        "_common.sh",
        "sync_deps.sh",
        "sync_data.sh",
        "sync_dependencies.sh",
        "perceptive_locomotion.sh",
    ):
        subprocess.run(["bash", "-n", str(SETUP / name)], check=True)


def test_split_sync_help_is_side_effect_free() -> None:
    deps = subprocess.run(
        ["bash", str(SETUP / "sync_deps.sh"), "--help"],
        check=True,
        capture_output=True,
        text=True,
    )
    data = subprocess.run(
        ["bash", str(SETUP / "sync_data.sh"), "--help"],
        check=True,
        capture_output=True,
        text=True,
    )

    assert "--no-smpl" in deps.stdout
    assert "default: all" in data.stdout
    assert "--no-smpl" in data.stdout


def test_no_smpl_lock_selection_omits_reconstructed_motions() -> None:
    command = (
        f". {SETUP / '_common.sh'}; "
        "lock_rows data/ reconstructed_motions"
    )
    result = subprocess.run(
        ["bash", "-c", command],
        check=True,
        capture_output=True,
        text=True,
    )

    assert "retargeted_motions|" in result.stdout
    assert result.stdout.strip().endswith("|1")
    assert "reconstructed_motions|" not in result.stdout


def test_no_smpl_enables_sparse_sync_but_full_mode_can_disable_it() -> None:
    data = (SETUP / "sync_data.sh").read_text()
    common = (SETUP / "_common.sh").read_text()

    assert "lean_sparse=$((1 - WITH_SMPL))" in data
    assert 'sync_rows "$rows" "$lean_sparse"' in data
    assert "sparse-checkout set --no-cone" in common
    assert "sparse-checkout disable" in common
    assert "lfs pull" in common
    assert '-I "$include"' in common


def test_setup_is_uv_only() -> None:
    common = (SETUP / "_common.sh").read_text()
    assert "PIP_CMD=(uv pip)" in common
    assert 'local venv="$REPO_ROOT/.venv"' in common
    assert '${VIRTUAL_ENV:-$REPO_ROOT/.venv}' not in common
    assert 'uv venv --python' in common
    assert 'source .venv/bin/activate' in common
    assert 'uv venv --python "$(cat "$REPO_ROOT/.python-version")"' not in common
    assert "DEPS_PIP_CMD" not in common
    assert "CONDA_PREFIX" not in common
    assert "pip_subpath" not in common
    assert "unzip_csv" not in common


def test_legacy_combined_sync_delegates_to_split_scripts() -> None:
    body = (SETUP / "sync_dependencies.sh").read_text()
    assert 'sync_deps.sh" "$@"' in body
    assert 'sync_data.sh" inhouse "$@"' in body


def test_perloco_setup_keeps_only_body_models_not_grail_code() -> None:
    body = (SETUP / "perceptive_locomotion.sh").read_text()
    assert "github.com/NVlabs/GRAIL" not in body
    assert 'for model in NEUTRAL MALE FEMALE' in body
    assert 'SMPLX_${model}.npz' in body
