#!/usr/bin/env bash
# Sync ORCS code: the environment, ORCS itself, and deps.lock code rows.
#
#   sync_deps.sh              full contributor install (default)
#   sync_deps.sh --no-smpl    omit the SMPL Python tooling
#   sync_deps.sh --check      verify code checkout SHAs only; no network/pip
#
# Create + activate ./.venv first; sync never mutates a consumer environment.
# Data belongs to sync_data.sh.

set -euo pipefail
. "$(dirname "$0")/_common.sh"

CHECK_ONLY=0
WITH_SMPL=1
for arg in "$@"; do
    case "$arg" in
        --check) CHECK_ONLY=1 ;;
        --no-smpl) WITH_SMPL=0 ;;
        -h|--help) sed -n '2,10p' "$0"; exit 0 ;;
        *) echo "[ERROR] unknown argument: $arg (expected --check | --no-smpl)"; exit 2 ;;
    esac
done

rows=$(lock_rows dependencies/)
if [ "$CHECK_ONLY" = 0 ]; then
    use_venv

    extras="dev,perloco"
    [ "$WITH_SMPL" = 0 ] || extras="$extras,smpl"
    echo
    echo "=== orcs (self: $extras) ==="
    "${PIP_CMD[@]}" install -e "$REPO_ROOT[$extras]"

    # Always last: mjlab's PyPI rsl-rl pin must not replace the validated fork.
    sync_rows "$rows"

    echo
    echo "=== generate: assets ==="
    python -m assets.cli generate --quiet
fi

verify_rows "$rows" dependencies/
