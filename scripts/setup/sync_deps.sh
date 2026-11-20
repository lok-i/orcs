#!/usr/bin/env bash
# Sync ORCS code: the environment, ORCS itself, and deps.lock code rows.
#
#   sync_deps.sh              full contributor install (default)
#   sync_deps.sh --no-smpl    omit the SMPL Python tooling
#   sync_deps.sh --optix      include optional OptiX bindings (compatible driver required)
#   sync_deps.sh --check      verify pins, patches and installs; no downloads
#
# Creates Python 3.12 ./.venv; rejects an unrelated active consumer environment.
# Data belongs to sync_data.sh.

set -euo pipefail
. "$(dirname "$0")/_common.sh"

CHECK_ONLY=0
WITH_SMPL=1 WITH_OPTIX=0
for arg in "$@"; do
    case "$arg" in
        --check) CHECK_ONLY=1 ;;
        --no-smpl) WITH_SMPL=0 ;;
        --optix) WITH_OPTIX=1 ;;
        -h|--help) sed -n '2,11p' "$0"; exit 0 ;;
        *) echo "[ERROR] unknown argument: $arg (expected --check | --no-smpl | --optix)"; exit 2 ;;
    esac
done

rows=$(lock_rows dependencies/)
if [ "$CHECK_ONLY" = 0 ]; then
    use_venv

    extras="dev,perloco"
    [ "$WITH_SMPL" = 0 ] || extras="$extras,smpl"
    [ "$WITH_OPTIX" = 0 ] || extras="$extras,optix"
    sync_rows "$rows"
    install_code "$extras" "$rows"

    echo
    echo "=== generate: assets ==="
    python -m assets.cli generate --quiet
fi

verify_rows "$rows" dependencies/

"$REPO_ROOT/.venv/bin/python" "$LOCK_HELPER" check-patches "$REPO_ROOT"
"$REPO_ROOT/.venv/bin/python" "$LOCK_HELPER" check-install "$REPO_ROOT"
uv pip check --python "$REPO_ROOT/.venv/bin/python"
