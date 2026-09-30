#!/usr/bin/env bash
# Sync data and generated inputs. Run sync_deps.sh first.
#
#   sync_data.sh [MODE...]       default: all
#     inhouse   locked retargeted/reconstructed motions + nominal stand
#     omre      OmniRetarget terrain motions
#     grail     GRAIL terrain motions
#     all       all of the above
#   --no-smpl   sparse native motion roster; skip reconstructed/GRAIL SMPL data
#   --check     verify selected lock rows only; no network or generation

set -euo pipefail
. "$(dirname "$0")/_common.sh"

CHECK_ONLY=0
WITH_SMPL=1
modes=()
for arg in "$@"; do
    case "$arg" in
        --check) CHECK_ONLY=1 ;;
        --no-smpl) WITH_SMPL=0 ;;
        all|inhouse|omre|grail) modes+=("$arg") ;;
        -h|--help) sed -n '2,10p' "$0"; exit 0 ;;
        *) echo "[ERROR] unknown argument: $arg (expected all|inhouse|omre|grail|--no-smpl|--check)"; exit 2 ;;
    esac
done
[ ${#modes[@]} -gt 0 ] || modes=(all)
has() { [[ " ${modes[*]} " == *" $1 "* || " ${modes[*]} " == *" all "* ]]; }

rows=""
if has inhouse; then
    if [ "$WITH_SMPL" = 1 ]; then
        rows=$(lock_rows data/)
    else
        rows=$(lock_rows data/ reconstructed_motions)
    fi
fi

if [ "$CHECK_ONLY" = 1 ]; then
    [ -z "$rows" ] || verify_rows "$rows" data/
    exit
fi

use_venv

if has inhouse; then
    lean_sparse=$((1 - WITH_SMPL))
    sync_rows "$rows" "$lean_sparse"

    echo
    echo "=== generate: nominal motion ==="
    nominal_root="${ORCS_DATA_ROOT:-$REPO_ROOT/data}/nominal_motions"
    nominal="$nominal_root/nominal/stand/sample1/motion.npz"
    if [ -f "$nominal" ]; then
        echo "[ NOM OK ] already present"
    else
        orcs-make-nominal --out "$nominal_root" --device cpu
    fi
fi

sources=()
has omre && sources+=(omni)
has grail && sources+=(grail)
if [ ${#sources[@]} -gt 0 ]; then
    args=(--sources "$(IFS=,; echo "${sources[*]}")")
    [ "$WITH_SMPL" = 1 ] || args+=(--no-smpl)
    bash "$SETUP_DIR/perceptive_locomotion.sh" "${args[@]}"
fi

[ -z "$rows" ] || verify_rows "$rows" data/
