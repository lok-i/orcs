#!/usr/bin/env bash
# Fetch + stage the PerLoco (terrain, motion) datasets. Optional: no other task
# needs it, and `import orcs` degrades to a skipped registration without it.
#
#   1. sparse-clone the source datasets    -> $ORCS_DATA_ROOT/<dataset>
#   2. shallow-clone the GRAIL code repo   -> $ORCS_DEPS_ROOT/GRAIL
#   3. WAIT for you to drop the SMPL-X body models in (licensed, not fetchable)
#   4. stage exactly what the rosters ask for -> $ORCS_DATA_ROOT/terrain_motions
#
# Idempotent + resumable: every step no-ops when its output is already there,
# so a failed download is a re-run, not a restart.
#
# Requires: git-lfs, and the active env from `sync_dependencies.sh` (staging
# imports orcs, mjlab, pxr/USD for GRAIL geometry, smplx+joblib for --smpl).

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
STAGE="$REPO_ROOT/scripts/stage_terrain_motions.py"

SOURCES="omni,grail"
GRAIL_CATEGORIES="curb"     # stair1/stair2 land here when a roster wants them
WITH_VIDEO=0                # curb: 1.1 GB without, 14 GB with, and nothing reads it
WITH_SMPL=1                 # the -Smpl command space; needs the SMPL-X models
SKIP_CLONE=0
SKIP_STAGE=0
ASSUME_YES=0

usage() {
    sed -n '2,14p' "$0" | sed 's/^# \?//'
    cat <<'EOF'

  --sources omni,grail      which datasets           (default: both)
  --grail-categories curb   GRAIL terrain categories (default: curb)
  --with-video              keep GRAIL's video/ dir  (+13 GB, unread)
  --no-smpl                 skip SMPL-X; -Smpl task will not register
  --skip-clone              stage what is already on disk
  --skip-stage              fetch only
  --yes                     non-interactive (fail instead of prompting)
EOF
    exit 0
}

while [ $# -gt 0 ]; do
    case "$1" in
        --sources)           SOURCES="$2"; shift 2 ;;
        --grail-categories)  GRAIL_CATEGORIES="$2"; shift 2 ;;
        --with-video)        WITH_VIDEO=1; shift ;;
        --no-smpl)           WITH_SMPL=0; shift ;;
        --skip-clone)        SKIP_CLONE=1; shift ;;
        --skip-stage)        SKIP_STAGE=1; shift ;;
        --yes|-y)            ASSUME_YES=1; shift ;;
        -h|--help)           usage ;;
        *) echo "[ERROR] unknown flag: $1"; usage ;;
    esac
done

has() { [[ ",$SOURCES," == *",$1,"* ]]; }

# Roots come from orcs.core.paths, never from path math here — so ORCS_DATA_ROOT
# / ORCS_DEPS_ROOT move the download exactly like they move the runtime lookup.
# `tail -1`: importing orcs pulls mjlab, which chatters on stdout.
read -r DATA_ROOT DEPS_ROOT SMPLX_DIR ROSTER_DIR < <(python -c '
from orcs.core.paths import DATA_ROOT, DEPS_ROOT
from orcs.tasks.perloco.roster import ROSTER_DIR
from orcs.tasks.perloco.sources.smplx_fk import smplx_dir
print(DATA_ROOT, DEPS_ROOT, smplx_dir(), ROSTER_DIR)' | tail -1)

echo "[ENV] data $DATA_ROOT"
echo "[ENV] deps $DEPS_ROOT"

command -v git-lfs &>/dev/null || { echo "[ERROR] git-lfs not on PATH"; exit 1; }

# The ROSTER is the single source of "which tiles exist" — it drives the LFS
# scope AND the staging flags, so a roster edit changes both and neither can
# drift. tomllib only, no orcs import, so stdout carries the answer alone.
roster_read() {
    python -c '
import sys, tomllib
spec = tomllib.loads(open(sys.argv[1], "rb").read().decode())
v = spec.get(sys.argv[2], "*")
print("" if v == "*" else " ".join(map(str, v)))' "$ROSTER_DIR/$1.toml" "$2"
}
roster_families() { roster_read "$1" families; }

# ── 1. datasets ─────────────────────────────────────────────────────────────
# TWO independent filters, and missing either one downloads the whole dataset:
#
#   sparse-checkout   which POINTERS land in the working tree
#   lfs pull -I       which BLOBS get fetched
#
# **`git lfs pull` does not read sparse-checkout.** `git lfs fetch` resolves
# every LFS object reachable from the ref, so an unscoped pull on GRAIL fetches
# 106k objects / 12+ GB no matter how narrow the working tree is. That is not a
# tuning miss; it is the difference between 38 MB and a wasted afternoon.
#
# `--no-cone` on `set`, always: cone mode can only include whole subtrees, and
# the point here is taking curb/ WITHOUT its 13 GB of video.
#
#   clone_sparse <url> <dest> <lfs-include-csv> <sparse patterns...>
# An empty <lfs-include-csv> means "every blob in the sparse tree".
clone_sparse() {
    local url="$1" dest="$2" lfs_inc="$3"; shift 3
    echo
    echo "=== $(basename "$dest") ==="
    if [ ! -d "$dest/.git" ]; then
        echo "[ CLONE  ] $url"
        GIT_LFS_SKIP_SMUDGE=1 git clone --no-checkout "$url" "$dest"
    fi
    GIT_LFS_SKIP_SMUDGE=1 git -C "$dest" sparse-checkout set --no-cone -- "$@"
    GIT_LFS_SKIP_SMUDGE=1 git -C "$dest" checkout
    echo "[ SPARSE ] $(git -C "$dest" ls-files | wc -l) files in tree"

    local n
    if [ -n "$lfs_inc" ]; then
        n=$(git -C "$dest" lfs ls-files -I "$lfs_inc" | wc -l)
        echo "[ LFS    ] $n blobs (scoped); $(git -C "$dest" lfs ls-files | wc -l) exist"
        [ "$n" -gt 0 ] || { echo "[ERROR] include matched 0 blobs: $lfs_inc"; exit 1; }
        git -C "$dest" lfs pull --include="$lfs_inc"
    else
        echo "[ LFS    ] $(git -C "$dest" lfs ls-files | wc -l) blobs"
        git -C "$dest" lfs pull
    fi
    echo "[ OK     ] $(du -sh "$dest" | cut -f1)"
}

if [ "$SKIP_CLONE" = 0 ] && has omni; then
    # robot-object*.zip are the manipulation splits — perloco reads neither, and
    # robot-object.zip alone is 273 MB.
    clone_sparse https://huggingface.co/datasets/omniretarget/OmniRetarget_Dataset \
        "$DATA_ROOT/OmniRetarget_Dataset" \
        'robot-terrain.zip,models/**' \
        '/robot-terrain.zip' '/models/' '/visualize.py' '/README.md'
fi

if [ "$SKIP_CLONE" = 0 ] && has grail; then
    # ONLY the three dirs GrailSource opens: robot/ (the retargeted clip),
    # object_usd/ (tile geometry, and NOT lfs — it rides the checkout) and
    # recon/ (the SMPL-X human, --smpl). objects/ and meta/ are lfs and unread.
    patterns=() lfs=()
    IFS=',' read -ra cats <<< "$GRAIL_CATEGORIES"
    for c in "${cats[@]}"; do
        for d in robot object_usd recon; do
            patterns+=("/data/$c/$d/")
        done
        if [ "$WITH_VIDEO" = 1 ]; then
            patterns+=("/data/$c/video/")
            lfs+=("data/$c/video/**")
        fi
        # Blobs are scoped to the ROSTERED families — a curb category ships
        # 1769 takes and the roster names 8 of them. 126 files, not 8846.
        for f in $(roster_families grail); do
            lfs+=("data/$c/robot/*__${f}__*" "data/$c/recon/*__${f}__*")
        done
    done
    clone_sparse https://huggingface.co/datasets/nvidia/PhysicalAI-Robotics-Locomanipulation-GRAIL \
        "$DATA_ROOT/PhysicalAI-Robotics-Locomanipulation-GRAIL" \
        "$(IFS=,; echo "${lfs[*]}")" \
        '/README.md' "${patterns[@]}"
fi

# ── 2. GRAIL code ───────────────────────────────────────────────────────────
# Reference only (retargeter + its vendored SONIC), never imported by orcs. It
# is also where SMPL-X conventionally lands, which is why it comes before §3.
if [ "$SKIP_CLONE" = 0 ] && has grail; then
    echo
    echo "=== GRAIL (code) ==="
    if [ -d "$DEPS_ROOT/GRAIL/.git" ]; then
        echo "[ HEAD OK] already cloned"
    else
        # No submodules: they are the recon/generation stack (MoGe, FoundationPose,
        # Hunyuan3D, ...), gigabytes that staging never touches.
        git clone --depth 1 https://github.com/NVlabs/GRAIL "$DEPS_ROOT/GRAIL"
    fi
fi

# ── 3. SMPL-X body models — the manual gate ─────────────────────────────────
if [ "$WITH_SMPL" = 1 ]; then
    echo
    echo "=== SMPL-X body models ==="
    mkdir -p "$SMPLX_DIR/smplx"
    while [ ! -f "$SMPLX_DIR/smplx/SMPLX_NEUTRAL.npz" ]; do
        cat <<EOF
[ MANUAL ] SMPL-X is licensed — download it yourself, we cannot fetch it.

  1. register + download "SMPL-X v1.1 (NPZ+PKL)" from https://smpl-x.is.tue.mpg.de
  2. unzip so that this path exists:

       $SMPLX_DIR/smplx/SMPLX_NEUTRAL.npz

  (ORCS_SMPLX_DIR overrides that location for both this script and staging.)
EOF
        if [ "$ASSUME_YES" = 1 ]; then
            echo "[ERROR] --yes given and SMPL-X is absent; re-run without --yes, or --no-smpl"
            exit 1
        fi
        read -r -p "  press ENTER once it is in place (or Ctrl-C to abort) "
    done
    echo "[ OK     ] $SMPLX_DIR/smplx"
fi

# ── 4. stage ────────────────────────────────────────────────────────────────
# Families/levels come from the ROSTER, not from this script: the roster is what
# the env builds its grid from, so a hardcoded list here would be a second
# source of "which tiles exist" and would drift the day one is edited.
if [ "$SKIP_STAGE" = 0 ]; then
    for s in omni grail; do
        has "$s" || continue
        echo
        echo "=== stage $s ==="
        args=()
        for key in families levels; do
            read -r -a vals <<< "$(roster_read "$s" "$key")"
            [ ${#vals[@]} -gt 0 ] && args+=("--$key" "${vals[@]}")
        done
        [ "$s" = grail ] && [ "$WITH_SMPL" = 1 ] && args+=(--smpl)
        echo "[ STAGE  ] $STAGE --source $s ${args[*]}"
        python "$STAGE" --source "$s" "${args[@]}"
    done
fi

# ── 5. verify ───────────────────────────────────────────────────────────────
echo
echo "=== registered ==="
python -c '
import orcs
from mjlab.tasks.registry import list_tasks
print("\n".join(t for t in list_tasks() if t.startswith("Orcs-PerLoco")) or "  (none)")
print("skip reason:", orcs.tasks.perloco.SKIP_REASON)'
