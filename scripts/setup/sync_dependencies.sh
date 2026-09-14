#!/usr/bin/env bash
# Sync custom dependencies + data declared in deps.lock.
#
# For each entry:
#   1. shallow-fetch the pinned SHA into <path> (init+fetch-by-sha, branch-agnostic)
#   2. if `lfs`: git lfs pull
#   3. if `pip_install`: install -e <path> into the active env
#      (auto-picks `pip` for conda, `uv pip` for uv; override: DEPS_PIP_CMD)
#   4. generate machine-local object XMLs through the installed assets package
#
# Idempotent: skips fetch when HEAD already matches the pinned SHA.
# Assumes: git-lfs binary is on PATH; an active conda/venv if any entry has pip_install=true.

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
LOCK_FILE="$REPO_ROOT/deps.lock"

# Pick the package installer for the *active* python env:
#   - conda env active        -> `pip install`        (conda machines)
#   - uv-managed venv active  -> `uv pip install`     (uv machines)
# Override with DEPS_PIP_CMD="uv pip" | "pip" to force a choice.
detect_pip_cmd() {
    if [ -n "${DEPS_PIP_CMD:-}" ]; then
        echo "$DEPS_PIP_CMD"; return
    fi
    # conda wins when its env is active (uv-on-PATH shouldn't hijack it)
    if [ -n "${CONDA_PREFIX:-}" ] || [ -n "${CONDA_DEFAULT_ENV:-}" ]; then
        echo "pip"; return
    fi
    # uv-managed venv: uv present + a non-conda venv active
    if command -v uv &>/dev/null && [ -n "${VIRTUAL_ENV:-}" ]; then
        echo "uv pip"; return
    fi
    # fallbacks: prefer uv if installed, else plain pip
    if command -v uv &>/dev/null; then echo "uv pip"; else echo "pip"; fi
}
PIP_CMD=$(detect_pip_cmd)
echo "[ENV] python installer: $PIP_CMD  (override via DEPS_PIP_CMD)"

# Parallel LFS transfers. The datasets are ~14k objects averaging ~270 KB, so
# wall-clock is dominated by per-object round-trips, not bandwidth.
LFS_CONCURRENCY="${LFS_CONCURRENCY:-16}"

# Ensure git-lfs is set up for the main repo (covers fresh environments
# where git lfs install was never run globally)
if command -v git-lfs &>/dev/null || git lfs version &>/dev/null 2>&1; then
    git -C "$REPO_ROOT" lfs install --local 2>/dev/null || true
    echo "[LFS] Pulling main repo LFS files..."
    git -C "$REPO_ROOT" lfs pull
else
    echo "[WARN] git-lfs not found — LFS files will remain pointers."
    echo "       Install git-lfs or run: module load git-lfs"
fi

# Run a command with a rotating spinner + elapsed counter (stderr). Forwards exit code.
# Usage: spin "label" <cmd> <args...>
spin() {
    local label="$1"; shift
    "$@" &
    local pid=$!
    trap 'kill -- $pid 2>/dev/null; printf "\r\033[K" >&2; exit 130' INT
    local chars='|/-\' i=0 start=$SECONDS
    while kill -0 "$pid" 2>/dev/null; do
        local e=$((SECONDS - start))
        printf "\r[ %-7s] %s  %02d:%02d  " "$label" "${chars:i++%${#chars}:1}" $((e/60)) $((e%60)) >&2
        sleep 0.15
    done
    wait "$pid"; local rc=$?
    printf "\r\033[K" >&2
    trap - INT
    return $rc
}

if [ ! -f "$LOCK_FILE" ]; then
    echo "[ERROR] deps.lock not found at $LOCK_FILE"
    exit 1
fi

# emit `name|url|sha|path|pip_install|lfs|pip_subpath|unzip|pip_no_deps` per entry
# (unzip is a comma-separated list of zip paths relative to <path>)
# `|` chosen as a non-whitespace separator — IFS=$'\t' collapses consecutive tabs (tab is
# treated as whitespace IFS), which mangles empty fields. `|` doesn't collapse.
entries=$(python3 - "$LOCK_FILE" <<'PYEOF'
import json, sys
with open(sys.argv[1]) as f:
    lock = json.load(f)
for name, info in lock.items():
    print("|".join([
        name,
        info["url"],
        info["sha"],
        info["path"],
        "1" if info.get("pip_install") else "0",
        "1" if info.get("lfs") else "0",
        info.get("pip_subpath", ""),
        ",".join(info.get("unzip", [])),
        "1" if info.get("pip_no_deps") else "0",
    ]))
PYEOF
)

sync_one() {
    local name="$1" url="$2" sha="$3" rel_path="$4" pip_install="$5" lfs="$6" pip_subpath="$7" unzip_csv="$8" pip_no_deps="$9"
    local path="$REPO_ROOT/$rel_path"

    echo
    echo "=== $name @ ${sha:0:12} ==="
    echo "    url:  $url"
    echo "    path: $rel_path"

    if [ ! -d "$path/.git" ]; then
        echo "[ CLONE  ] $name -> $rel_path"
        mkdir -p "$path"
        git -C "$path" init -q
        git -C "$path" remote add origin "$url"
        if [ "$lfs" = "1" ]; then
            git -C "$path" lfs install --local
        fi
    fi

    local current
    current=$(git -C "$path" rev-parse HEAD 2>/dev/null || echo "none")

    if [ "$current" != "$sha" ]; then
        echo "[ FETCH  ] depth=1 $sha"
        git -C "$path" fetch --progress --depth 1 origin "$sha"
        if [ "$lfs" = "1" ]; then
            # Do NOT smudge during checkout: `git-lfs filter-process` materializes
            # blobs strictly one at a time and ignores lfs.concurrenttransfers, so
            # a 14k-object dataset costs 14k sequential round-trips. Write pointers
            # instead (instant), and let the `git lfs pull` below move the bytes
            # concurrently. Same files either way.
            git -C "$path" config lfs.concurrenttransfers "$LFS_CONCURRENCY"
            spin "checkout" env GIT_LFS_SKIP_SMUDGE=1 \
                git -C "$path" checkout -q "$sha"
        else
            git -C "$path" checkout -q "$sha"
        fi
    else
        echo "[ HEAD OK] already at $sha"
    fi

    # Run lfs pull for LFS repos — idempotent and fast when already smudged.
    # Catches cases where initial checkout happened without git-lfs on PATH.
    # Exclude zips we've already unpacked + deleted, else lfs checkout keeps
    # re-materializing the (multi-GB) blob every run because it's "missing".
    if [ "$lfs" = "1" ]; then
        local lfs_exclude=""
        if [ -n "$unzip_csv" ]; then
            IFS=',' read -ra _unpacked <<< "$unzip_csv"
            for z in "${_unpacked[@]}"; do
                [ -f "$path/$z.unpacked" ] && lfs_exclude="${lfs_exclude:+$lfs_exclude,}$z"
            done
        fi
        if [ -n "$lfs_exclude" ]; then
            spin "LFS pull" git -C "$path" lfs pull --exclude="$lfs_exclude"
        else
            spin "LFS pull" git -C "$path" lfs pull
        fi
    fi

    if [ -n "$unzip_csv" ]; then
        IFS=',' read -ra zip_list <<< "$unzip_csv"
        for zip_rel in "${zip_list[@]}"; do
            local zip_file="$path/$zip_rel"
            local marker="$zip_file.unpacked"
            if [ -f "$marker" ]; then
                echo "[ UNZIP  ] already extracted: $zip_rel"
                # reap a stale re-fetched zip (e.g. restored by an older lfs pull)
                if [ -f "$zip_file" ]; then
                    rm -f "$zip_file"
                    echo "[ UNZIP  ] removed stale re-fetched zip: $zip_rel"
                fi
                continue
            fi
            if [ ! -f "$zip_file" ]; then
                echo "[ UNZIP  ] error: zip not found: $zip_rel"
                continue
            fi
            UNZIP_DISABLE_ZIPBOMB_DETECTION=TRUE spin "UNZIP" unzip -o -q "$zip_file" -d "$path"
            touch "$marker"
            rm -f "$zip_file"  # reclaim disk; marker keeps state across reruns
            echo "[ UNZIP  ] $zip_rel -> $rel_path/  (zip removed, marker kept)"
        done
    fi

    if [ "$pip_install" = "1" ]; then
        local install_path="$path"
        local install_rel="$rel_path"
        if [ -n "$pip_subpath" ]; then
            install_path="$path/$pip_subpath"
            install_rel="$rel_path/$pip_subpath"
        fi
        if [ "$pip_no_deps" = "1" ]; then
            echo "[ PIP    ] $PIP_CMD install --no-deps -e $install_rel"
            $PIP_CMD install --no-deps -e "$install_path"
        else
            echo "[ PIP    ] $PIP_CMD install -e $install_rel"
            $PIP_CMD install -e "$install_path"
        fi
    fi
}

while IFS='|' read -r name url sha rel_path pip_install lfs pip_subpath unzip_csv pip_no_deps; do
    [ -z "$name" ] && continue
    sync_one "$name" "$url" "$sha" "$rel_path" "$pip_install" "$lfs" "$pip_subpath" "$unzip_csv" "$pip_no_deps"
done <<< "$entries"

echo
echo "=== generate: assets ==="
python3 -m assets.cli generate --quiet

# Nominal stand clip — Orcs-Dodge-AdaptSonic's reference motion. Generated, not
# fetched, so it belongs here rather than in deps.lock. Non-fatal: it steps
# mujoco-warp, which a GPU-less login node cannot do, and every other task is
# unaffected.
echo
echo "=== generate: nominal motion ==="
NOMINAL="${ORCS_DATA_ROOT:-$REPO_ROOT/data}/nominal_motions/nominal/stand/sample1/motion.npz"
if [ -f "$NOMINAL" ]; then
    echo "[ SKIP   ] already present: ${NOMINAL#"$REPO_ROOT/"}"
elif python3 -m orcs.cli.make_nominal_motion; then
    echo "[ OK     ] nominal clip generated"
else
    echo "[WARN] nominal-motion generation failed (it needs a GPU for mujoco-warp)."
    echo "       Orcs-Dodge-AdaptSonic stays unregistered until you run, on a GPU:"
    echo "         orcs-make-nominal"
fi
