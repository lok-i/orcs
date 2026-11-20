# Shared by sync_deps.sh and sync_data.sh — sourced, never run.
# Vibe owns the reference implementation; ORCS differs only in its lock rows.

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
SETUP_DIR="$REPO_ROOT/scripts/setup"
LOCK_FILE="$REPO_ROOT/deps.lock"
LOCK_HELPER="$REPO_ROOT/scripts/setup/locked_deps.py"

# Bound git-lfs on shared machines. Lower both when a login node has a tight
# process cap: LFS_JOBS=1 LFS_GOMAXPROCS=2.
LFS_JOBS=${LFS_JOBS:-8}
LFS_GOMAXPROCS=${LFS_GOMAXPROCS:-4}
LFS_RETRIES=${LFS_RETRIES:-3}

lfs_git() {
    GOMAXPROCS=$LFS_GOMAXPROCS git -c lfs.concurrenttransfers="$LFS_JOBS" "$@"
}

lfs_missing() {
    local repo="$1" include="${2:-}"
    local args=(-C "$repo" lfs ls-files)
    [ -z "$include" ] || args+=(-I "$include")
    lfs_git "${args[@]}" 2>/dev/null | sed -n 's/^[0-9a-f]* - //p' || true
}

lfs_retry() {
    local n=1
    until lfs_git "$@"; do
        if [ "$n" -ge "$LFS_RETRIES" ]; then
            echo "[ERROR] git-lfs failed ${LFS_RETRIES}x."
            echo "        Retry with LFS_JOBS=1 LFS_GOMAXPROCS=2."
            return 1
        fi
        echo "[ RETRY  ] git-lfs attempt $n/$LFS_RETRIES failed; backing off $((n * 10))s"
        sleep $((n * 10)); n=$((n + 1))
    done
}

fetch_sha() {
    local repo="$1" sha="$2"
    if git -C "$repo" -c fetch.negotiationAlgorithm=noop \
           fetch --progress --depth 1 origin "$sha"; then
        return 0
    fi
    echo "[ RETRY  ] no-negotiation fetch failed; using git's default"
    git -C "$repo" fetch --progress --depth 1 origin "$sha"
}

spin() {
    local label="$1"; shift
    "$@" &
    local pid=$!
    trap 'kill -- $pid 2>/dev/null; printf "\r\033[K" >&2; exit 130' INT
    local chars='|/-\' i=0 start=$SECONDS
    while kill -0 "$pid" 2>/dev/null; do
        local elapsed=$((SECONDS - start))
        printf "\r[ %-7s] %s  %02d:%02d  " "$label" \
            "${chars:i++%${#chars}:1}" $((elapsed / 60)) $((elapsed % 60)) >&2
        sleep 0.15
    done
    wait "$pid"; local rc=$?
    printf "\r\033[K" >&2
    trap - INT
    return $rc
}

# uv-only. Environment creation is an explicit prerequisite, not a side effect
# of another checkout. Setup creates ORCS's own .venv if absent and rejects an
# unrelated active environment such as Vibe's.
use_venv() {
    command -v uv >/dev/null || { echo "[ERROR] uv is required."; exit 1; }
    local venv="$REPO_ROOT/.venv"
    if [ -n "${VIRTUAL_ENV:-}" ] && [ "$(realpath "$VIRTUAL_ENV")" != "$(realpath -m "$venv")" ]; then
        echo "[ERROR] another environment is active: $VIRTUAL_ENV"
        echo "        Deactivate it; setup only manages $venv."
        exit 1
    fi
    if [ ! -f "$venv/pyvenv.cfg" ]; then
        uv venv --python "$(cat "$REPO_ROOT/.python-version")" --prompt "$(basename "$REPO_ROOT")" "$venv"
    fi
    if ! "$venv/bin/python" -c 'import sys; sys.exit(sys.version_info[:2] != (3, 12))'; then
        echo "[ERROR] this stack requires Python 3.12; the existing environment is unchanged."
        echo "        Use a fresh checkout/environment for the upgrade."
        exit 1
    fi
    export VIRTUAL_ENV="$venv" PATH="$venv/bin:$PATH"
    PIP_CMD=(uv --no-config pip)
    echo "[ENV] installer: uv pip -> $venv"
}

# lock_rows <path-prefix> [excluded-name...] -> pipe-separated records.
lock_rows() {
    [ -f "$LOCK_FILE" ] || {
        echo "[ERROR] deps.lock not found at $LOCK_FILE" >&2
        exit 1
    }
    python3 - "$LOCK_FILE" "$@" <<'PYEOF'
import json, sys

with open(sys.argv[1]) as f:
    lock = json.load(f)
prefix = sys.argv[2]
excluded = set(sys.argv[3:])
for name, info in lock.items():
    if name in excluded or not info["path"].startswith(prefix):
        continue
    print("|".join([
        name,
        info["url"],
        info["sha"],
        info["path"],
        "1" if info.get("pip_install") else "0",
        "1" if info.get("lfs") else "0",
        "1" if info.get("pip_no_deps") else "0",
        "1" if info.get("sparse") else "0",
    ]))
PYEOF
}

lean_include() {
    local repo="$1" globs err
    err=$(mktemp)
    globs=$(python -m orcs.cli.sync_native_data --lfs-include --retargeted-root "$repo" 2>"$err" \
        | sed -n 's/^include //p') || true
    if [ -z "$globs" ]; then
        echo "[ERROR] ORCS motion roster resolved no LFS includes:" >&2
        tail -20 "$err" >&2
        rm -f "$err"
        return 1
    fi
    rm -f "$err"

    # Keep all metadata so a future roster edit can resolve against this same
    # pointer-capable checkout.
    local patterns=('*.json')
    while IFS= read -r glob; do
        patterns+=("/$glob")
    done <<< "$globs"
    env GIT_LFS_SKIP_SMUDGE=1 git -C "$repo" sparse-checkout set --no-cone \
        "${patterns[@]}" >&2
    paste -sd, <<< "$globs"
}

sync_one() {
    local name="$1" url="$2" sha="$3" rel_path="$4"
    local pip_install="$5" lfs="$6" pip_no_deps="$7" sparse="$8"
    local lean_sparse="$9"
    local path="$REPO_ROOT/$rel_path"

    echo
    echo "=== $name @ ${sha:0:12} ==="
    echo "    url:  $url"
    echo "    path: $rel_path"

    if [ ! -e "$path/.git" ]; then
        echo "[ CLONE  ] $name -> $rel_path"
        mkdir -p "$path"
        git -C "$path" init -q
        git -C "$path" remote add origin "$url"
        [ "$lfs" = 0 ] || git -C "$path" lfs install --local
    fi

    local current
    current=$(git -C "$path" rev-parse HEAD 2>/dev/null || echo none)
    if [ "$current" != "$sha" ]; then
        if [ -f "$LOCK_HELPER" ]; then
            python "$LOCK_HELPER" before-checkout "$REPO_ROOT" "$name"
        fi
        echo "[ FETCH  ] depth=1 $sha"
        fetch_sha "$path" "$sha"
        if [ "$lfs" = 1 ]; then
            spin checkout env GIT_LFS_SKIP_SMUDGE=1 GOMAXPROCS="$LFS_GOMAXPROCS" \
                git -C "$path" -c lfs.concurrenttransfers="$LFS_JOBS" checkout -q "$sha"
        else
            git -C "$path" checkout -q "$sha"
        fi
    else
        echo "[ HEAD OK] already at $sha"
    fi

    if [ "$lfs" = 1 ]; then
        local include="" missing
        if [ "$sparse" = 1 ] && [ "$lean_sparse" = 1 ]; then
            include=$(lean_include "$path")
            echo "[ SPARSE ] $(tr ',' '\n' <<< "$include" | wc -l) glob(s) from ORCS's task roster"
        elif [ "$sparse" = 1 ] \
            && [ "$(git -C "$path" config --bool core.sparseCheckout || true)" = true ]; then
            echo "[ FULL   ] disabling sparse checkout"
            env GIT_LFS_SKIP_SMUDGE=1 git -C "$path" sparse-checkout disable
        fi

        missing=$(lfs_missing "$path" "$include")
        if [ -z "$missing" ]; then
            echo "[ LFS OK ] every object present — skipping pull"
        else
            echo "[ LFS    ] $(wc -l <<< "$missing") file(s) on pointers"
            local pull=(-C "$path" lfs pull)
            [ -z "$include" ] || pull+=(-I "$include")
            spin "LFS pull" lfs_retry "${pull[@]}"
        fi
    fi

}

sync_rows() {
    local rows="$1" lean_sparse="${2:-0}"
    local name url sha rel_path pip_install lfs pip_no_deps sparse
    while IFS='|' read -r name url sha rel_path pip_install lfs pip_no_deps sparse; do
        [ -z "$name" ] && continue
        sync_one "$name" "$url" "$sha" "$rel_path" "$pip_install" "$lfs" \
            "$pip_no_deps" "$sparse" "$lean_sparse"
    done <<< "$rows"
}

verify_rows() {
    local rows="$1" label="$2" rc=0
    local name url sha rel_path rest head status note
    echo
    echo "=== verify: $label HEAD vs deps.lock ==="
    while IFS='|' read -r name url sha rel_path rest; do
        [ -z "$name" ] && continue
        head=$(git -C "$REPO_ROOT/$rel_path" rev-parse -q --verify HEAD 2>/dev/null || true)
        if [ -z "$head" ]; then
            status="[ MISSING]"; note="no git worktree at $rel_path"; rc=1
        elif [ "$head" = "$sha" ]; then
            status="[   OK   ]"; note="${sha:0:12}"
        else
            status="[  DRIFT ]"; note="${head:0:12} != pinned ${sha:0:12}"; rc=1
        fi
        printf '%s %-24s %s\n' "$status" "$name" "$note"
    done <<< "$rows"
    if [ "$rc" != 0 ]; then
        echo "[ERROR] checkout does not match deps.lock; sync again after pushing the pin."
    fi
    return $rc
}

# The consumer supplies every editable dependency to ONE resolver invocation.
# This prevents transient PyPI replacements and resolves extras with the fork.
install_code() {
    local extras="$1" rows="$2"
    local args=(-e "$REPO_ROOT[$extras]") name url sha rel_path pip_install rest
    while IFS='|' read -r name url sha rel_path pip_install rest; do
        [ "$pip_install" != 1 ] || args+=(-e "$REPO_ROOT/$rel_path")
    done <<< "$rows"
    python "$LOCK_HELPER" patch "$REPO_ROOT"
    "${PIP_CMD[@]}" install --no-sources --default-index https://pypi.org/simple --python "$VIRTUAL_ENV/bin/python" "${args[@]}"
    "${PIP_CMD[@]}" check --python "$VIRTUAL_ENV/bin/python"
    python "$LOCK_HELPER" check-install "$REPO_ROOT"
}
