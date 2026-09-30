#!/usr/bin/env bash
# Compatibility entry point for the former combined sync.
# New callers should run sync_deps.sh, then sync_data.sh with the desired modes.

set -euo pipefail
SETUP_DIR="$(cd "$(dirname "$0")" && pwd)"

echo "[DEPRECATED] sync_dependencies.sh is now split: sync_deps.sh + sync_data.sh"
bash "$SETUP_DIR/sync_deps.sh" "$@"
# Preserve the former scope: locked datasets + nominal motion, not PerLoco sources.
bash "$SETUP_DIR/sync_data.sh" inhouse "$@"
