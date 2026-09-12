#!/usr/bin/env bash
# Download and checksum all public ORCS model checkpoints.

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
export PYTHONPATH="$REPO_ROOT/src${PYTHONPATH:+:$PYTHONPATH}"
exec "${PYTHON:-python3}" -m orcs.cli.download_released_models "$@"
