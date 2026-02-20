#!/usr/bin/env bash
set -euo pipefail

REMOTE="${1:?Usage: sync_down.sh user@host}"
PROJECT_DIR="$(cd "$(dirname "$0")/.." && pwd)"

rsync -avz --progress \
    "${REMOTE}:/workspace/deep-past/outputs/" "${PROJECT_DIR}/outputs/"

echo "Downloaded outputs from ${REMOTE}"
