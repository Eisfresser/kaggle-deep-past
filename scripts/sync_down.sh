#!/usr/bin/env bash
# Usage: sync_down.sh [user@host]   (defaults to root@RUNPOD_SSH_HOST from .env)
set -euo pipefail

PROJECT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
source "${PROJECT_DIR}/scripts/_resolve_remote.sh" "${1:-}"

rsync -avz --progress -e "ssh -p ${SSH_PORT}" \
    "${REMOTE}:/workspace/deep-past/outputs/" "${PROJECT_DIR}/outputs/"

echo "Downloaded outputs from ${REMOTE}"
