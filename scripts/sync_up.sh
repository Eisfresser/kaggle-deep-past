#!/usr/bin/env bash
# Sync code, data, and .env (for WANDB_API_KEY etc.) to RunPod machine.
# Targets /workspace/deep-past (persistent volume — survives pod restarts).
# Usage: sync_up.sh [user@host]   (defaults to root@RUNPOD_SSH_HOST from .env)
set -euo pipefail

PROJECT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
source "${PROJECT_DIR}/scripts/_resolve_remote.sh" "${1:-}"

rsync -avz --progress \
    --exclude '.git' \
    --exclude '__pycache__' \
    --exclude 'outputs/checkpoints' \
    --exclude 'outputs/merged' \
    --exclude 'wandb' \
    --exclude '.venv' \
    "${PROJECT_DIR}/" "${REMOTE}:/workspace/deep-past/"

echo "Synced to ${REMOTE}:/workspace/deep-past/"
