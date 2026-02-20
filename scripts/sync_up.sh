#!/usr/bin/env bash
# Sync code, data, and .env (for WANDB_API_KEY etc.) to RunPod machine.
# Targets /workspace/deep-past (persistent volume — survives pod restarts).
# Usage: sync_up.sh user@host
set -euo pipefail

REMOTE="${1:?Usage: sync_up.sh user@host}"
PROJECT_DIR="$(cd "$(dirname "$0")/.." && pwd)"

rsync -avz --progress \
    --exclude '.git' \
    --exclude '__pycache__' \
    --exclude 'outputs/checkpoints' \
    --exclude 'outputs/merged' \
    --exclude 'wandb' \
    --exclude '.venv' \
    "${PROJECT_DIR}/" "${REMOTE}:/workspace/deep-past/"

echo "Synced to ${REMOTE}:/workspace/deep-past/"
