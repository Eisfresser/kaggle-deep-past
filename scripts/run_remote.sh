#!/usr/bin/env bash
# Usage: run_remote.sh [user@host] [config]   (defaults to root@RUNPOD_SSH_HOST from .env)
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
source "${SCRIPT_DIR}/_resolve_remote.sh" "${1:-}"
CONFIG="${2:-configs/cloud.yaml}"
SESSION="training"

echo "Starting training on ${REMOTE} with ${CONFIG}..."
ssh "${REMOTE}" "bash -lc '
    cd /workspace/deep-past
    export HF_HOME=/workspace/.cache/huggingface
    tmux kill-session -t ${SESSION} 2>/dev/null || true
    tmux new-session -d -s ${SESSION} \
        \"source \\\"$HOME/.local/bin/env\\\" && export HF_HOME=/workspace/.cache/huggingface && set -a && source .env && set +a && uv run python src/train.py ${CONFIG} 2>&1 | tee outputs/train.log\"
'"

echo "Training started in tmux session '${SESSION}'."
echo "Monitor: ssh ${REMOTE} -t 'tmux attach -t ${SESSION}'"
echo "Or watch: ssh ${REMOTE} 'tail -f /workspace/deep-past/outputs/train.log'"
echo "Or fire-and-forget: ./scripts/watch_remote.sh ${REMOTE}"
