#!/usr/bin/env bash
set -euo pipefail

REMOTE="${1:?Usage: run_remote.sh user@host [config]}"
CONFIG="${2:-configs/cloud.yaml}"
SESSION="training"

echo "Starting training on ${REMOTE} with ${CONFIG}..."
ssh "${REMOTE}" "bash -lc '
    cd ~/deep-past
    tmux kill-session -t ${SESSION} 2>/dev/null || true
    tmux new-session -d -s ${SESSION} \
        \"source \\\"$HOME/.local/bin/env\\\" && set -a && source .env && set +a && uv run python src/train.py ${CONFIG} 2>&1 | tee outputs/train.log\"
'"

echo "Training started in tmux session '${SESSION}'."
echo "Monitor: ssh ${REMOTE} -t 'tmux attach -t ${SESSION}'"
echo "Or watch: ssh ${REMOTE} 'tail -f ~/deep-past/outputs/train.log'"
echo "Or fire-and-forget: ./scripts/watch_remote.sh ${REMOTE}"
