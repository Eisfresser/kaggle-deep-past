#!/usr/bin/env bash
# Run training or a full sweep on the remote RunPod machine in a tmux session.
#
# Usage:
#   run_remote.sh [user@host] [config]      # single config (default: configs/cloud.yaml)
#   run_remote.sh [user@host] --sweep        # run all configs/sweep_*.yaml sequentially
#
# Defaults to root@RUNPOD_SSH_HOST from .env if no host is given.
# Pair with watch_remote.sh to auto-sync results and shutdown the pod when done.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
source "${SCRIPT_DIR}/_resolve_remote.sh" "${1:-}"
MODE="${2:---single}"
SESSION="training"

PREAMBLE='source "$HOME/.local/bin/env" && export HF_HOME=/workspace/.cache/huggingface && set -a && source .env && set +a'

if [ "${MODE}" = "--sweep" ]; then
    CMD="./scripts/sweep.sh 2>&1 | tee outputs/sweep.log"
    echo "Starting sweep on ${REMOTE} (all configs/sweep_*.yaml)..."
else
    CONFIG="${MODE:-configs/cloud.yaml}"
    # Treat --single with no explicit config as cloud.yaml
    if [ "${CONFIG}" = "--single" ]; then
        CONFIG="configs/cloud.yaml"
    fi
    CMD="uv run python src/train.py ${CONFIG} 2>&1 | tee outputs/train.log"
    echo "Starting training on ${REMOTE} with ${CONFIG}..."
fi

ssh "${REMOTE}" "bash -lc '
    cd /workspace/deep-past
    export HF_HOME=/workspace/.cache/huggingface
    tmux kill-session -t ${SESSION} 2>/dev/null || true
    tmux new-session -d -s ${SESSION} \
        \"${PREAMBLE} && ${CMD}\"
'"

echo "Started in tmux session '${SESSION}'."
echo "Monitor:          ssh ${REMOTE} -t 'tmux attach -t ${SESSION}'"
if [ "${MODE}" = "--sweep" ]; then
    echo "Watch log:        ssh ${REMOTE} 'tail -f /workspace/deep-past/outputs/sweep.log'"
else
    echo "Watch log:        ssh ${REMOTE} 'tail -f /workspace/deep-past/outputs/train.log'"
fi
echo "Auto-shutdown:    ./scripts/watch_remote.sh ${REMOTE}"
