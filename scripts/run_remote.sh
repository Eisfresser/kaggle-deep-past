#!/usr/bin/env bash
# Run training or a full sweep on the remote RunPod machine in a tmux session.
#
# Usage:
#   run_remote.sh [user@host] [config]   # single training run (default: configs/cloud.yaml)
#   run_remote.sh [--sweep]              # sweep all configs/sweep_*.yaml
#   run_remote.sh [user@host] --sweep    # sweep on explicit host
#
# --sweep can appear anywhere in the arg list.
# Defaults to root@RUNPOD_SSH_HOST from .env if no host is given.
# Pair with watch_remote.sh to auto-sync results and shutdown the pod when done.
set -euo pipefail

# ── Parse args: pull --sweep out, leave positional args intact ────────────
SWEEP=false
POSITIONAL=()
for arg in "$@"; do
    if [ "${arg}" = "--sweep" ]; then
        SWEEP=true
    else
        POSITIONAL+=("${arg}")
    fi
done

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
source "${SCRIPT_DIR}/_resolve_remote.sh" "${POSITIONAL[0]:-}"
SESSION="training"

PREAMBLE='source "$HOME/.local/bin/env" && export HF_HOME=/workspace/.cache/huggingface && set -a && source .env && set +a'

if [ "${SWEEP}" = true ]; then
    CMD="./scripts/sweep.sh 2>&1 | tee outputs/sweep.log"
    echo "Starting sweep on ${REMOTE} (all configs/sweep_*.yaml)..."
else
    CONFIG="${POSITIONAL[1]:-configs/cloud.yaml}"
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
if [ "${SWEEP}" = true ]; then
    echo "Watch log:        ssh ${REMOTE} 'tail -f /workspace/deep-past/outputs/sweep.log'"
else
    echo "Watch log:        ssh ${REMOTE} 'tail -f /workspace/deep-past/outputs/train.log'"
fi
echo "Auto-shutdown:    ./scripts/watch_remote.sh ${REMOTE}"
