#!/usr/bin/env bash
# Poll the RunPod machine until training finishes, sync results down, then
# stop the pod to save costs. Run locally after run_remote.sh.
#
# Usage: watch_remote.sh user@host [--no-shutdown]
#
# Requires: runpodctl installed locally (https://github.com/runpod/runpodctl)
# Pass --no-shutdown to skip the shutdown step.
set -euo pipefail

REMOTE="${1:?Usage: watch_remote.sh user@host [--no-shutdown]}"
NO_SHUTDOWN="${2:-}"
POLL_INTERVAL=60   # seconds between checks
SESSION="training"
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"

echo "Watching ${REMOTE} for tmux session '${SESSION}' to finish..."
echo "Polling every ${POLL_INTERVAL}s. Press Ctrl-C to stop watching (training continues)."

while true; do
    # Check if the tmux session still exists on the remote
    if ssh -o ConnectTimeout=10 "${REMOTE}" "tmux has-session -t ${SESSION} 2>/dev/null"; then
        # Still running — show last line of log
        LAST_LINE=$(ssh -o ConnectTimeout=10 "${REMOTE}" \
            "tail -1 /workspace/deep-past/outputs/train.log 2>/dev/null" || echo "(no log yet)")
        echo "[$(date +%H:%M:%S)] Training in progress... ${LAST_LINE}"
        sleep "${POLL_INTERVAL}"
    else
        echo ""
        echo "[$(date +%H:%M:%S)] Training session finished!"
        break
    fi
done

# Sync results down
echo "=== Syncing outputs ==="
"${SCRIPT_DIR}/sync_down.sh" "${REMOTE}"

# Check if training succeeded (look for final checkpoint)
if [ -d "outputs/checkpoints/cloud/final" ] || \
   ls outputs/checkpoints/cloud/checkpoint-* &>/dev/null; then
    echo "Checkpoints downloaded successfully."
else
    echo "WARNING: No checkpoints found in outputs/. Training may have failed."
    echo "Check the log: outputs/train.log"
fi

# Stop the RunPod instance
if [ "${NO_SHUTDOWN}" = "--no-shutdown" ]; then
    echo "Skipping shutdown (--no-shutdown)."
else
    echo "=== Stopping RunPod instance ==="
    if command -v runpodctl &>/dev/null; then
        # Get pod ID from the SSH host or let runpodctl figure it out
        POD_ID=$(ssh -o ConnectTimeout=10 "${REMOTE}" \
            'echo "${RUNPOD_POD_ID:-}"' 2>/dev/null || echo "")
        if [ -n "${POD_ID}" ]; then
            runpodctl stop pod "${POD_ID}"
            echo "RunPod pod ${POD_ID} stopped."
        else
            echo "Could not detect RUNPOD_POD_ID on remote."
            echo "Stop manually: runpodctl stop pod <pod-id>"
        fi
    else
        echo "runpodctl not found. Install it:"
        echo "  brew install runpod/runpodctl/runpodctl   # macOS"
        echo "  # or: https://github.com/runpod/runpodctl"
        echo "Then stop manually: runpodctl stop pod <pod-id>"
    fi
fi

echo "Done."
