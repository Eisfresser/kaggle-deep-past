#!/usr/bin/env bash
# Setup script for the RunPod cloud machine.
# Can be run EITHER locally (Mac) or directly on the pod — it detects which.
#
# From Mac:     setup_cloud.sh [user@host] [git-repo-url]
# On the pod:   setup_cloud.sh [git-repo-url]
#
# When run locally it SSHs into the pod, uploads itself, and executes there.
# Uses RUNPOD_SSH_HOST from .env when no host arg is given (same as other scripts).
set -euo pipefail

# ── Detect: are we on the pod or on a local dev machine? ─────────────────
if [ -d /workspace ] && [ -f /etc/hostname ] && grep -qi runpod /etc/hostname 2>/dev/null; then
    ON_POD=true
elif [ -d /workspace ] && [ -n "${RUNPOD_POD_ID:-}" ]; then
    ON_POD=true
else
    ON_POD=false
fi

if [ "${ON_POD}" = false ]; then
    # ── Running locally — forward to the pod via SSH ──────────────────────
    SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
    source "${SCRIPT_DIR}/_resolve_remote.sh" "${1:-}"
    REPO_URL="${2:-}"

    echo "=== Running setup_cloud.sh on ${REMOTE} via SSH ==="

    # Upload this script and execute it remotely
    if [ -n "${REPO_URL}" ]; then
        ssh -p "${SSH_PORT}" "${REMOTE}" "bash -s -- '${REPO_URL}'" < "${SCRIPT_DIR}/setup_cloud.sh"
    else
        ssh -p "${SSH_PORT}" "${REMOTE}" "bash -s" < "${SCRIPT_DIR}/setup_cloud.sh"
    fi
    exit $?
fi

# ── From here on we are running ON the pod ────────────────────────────────
WORK=/workspace/deep-past

# ── uv ──────────────────────────────────────────────────────────────────────
if command -v uv &>/dev/null; then
    echo "=== uv already installed ==="
else
    echo "=== Installing uv ==="
    curl -LsSf https://astral.sh/uv/install.sh | sh
fi
source "$HOME/.local/bin/env" 2>/dev/null || true

# ── Repo ────────────────────────────────────────────────────────────────────
if [ -d "${WORK}/.git" ]; then
    echo "=== Repo already at ${WORK}, pulling latest ==="
    cd "${WORK}"
    git pull --ff-only || echo "Pull failed (maybe dirty) — continuing with existing code."
else
    REPO_URL="${1:?Usage: setup_cloud.sh <git-repo-url>  (first run only)}"
    echo "=== Cloning repo → ${WORK} ==="
    git clone "${REPO_URL}" "${WORK}"
    cd "${WORK}"
fi

# ── Python + deps ───────────────────────────────────────────────────────────
# uv caches in .venv inside the project dir (on the volume), so this is
# nearly instant on restart — only re-resolves if pyproject.toml changed.
echo "=== Syncing Python + deps ==="
uv python install 3.11
uv sync --extra cuda

# ── HuggingFace cache on volume ────────────────────────────────────────────
# Keep downloaded model weights on the persistent volume so they survive
# pod restarts. Default HF cache is in /root/.cache (container disk = wiped).
export HF_HOME=/workspace/.cache/huggingface
mkdir -p "${HF_HOME}"
echo "HF_HOME=${HF_HOME} (persistent)"

# ── .env ────────────────────────────────────────────────────────────────────
if [ -f .env ]; then
    set -a; source .env; set +a
    echo "=== Loaded .env (WANDB_API_KEY, KAGGLE_USERNAME, etc.) ==="
else
    echo "WARNING: No .env found. Run sync_up.sh first or set vars manually."
fi

# ── GPU check ───────────────────────────────────────────────────────────────
echo "=== Verify GPU ==="
uv run python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}, GPUs: {torch.cuda.device_count()}')"

echo ""
echo "=== Ready. Run training with: ==="
echo "cd ${WORK} && uv run python src/train.py configs/cloud.yaml"
