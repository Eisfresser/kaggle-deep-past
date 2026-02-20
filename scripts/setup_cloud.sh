#!/usr/bin/env bash
# Run ON the RunPod machine after first SSH login.
# Idempotent — safe to re-run after pod restart. Skips steps already done.
#
# Uses /workspace/ (persistent volume) so Python, deps, model cache, and
# checkpoints survive pod stop/restart.
#
# First run:  setup_cloud.sh <git-repo-url>
# After restart: setup_cloud.sh  (no args needed — repo already there)
set -euo pipefail

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
