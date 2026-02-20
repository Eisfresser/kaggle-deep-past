#!/usr/bin/env bash
# Run ON the RunPod machine after first SSH login.
# Expects: Ubuntu + CUDA drivers pre-installed (standard on RunPod)
# Expects: .env already synced by sync_up.sh (contains WANDB_API_KEY etc.)
set -euo pipefail

echo "=== Installing uv ==="
curl -LsSf https://astral.sh/uv/install.sh | sh
source "$HOME/.local/bin/env"

echo "=== Cloning repo ==="
REPO_URL="${1:?Usage: setup_cloud.sh <git-repo-url>}"
git clone "${REPO_URL}" ~/deep-past
cd ~/deep-past

echo "=== Installing Python + deps ==="
uv python install 3.11
uv sync --extra cuda

echo "=== Loading .env ==="
if [ -f .env ]; then
    set -a; source .env; set +a
    echo "Loaded .env (WANDB_API_KEY, KAGGLE_USERNAME, etc.)"
else
    echo "WARNING: No .env found. Run sync_up.sh first or set vars manually."
fi

echo "=== Verify GPU ==="
python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}, GPUs: {torch.cuda.device_count()}')"

echo "=== Done. Run training with: ==="
echo "cd ~/deep-past && uv run python src/train.py configs/cloud.yaml"
