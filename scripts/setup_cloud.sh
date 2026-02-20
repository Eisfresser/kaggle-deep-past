#!/usr/bin/env bash
# Setup script for the RunPod cloud machine.
# Can be run EITHER locally (Mac) or directly on the pod — it detects which.
#
# From Mac:     setup_cloud.sh [git-repo-url]            (host from .env)
#               setup_cloud.sh [user@host] [git-repo-url] (explicit host)
# On the pod:   setup_cloud.sh [git-repo-url]
#
# When run locally it SSHs into the pod, uploads itself, and executes there.
# Uses RUNPOD_SSH_HOST from .env when no host arg is given (same as other scripts).
set -euo pipefail

# ── Detect: are we on the pod or on a local dev machine? ─────────────────
# /workspace is the RunPod persistent volume — it won't exist on a Mac.
if [ -d /workspace ]; then
    ON_POD=true
else
    ON_POD=false
fi

if [ "${ON_POD}" = false ]; then
    # ── Running locally — forward to the pod via SSH ──────────────────────
    SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"

    # If $1 looks like a URL (https:// or git@), it's the repo URL, not a host.
    # This allows: setup_cloud.sh <url>           (use .env host)
    #              setup_cloud.sh <host> <url>     (explicit host)
    if [[ "${1:-}" == https://* || "${1:-}" == git@* ]]; then
        _HOST_ARG=""
        REPO_URL="${1}"
    else
        _HOST_ARG="${1:-}"
        REPO_URL="${2:-}"
    fi

    source "${SCRIPT_DIR}/_resolve_remote.sh" "${_HOST_ARG}"
    unset _HOST_ARG

    echo "=== Uploading setup_cloud.sh to ${REMOTE} ==="

    # Upload script to pod, then run it in a detached tmux session so the
    # setup survives SSH disconnects (flash-attn compile takes ~30-60 min).
    ssh -p "${SSH_PORT}" "${REMOTE}" "cat > /tmp/setup_cloud.sh && chmod +x /tmp/setup_cloud.sh" \
        < "${SCRIPT_DIR}/setup_cloud.sh"

    _TMUX_CMD="/tmp/setup_cloud.sh${REPO_URL:+ '${REPO_URL}'} 2>&1 | tee /tmp/setup.log"
    ssh -p "${SSH_PORT}" "${REMOTE}" \
        "command -v tmux >/dev/null || { apt-get update -qq && apt-get install -y -q tmux rsync btop; }; \
         tmux kill-session -t setup 2>/dev/null || true; \
         tmux new-session -d -s setup \"${_TMUX_CMD}\""
    unset _TMUX_CMD

    echo ""
    echo "Setup is running in tmux session 'setup' on ${REMOTE}."
    echo "It will survive SSH disconnects. Monitor with:"
    echo "  Attach:  ssh -p ${SSH_PORT} ${REMOTE} -t 'tmux attach -t setup'"
    echo "  Log:     ssh -p ${SSH_PORT} ${REMOTE} 'tail -f /tmp/setup.log'"
    exit 0
fi

# ── From here on we are running ON the pod ────────────────────────────────
WORK=/workspace/deep-past

# ── system tools ────────────────────────────────────────────────────────────
command -v tmux >/dev/null || { apt-get update -qq && apt-get install -y -q tmux; }

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
# The compiled .venv (incl. flash-attn) is backed up to /workspace/.venv-cache/
# so it survives pod deletion. A metadata file records the environment it was
# built for; if it doesn't match the current pod, the cache is skipped and a
# fresh build happens instead.
VENV_CACHE="/workspace/.venv-cache/deep-past"
VENV_META="${VENV_CACHE}.meta"
echo "=== Syncing Python + deps ==="
uv python install 3.11
_py=$(python3 --version 2>&1)
_driver=$(nvidia-smi --query-gpu=driver_version --format=csv,noheader 2>/dev/null | head -1 || echo "no-gpu")
if [ ! -d "${WORK}/.venv" ] && [ -d "${VENV_CACHE}" ] && [ -f "${VENV_META}" ]; then
    _saved_py=$(grep '^python=' "${VENV_META}" | cut -d= -f2-)
    _saved_driver=$(grep '^driver=' "${VENV_META}" | cut -d= -f2-)
    if [ "${_py}" = "${_saved_py}" ] && [ "${_driver}" = "${_saved_driver}" ]; then
        echo "=== Restoring .venv from cache (${_py}, driver=${_driver}) ==="
        cp -a "${VENV_CACHE}" "${WORK}/.venv"
    else
        echo "=== Cache environment mismatch — rebuilding .venv ==="
        echo "  saved:   python=${_saved_py}  driver=${_saved_driver}"
        echo "  current: python=${_py}  driver=${_driver}"
    fi
fi
# MAX_JOBS limits parallel C++ compile workers for flash-attn.
# Each job uses ~15 GB RAM; 2 jobs is safe on a 61 GB pod.
MAX_JOBS=2 nice uv sync --extra cuda
# Back up the venv so it survives pod deletion, with metadata for validation.
_torch=$(uv run python -c "import torch; print(torch.__version__)" 2>/dev/null || echo "unknown")
echo "=== Backing up .venv to ${VENV_CACHE} ==="
mkdir -p "$(dirname "${VENV_CACHE}")"
rm -rf "${VENV_CACHE}"
cp -a "${WORK}/.venv" "${VENV_CACHE}"
printf 'python=%s\ndriver=%s\ntorch=%s\n' "${_py}" "${_driver}" "${_torch}" > "${VENV_META}"
echo "    Backup complete ($(du -sh "${VENV_CACHE}" | cut -f1)). torch=${_torch}"
unset _py _driver _saved_py _saved_driver _torch

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
