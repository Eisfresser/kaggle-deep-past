# Shared helper — sourced by other scripts, not run directly.
# Resolves REMOTE and SSH_PORT from $1 or falls back to env vars from .env.
#
# After sourcing, callers have:
#   REMOTE   — user@host
#   SSH_PORT — port number (defaults to 22 if RUNPOD_SSH_PORT is unset)
#
# Usage (in calling script):
#   source "$(dirname "$0")/_resolve_remote.sh" "${1:-}"

_ARG="${1:-}"

# Always load .env — needed for RUNPOD_SSH_PORT and, when no host arg, RUNPOD_SSH_HOST.
_ENV_FILE="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)/.env"
if [ -f "${_ENV_FILE}" ]; then
    _HOST_FROM_ENV=$(grep -E '^RUNPOD_SSH_HOST=' "${_ENV_FILE}" | cut -d= -f2- | tr -d '[:space:]')
    _PORT_FROM_ENV=$(grep -E '^RUNPOD_SSH_PORT=' "${_ENV_FILE}" | cut -d= -f2- | tr -d '[:space:]')
    RUNPOD_SSH_HOST="${RUNPOD_SSH_HOST:-${_HOST_FROM_ENV}}"
    RUNPOD_SSH_PORT="${RUNPOD_SSH_PORT:-${_PORT_FROM_ENV}}"
fi

SSH_PORT="${RUNPOD_SSH_PORT:-22}"

if [ -n "${_ARG}" ]; then
    REMOTE="${_ARG}"
else
    if [ -z "${RUNPOD_SSH_HOST:-}" ]; then
        echo "ERROR: No host specified and RUNPOD_SSH_HOST not set in .env"
        echo "Usage: $(basename "$0") [user@host]"
        echo "   or: set RUNPOD_SSH_HOST in .env"
        exit 1
    fi

    # Default to root@ if the value doesn't already contain @
    if [[ "${RUNPOD_SSH_HOST}" == *@* ]]; then
        REMOTE="${RUNPOD_SSH_HOST}"
    else
        REMOTE="root@${RUNPOD_SSH_HOST}"
    fi
fi

unset _ARG _ENV_FILE _HOST_FROM_ENV _PORT_FROM_ENV
