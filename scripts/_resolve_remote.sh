# Shared helper — sourced by other scripts, not run directly.
# Resolves REMOTE from $1 or falls back to root@RUNPOD_SSH_HOST from .env.
#
# Usage (in calling script):
#   source "$(dirname "$0")/_resolve_remote.sh" "${1:-}"

_ARG="${1:-}"

if [ -n "${_ARG}" ]; then
    REMOTE="${_ARG}"
else
    # Try loading RUNPOD_SSH_HOST from .env
    _ENV_FILE="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)/.env"
    if [ -f "${_ENV_FILE}" ]; then
        RUNPOD_SSH_HOST=$(grep -E '^RUNPOD_SSH_HOST=' "${_ENV_FILE}" | cut -d= -f2- | tr -d '[:space:]')
    fi

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

unset _ARG _ENV_FILE
