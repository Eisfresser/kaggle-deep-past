#!/usr/bin/env bash
set -euo pipefail

# Load env vars from .env if present
ENV_FILE="$(cd "$(dirname "$0")/.." && pwd)/.env"
if [ -f "${ENV_FILE}" ]; then
    KAGGLE_DATASET_SLUG="${KAGGLE_DATASET_SLUG:-$(grep -E '^KAGGLE_DATASET_SLUG=' "${ENV_FILE}" | cut -d= -f2- | tr -d '[:space:]')}"
fi

SLUG="${1:-${KAGGLE_DATASET_SLUG:-}}"
if [ -z "${SLUG}" ]; then
    echo "ERROR: No slug specified and KAGGLE_DATASET_SLUG not set in .env"
    echo "Usage: upload_kaggle_dataset.sh [your-username/dataset-name]"
    exit 1
fi
MODEL_DIR="${2:-outputs/merged}"

cat > "${MODEL_DIR}/dataset-metadata.json" << EOF
{
  "title": "$(basename "${SLUG}")",
  "id": "${SLUG}",
  "licenses": [{"name": "apache-2.0"}]
}
EOF

if uv run kaggle datasets status "${SLUG}" 2>/dev/null; then
    echo "Updating existing dataset..."
    uv run kaggle datasets version -p "${MODEL_DIR}" -m "$(date +%Y%m%d-%H%M)" --dir-mode zip
else
    echo "Creating new dataset..."
    uv run kaggle datasets create -p "${MODEL_DIR}" --dir-mode zip
fi

echo "Dataset uploaded: https://www.kaggle.com/datasets/${SLUG}"
