#!/usr/bin/env bash
set -euo pipefail

SLUG="${1:?Usage: upload_kaggle_dataset.sh your-username/dataset-name}"
MODEL_DIR="${2:-outputs/merged}"

cat > "${MODEL_DIR}/dataset-metadata.json" << EOF
{
  "title": "$(basename "${SLUG}")",
  "id": "${SLUG}",
  "licenses": [{"name": "apache-2.0"}]
}
EOF

if kaggle datasets status "${SLUG}" 2>/dev/null; then
    echo "Updating existing dataset..."
    kaggle datasets version -p "${MODEL_DIR}" -m "$(date +%Y%m%d-%H%M)" --dir-mode zip
else
    echo "Creating new dataset..."
    kaggle datasets create -p "${MODEL_DIR}" --dir-mode zip
fi

echo "Dataset uploaded: https://www.kaggle.com/datasets/${SLUG}"
