#!/usr/bin/env bash
set -euo pipefail

COMPETITION="deep-past-initiative-machine-translation"

mkdir -p data/raw
kaggle competitions download -c "${COMPETITION}" -p data/raw/
cd data/raw && unzip -o "*.zip" && rm -f *.zip
echo "Data downloaded to data/raw/"
