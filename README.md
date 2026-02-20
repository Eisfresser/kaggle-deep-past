# kaggle-deep-past

Deep Past Challenge on kaggle [https://www.kaggle.com/competitions/deep-past-initiative-machine-translation] 

```bash
# download competition data
uv run bash scripts/download_data.sh

# clean the data
uv run python src/preprocess.py

# build json for training
uv run python src/dataset.py

# inference without lora
uv run python src/inference.py configs/local.yaml --no-lora --split val

# local smoke test for training
uv run python src/train.py configs/local.yaml

# inference with lora
uv run python src/inference.py configs/local.yaml --split val

```