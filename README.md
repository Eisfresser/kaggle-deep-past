# kaggle-deep-past

Deep Past Challenge on kaggle [https://www.kaggle.com/competitions/deep-past-initiative-machine-translation] 

Local smoke test
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

Cloud sweep
```bash
# setup runpod
./scripts/setup_cloud.sh https://github.com/Eisfresser/kaggle-deep-past.git

# upload latest code and .env
./scripts/sync_up.sh

# start sweep
./scripts/run_remote.sh --sweep

# wait for completion, syncs down automatically and stops the pod
./scripts/watch_remote.sh

# pick the best model and export it
./scripts/export_model.sh sweeps/qwen3-1.7b_r64-a128_lr2e-4_ep5/final outputs/merged

# upload best model to kaggle
./scripts/upload_kaggle_dataset.sh

# build the notebook from py files
uv run python scripts/build_notebook.py

# push notebook to kaggle
uv run kaggle kernels push -p notebooks/



```

Nvidia SMI during training of  
    model_name: "Qwen/Qwen3-1.7B"
    max_seq_length: 2048
    load_in_4bit: true
    inference_batch_size: 16
```text
Fri Feb 20 19:35:53 2026
+-----------------------------------------------------------------------------------------+
| NVIDIA-SMI 575.57.08              Driver Version: 575.57.08      CUDA Version: 12.9     |
|-----------------------------------------+------------------------+----------------------+
| GPU  Name                 Persistence-M | Bus-Id          Disp.A | Volatile Uncorr. ECC |
| Fan  Temp   Perf          Pwr:Usage/Cap |           Memory-Usage | GPU-Util  Compute M. |
|                                         |                        |               MIG M. |
|=========================================+========================+======================|
|   0  NVIDIA GeForce RTX 4090        On  |   00000000:02:00.0 Off |                  Off |
| 39%   63C    P2            348W /  450W |   18520MiB /  24564MiB |    100%      Default |
|                                         |                        |                  N/A |
+-----------------------------------------+------------------------+----------------------+

+-----------------------------------------------------------------------------------------+
| Processes:                                                                              |
|  GPU   GI   CI              PID   Type   Process name                        GPU Memory |
|        ID   ID                                                               Usage      |
|=========================================================================================|
|    0   N/A  N/A           24007      C   ...e/deep-past/.venv/bin/python3      18510MiB |
+-----------------------------------------------------------------------------------------+
```