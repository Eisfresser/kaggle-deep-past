# Deep Past Challenge — Implementation Plan

**Goal:** Fine-tune Qwen3-4B-Instruct-2507 for Akkadian → English translation  
**Dev:** MacBook Pro M3 18 GB · **Train:** Cloud CUDA (Vast.ai / RunPod)  
**Tooling:** `uv`, public GitHub repo, SSH-based cloud workflow  
**Deadline:** March 23, 2026

---

## 1. Key Competition Constraints

| Constraint | Implication |
|---|---|
| Code competition (Kaggle notebook) | Final submission = notebook, ≤9h, no internet. Model must be uploaded as Kaggle Dataset. |
| Train: ~1500 doc-level pairs | Very low resource. Data augmentation from publications.csv is critical. |
| Test: ~4000 sentence-level pairs | Need sentence segmentation + alignment strategy for training data too. |
| Metric: geomean(BLEU, chrF++) | Optimize for both precision (BLEU) and character-level recall (chrF++). |
| Heavy preprocessing needed | Determinatives, broken text markers, scribal notations, Ḫ→H substitution. |
| Pre-trained models allowed | Can use Qwen3-4B or any public model as base. |

---

## 2. Repository Structure

```
deep-past-akkadian/
├── pyproject.toml
├── .python-version                 # 3.11
├── .gitignore
├── README.md
│
├── configs/
│   ├── local.yaml                  # subset, short runs, MPS/CPU
│   ├── cloud.yaml                  # full dataset, CUDA, long runs
│   └── sweep.yaml                  # hyperparameter sweep definitions
│
├── src/deep_past/
│   ├── __init__.py
│   ├── preprocess.py               # Akkadian text cleaning pipeline
│   ├── align.py                    # sentence-level alignment from doc-level pairs
│   ├── augment.py                  # extract translations from publications.csv
│   ├── dataset.py                  # HF Dataset construction + chat template formatting
│   ├── train.py                    # QLoRA fine-tuning entry point
│   ├── inference.py                # batch translation generation
│   ├── evaluate.py                 # local BLEU + chrF++ scoring
│   └── submit.py                   # produce submission.csv from predictions
│
├── scripts/
│   ├── download_data.sh            # Kaggle API → data/raw/
│   ├── setup_cloud.sh              # bootstrap cloud GPU machine
│   ├── sync_up.sh                  # rsync code + data to cloud
│   ├── sync_down.sh                # rsync checkpoints + logs from cloud
│   ├── run_remote.sh               # SSH into cloud, run training
│   ├── export_model.sh             # merge LoRA + quantize for Kaggle upload
│   └── upload_kaggle_dataset.sh    # push model to Kaggle Datasets
│
├── notebooks/
│   └── submission.ipynb            # final Kaggle submission notebook (offline)
│
├── data/                           # .gitignore'd
│   ├── raw/                        # Kaggle competition CSVs
│   └── processed/                  # cleaned JSONL
│
├── outputs/                        # .gitignore'd
│   ├── checkpoints/
│   ├── merged/                     # LoRA-merged model
│   ├── predictions/
│   └── submissions/
│
└── wandb/                          # .gitignore'd
```

---

## 3. Dependencies — `pyproject.toml`

```toml
[project]
name = "deep-past-akkadian"
version = "0.1.0"
requires-python = ">=3.11"
dependencies = [
    "torch>=2.4",
    "transformers>=4.51",
    "peft>=0.15",
    "trl>=0.17",
    "datasets>=3.0",
    "accelerate>=1.0",
    "bitsandbytes>=0.45",
    "sacrebleu>=2.4",
    "pyyaml",
    "pandas",
    "wandb",
]

[project.optional-dependencies]
# Apple Silicon local dev (skip bitsandbytes, use MLX fallback)
mac = ["mlx-lm>=0.20"]
# Cloud CUDA (flash attention for speed)
cuda = ["flash-attn>=2.7"]

[project.scripts]
dp-preprocess = "deep_past.preprocess:main"
dp-train = "deep_past.train:main"
dp-infer = "deep_past.inference:main"
dp-eval = "deep_past.evaluate:main"
dp-submit = "deep_past.submit:main"
```

**Init and sync:**

```bash
# Local setup
uv init
uv sync                          # base deps
uv sync --extra mac              # + MLX on Mac

# Cloud setup (in setup_cloud.sh)
uv sync --extra cuda             # + flash-attn on CUDA
```

---

## 4. Implementation Phases

### Phase 0: Data Pipeline 

This is the highest-leverage work. Most competitors will use only the 1500 train
pairs. Extracting and aligning the publications data is a major advantage.

#### 0a. Preprocessing (`preprocess.py`)

Implement the cleaning pipeline per competition instructions:

```
Input (raw transliteration)          Output (cleaned)
─────────────────────────────        ────────────────────
a-lim{ki}                       →   a-lim {ki}
[KÙ.BABBAR]                    →   KÙ.BABBAR
Ḫa-bu-wa-al                    →   Ha-bu-wa-al
˹ša˺                           →   ša
ù / šu-ma                      →   ù šu-ma
[x]                             →   <gap>
[... ...]                       →   <big_gap>
```

Key operations:
1. Strip scribal notations: `!`, `?`, `/`, `:`, `˹ ˺`
2. Normalize brackets: `[ ]` → keep content, `[x]` → `<gap>`, `[... ...]` → `<big_gap>`
3. Handle `< >` (keep text, remove brackets) and `<< >>` (remove entirely)
4. Normalize `Ḫ ḫ` → `H h`
5. Standardize Unicode subscripts/superscripts to ASCII equivalents
6. Normalize determinatives (keep `{d}`, `{ki}` etc. as-is — they're semantically meaningful)
7. Strip/normalize line numbers from both transliteration and translation

#### 0b. Sentence Alignment (`align.py`)

Critical: training data is document-level, test data is sentence-level.

Use `Sentences_Oare_FirstWord_LinNum.csv` (provided by competition) to split
document-level pairs into sentence-level pairs. Strategy:

1. Parse line numbers from transliterations (lines like `1.`, `5.`, `10.` etc.)
2. Use the sentence boundary file to split transliteration at sentence boundaries
3. Align with translation text (heuristic: sentence count matching, length ratio, proper noun anchoring)
4. Validate alignment quality on a sample

#### 0c. Data Augmentation (`augment.py`)

Extract additional training pairs from `publications.csv` (~880 PDFs, OCR'd):

1. Identify pages with Akkadian content (`has_akkadian == True`)
2. Pattern-match document IDs (CDLI IDs, museum numbers, publication labels) against `published_texts.csv`
3. Extract translation blocks following transliterations
4. Detect language (many translations are French/German/Turkish) and translate non-English to English via an LLM API or Helsinki-NLP OPUS models
5. Align with corresponding transliterations from `published_texts.csv`
6. Quality-filter: drop pairs with length ratio outliers, encoding issues, or low confidence

**Expected yield:** Even extracting 500–1000 additional pairs would be a significant boost over the 1500 baseline.

#### 0d. Dataset Construction (`dataset.py`)

Format all pairs as instruction-tuned chat messages:

```json
{
  "messages": [
    {
      "role": "system",
      "content": "You are an expert translator of Old Assyrian Akkadian cuneiform texts. Translate the following transliterated Akkadian text into English. Produce a single fluent English sentence."
    },
    {
      "role": "user",
      "content": "Translate: um-ma A-šùr-na-da ma-la a-ḫi-a"
    },
    {
      "role": "assistant",
      "content": "Thus (says) Aššur-nādā: as much as my brother"
    }
  ]
}
```

Output: `data/processed/train.jsonl`, `data/processed/val.jsonl` (hold out ~10% for local eval).

Also build a version with **document context** — include preceding/following sentences as context in the user prompt, since translations often depend on surrounding text.

---

### Phase 1: Baseline Training

#### 1a. Training Script (`train.py`)

```python
# Simplified flow — actual script reads from config YAML

from unsloth import FastLanguageModel  # or use raw PEFT if Unsloth doesn't support MPS
from trl import SFTTrainer, SFTConfig
from datasets import load_dataset
import yaml, sys

def main():
    cfg = yaml.safe_load(open(sys.argv[1]))

    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=cfg["model_name"],     # "Qwen/Qwen3-4B-Instruct-2507"
        max_seq_length=cfg["max_seq_length"],  # 1024 local, 2048 cloud
        load_in_4bit=cfg["load_in_4bit"],      # True
    )

    model = FastLanguageModel.get_peft_model(
        model,
        r=cfg.get("lora_r", 32),
        lora_alpha=cfg.get("lora_alpha", 64),
        target_modules=cfg.get("target_modules", [
            "q_proj", "k_proj", "v_proj", "o_proj",
            "gate_proj", "up_proj", "down_proj"
        ]),
        lora_dropout=0.05,
    )

    dataset = load_dataset("json", data_files=cfg["train_data"], split="train")

    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=dataset,
        args=SFTConfig(
            output_dir=cfg["output_dir"],
            per_device_train_batch_size=cfg.get("batch_size", 2),
            gradient_accumulation_steps=cfg.get("grad_accum", 4),
            num_train_epochs=cfg.get("epochs", 3),
            learning_rate=cfg.get("lr", 2e-4),
            warmup_ratio=0.1,
            logging_steps=10,
            save_strategy="epoch",
            bf16=True,
            report_to="wandb",
            run_name=cfg.get("run_name", "deep-past"),
        ),
    )
    trainer.train()
    model.save_pretrained(cfg["output_dir"] + "/final")
```

#### 1b. Config Files

**`configs/local.yaml`** — Mac M3 quick iteration:
```yaml
model_name: "Qwen/Qwen3-4B-Instruct-2507"
max_seq_length: 512
load_in_4bit: true
train_data: "data/processed/train_small.jsonl"   # 100 samples
output_dir: "outputs/checkpoints/local"
batch_size: 1
grad_accum: 4
epochs: 1
lr: 2e-4
lora_r: 16
run_name: "local-smoke-test"
```

**`configs/cloud.yaml`** — full CUDA training:
```yaml
model_name: "Qwen/Qwen3-4B-Instruct-2507"
max_seq_length: 2048
load_in_4bit: true
train_data: "data/processed/train_full.jsonl"
output_dir: "outputs/checkpoints/cloud"
batch_size: 4
grad_accum: 4
epochs: 5
lr: 2e-4
lora_r: 32
lora_alpha: 64
run_name: "cloud-full-v1"
```

---

### Phase 2: Evaluation & Iteration 

#### 2a. Inference (`inference.py`)

Batch translate all test transliterations:

```python
def translate_batch(model, tokenizer, texts, cfg):
    """Generate translations with configurable decoding."""
    results = []
    for text in texts:
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": f"Translate: {text}"},
        ]
        input_ids = tokenizer.apply_chat_template(messages, return_tensors="pt")
        output = model.generate(
            input_ids,
            max_new_tokens=cfg.get("max_new_tokens", 512),
            temperature=cfg.get("temperature", 0.7),
            top_p=cfg.get("top_p", 0.8),
            do_sample=cfg.get("do_sample", True),
            num_beams=cfg.get("num_beams", 1),
        )
        results.append(tokenizer.decode(output[0][input_ids.shape[1]:], skip_special_tokens=True))
    return results
```

#### 2b. Local Evaluation (`evaluate.py`)

Score against validation set using SacreBLEU:

```python
import sacrebleu
import math

def score(predictions: list[str], references: list[str]) -> dict:
    bleu = sacrebleu.corpus_bleu(predictions, [references])
    chrf = sacrebleu.corpus_chrf(predictions, [references], word_order=2)  # chrF++
    geomean = math.sqrt(bleu.score * chrf.score)
    return {"bleu": bleu.score, "chrfpp": chrf.score, "geomean": geomean}
```

#### 2c. Iteration Strategy

| Lever | What to try |
|---|---|
| More data | Extract more pairs from publications.csv |
| System prompt | Add lexicon hints, determinative glossary |
| Few-shot in prompt | Include 2–3 example translations as context |
| LoRA rank | Sweep r=16, 32, 64 |
| Learning rate | Sweep 1e-4 to 5e-4 |
| Epochs | Watch for overfitting on val set |
| Decoding | beam search vs sampling, temperature tuning |
| Ensemble | Average predictions from multiple checkpoints |

---

### Phase 3: Kaggle Submission 

#### 3a. Model Export (`export_model.sh`)

```bash
#!/usr/bin/env bash
set -euo pipefail

# Merge LoRA weights into base model
python -c "
from unsloth import FastLanguageModel
model, tokenizer = FastLanguageModel.from_pretrained('outputs/checkpoints/cloud/final')
model.save_pretrained_merged('outputs/merged/', tokenizer, save_method='merged_16bit')
"

# Quantize to GGUF Q4_K_M for smaller upload (optional, if using llama.cpp in notebook)
# Or keep as safetensors if using transformers directly
echo "Merged model at outputs/merged/"
echo "Upload to Kaggle as a Dataset next."
```

#### 3b. Kaggle Dataset Upload (`upload_kaggle_dataset.sh`)

```bash
#!/usr/bin/env bash
set -euo pipefail

DATASET_SLUG="your-username/deep-past-qwen3-4b-akkadian"
MODEL_DIR="outputs/merged"

# Create Kaggle dataset metadata
cat > "${MODEL_DIR}/dataset-metadata.json" << EOF
{
  "title": "Deep Past Qwen3-4B Akkadian",
  "id": "${DATASET_SLUG}",
  "licenses": [{ "name": "apache-2.0" }]
}
EOF

kaggle datasets create -p "${MODEL_DIR}" --dir-mode zip
# Subsequent updates:
# kaggle datasets version -p "${MODEL_DIR}" -m "v2: more data" --dir-mode zip
```

#### 3c. Submission Notebook (`notebooks/submission.ipynb`)

The notebook runs offline in ≤9h. Skeleton:

```python
# Cell 1: Install deps (from Kaggle Dataset or pre-packaged wheels)
# Cell 2: Load model from /kaggle/input/deep-past-qwen3-4b-akkadian/
# Cell 3: Load + preprocess test.csv
# Cell 4: Generate translations (batch, with progress bar)
# Cell 5: Post-process (clean up artifacts, ensure one sentence per row)
# Cell 6: Write submission.csv
```

---

## 5. Helper Scripts

### `scripts/download_data.sh`

```bash
#!/usr/bin/env bash
set -euo pipefail

# Requires: pip install kaggle, KAGGLE_USERNAME + KAGGLE_KEY in env or ~/.kaggle/kaggle.json
COMPETITION="deep-past-initiative-machine-translation"

mkdir -p data/raw
kaggle competitions download -c "${COMPETITION}" -p data/raw/
cd data/raw && unzip -o "*.zip" && rm -f *.zip
echo "Data downloaded to data/raw/"
```

### `scripts/setup_cloud.sh`

```bash
#!/usr/bin/env bash
# Run ON the cloud machine after first SSH login.
# Expects: Ubuntu + CUDA drivers pre-installed (standard on Vast.ai/RunPod)
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

echo "=== Setting up wandb ==="
echo "Run: wandb login"

echo "=== Done. Run training with: ==="
echo "cd ~/deep-past && uv run dp-train configs/cloud.yaml"
```

### `scripts/sync_up.sh`

```bash
#!/usr/bin/env bash
# Sync code and data to cloud machine.
# Usage: sync_up.sh user@host
set -euo pipefail

REMOTE="${1:?Usage: sync_up.sh user@host}"
PROJECT_DIR="$(cd "$(dirname "$0")/.." && pwd)"

rsync -avz --progress \
    --exclude '.git' \
    --exclude '__pycache__' \
    --exclude 'outputs/checkpoints' \
    --exclude 'outputs/merged' \
    --exclude 'wandb' \
    --exclude '.venv' \
    "${PROJECT_DIR}/" "${REMOTE}:~/deep-past/"

echo "Synced to ${REMOTE}:~/deep-past/"
```

### `scripts/sync_down.sh`

```bash
#!/usr/bin/env bash
# Pull checkpoints and predictions from cloud.
# Usage: sync_down.sh user@host
set -euo pipefail

REMOTE="${1:?Usage: sync_down.sh user@host}"
PROJECT_DIR="$(cd "$(dirname "$0")/.." && pwd)"

rsync -avz --progress \
    "${REMOTE}:~/deep-past/outputs/" "${PROJECT_DIR}/outputs/"

echo "Downloaded outputs from ${REMOTE}"
```

### `scripts/run_remote.sh`

```bash
#!/usr/bin/env bash
# Run training on cloud via SSH. Detaches with tmux so you can disconnect.
# Usage: run_remote.sh user@host [config]
set -euo pipefail

REMOTE="${1:?Usage: run_remote.sh user@host [config]}"
CONFIG="${2:-configs/cloud.yaml}"
SESSION="training"

echo "Starting training on ${REMOTE} with ${CONFIG}..."
ssh "${REMOTE}" "bash -lc '
    cd ~/deep-past
    tmux kill-session -t ${SESSION} 2>/dev/null || true
    tmux new-session -d -s ${SESSION} \
        \"source \\\"$HOME/.local/bin/env\\\" && uv run dp-train ${CONFIG} 2>&1 | tee outputs/train.log\"
'"

echo "Training started in tmux session '${SESSION}'."
echo "Monitor: ssh ${REMOTE} -t 'tmux attach -t ${SESSION}'"
echo "Or watch: ssh ${REMOTE} 'tail -f ~/deep-past/outputs/train.log'"
```

### `scripts/export_model.sh`

```bash
#!/usr/bin/env bash
set -euo pipefail

CHECKPOINT="${1:-outputs/checkpoints/cloud/final}"
OUTPUT="${2:-outputs/merged}"

echo "Merging LoRA from ${CHECKPOINT} → ${OUTPUT}"
uv run python -c "
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

base = AutoModelForCausalLM.from_pretrained(
    'Qwen/Qwen3-4B-Instruct-2507', torch_dtype='auto', device_map='cpu'
)
model = PeftModel.from_pretrained(base, '${CHECKPOINT}')
merged = model.merge_and_unload()
merged.save_pretrained('${OUTPUT}')

tokenizer = AutoTokenizer.from_pretrained('${CHECKPOINT}')
tokenizer.save_pretrained('${OUTPUT}')
print('Merged model saved to ${OUTPUT}')
"
```

### `scripts/upload_kaggle_dataset.sh`

```bash
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
```

---

## 6. Typical Development Workflow

```
┌─────────────────────────────────────────────────────────┐
│  MacBook Pro M3                                         │
│                                                         │
│  1. Edit code in src/deep_past/                         │
│  2. uv run dp-preprocess          # build dataset       │
│  3. uv run dp-train configs/local.yaml  # smoke test    │
│  4. uv run dp-infer ...           # quick sanity check  │
│  5. git commit + push                                   │
│                                                         │
│  6. ./scripts/sync_up.sh user@gpu-box                   │
│  7. ./scripts/run_remote.sh user@gpu-box                │
│     ↕ (monitor via wandb dashboard or tmux attach)      │
│  8. ./scripts/sync_down.sh user@gpu-box                 │
│                                                         │
│  9. uv run dp-eval                # score locally       │
│ 10. uv run dp-submit              # generate CSV        │
│                                                         │
│ 11. ./scripts/export_model.sh     # merge LoRA          │
│ 12. ./scripts/upload_kaggle_dataset.sh user/model       │
│ 13. Submit notebook on Kaggle                           │
└─────────────────────────────────────────────────────────┘
```

---

## 7. Risk Assessment & Mitigations

| Risk | Impact | Mitigation |
|---|---|---|
| 18 GB RAM too tight for QLoRA | Training OOMs on Mac | Use `max_seq_length=512`, `batch_size=1`. Fallback: develop locally with CPU-only dry runs, train exclusively on cloud. |
| Unsloth doesn't support MPS | Can't fine-tune locally at all | Use raw PEFT + TRL (slower but works on MPS). Or use MLX-LM LoRA training. |
| Publications data extraction is noisy | Low-quality augmented data | Quality filter aggressively. Use a validation-set perplexity gate. |
| Kaggle notebook 9h limit | Model too slow for 4000 translations | Quantize to 4-bit GGUF, use llama.cpp or vLLM in notebook. Or use a smaller model if needed. |
| Sentence alignment is imperfect | Misaligned pairs degrade training | Hold out aligned pairs for validation. Use the provided `Sentences_Oare_FirstWord_LinNum.csv`. |

---

## 8. Timeline

| Week | Focus | Deliverable |
|---|---|---|
| 1 (now) | Data pipeline: preprocess, align, augment | `data/processed/train_full.jsonl` |
| 2 | Baseline training + eval loop | First model, validation scores |
| 3 | Iteration: data augmentation, hyperparams, prompting | Best model candidate |
| 4 | Kaggle notebook, model export, submission | Final submission by Mar 23 |
