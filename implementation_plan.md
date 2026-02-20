# Deep Past Challenge — Implementation Plan

**Goal:** Fine-tune Qwen3-1.5B-Instruct-2507 for Akkadian → English translation
**Dev:** MacBook Pro M3 18 GB · **Train:** Cloud CUDA (Vast.ai / RunPod)
**Tooling:** `uv`, public GitHub repo, SSH-based cloud workflow
**Deadline:** March 23, 2026

---

## 1. Key Competition Constraints

| Constraint | Implication |
|---|---|
| Code competition (Kaggle notebook) | Final submission = notebook, ≤9h, no internet. Model must be uploaded as Kaggle Dataset. |
| Train: ~1500 doc-level pairs | Very low resource. Lexicon and alignment work are critical. |
| Test: ~4000 sentence-level pairs | Training is doc-level, test is sentence-level — must handle this mismatch. |
| Metric: geomean(BLEU, chrF++) | Optimize for both precision (BLEU) and character-level recall (chrF++). |
| Heavy preprocessing needed | Determinatives, broken text markers, scribal notations, Ḫ→H substitution. |
| Pre-trained models allowed | Using Qwen3-1.5B: fast inference allows ensembling and iteration within 9h budget. |

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
├── src/
│   ├── preprocess.py               # Akkadian text cleaning pipeline
│   ├── lexicon.py                  # proper noun + word lookup from OA_Lexicon_eBL.csv
│   ├── dataset.py                  # HF Dataset construction + chat template formatting
│   ├── train.py                    # QLoRA fine-tuning entry point
│   ├── inference.py                # batched translation generation
│   ├── postprocess.py              # clean model output, fix proper nouns
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
│   ├── upload_kaggle_dataset.sh    # push model to Kaggle Datasets
│   └── build_notebook.py           # auto-generate submission.ipynb from src/
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

[tool.uv]
# src/ is on the import path; no package to install
# Run modules directly: uv run python -m preprocess, etc.

[tool.setuptools]
# Not a distributable package — just scripts
py-modules = []
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

**Running scripts:** Since `src/` is a flat directory (not a package), run
modules directly:

```bash
uv run python src/preprocess.py            # or with config arg
uv run python src/train.py configs/cloud.yaml
uv run python src/inference.py configs/cloud.yaml
uv run python src/evaluate.py
uv run python src/submit.py
```

---

## 4. Part A — First Submission (End-to-End Baseline)

The goal is to get a score on the leaderboard as fast as possible. Train on
document-level pairs directly (no sentence alignment yet), submit, then iterate.

### A0. Zero-Shot Baseline (no training)

Before any fine-tuning, run the base Qwen3-1.5B-Instruct on the validation set
to establish a floor score. This serves two purposes:

1. **Validates the full pipeline end-to-end** — preprocessing, inference,
   post-processing, evaluation, and submission generation all work before
   investing time in training.
2. **Establishes the baseline** — knowing the zero-shot score tells you exactly
   how much value fine-tuning adds. If zero-shot is already decent, prompt
   engineering (Part B3) may be higher leverage than training.

```bash
# Preprocess data
uv run python src/preprocess.py

# Build dataset (needed for val split, even though we don't train yet)
uv run python src/dataset.py

# Run inference with base model (no LoRA checkpoint)
uv run python src/inference.py configs/local.yaml --no-lora

# Evaluate
uv run python src/evaluate.py
```

The `--no-lora` flag (or equivalent config toggle) skips loading a LoRA
checkpoint and runs the base model directly. Record the zero-shot geomean
score as the floor to beat.

### A1. Preprocessing (`preprocess.py`)

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

Apply the same cleaning to both `train.csv` transliterations and `test.csv`
transliterations. Apply lighter cleaning (just whitespace normalization) to
English translations.

### A2. Dataset Construction (`dataset.py`)

For the baseline, train on **document-level pairs** directly from `train.csv`.
No sentence alignment needed yet — the model learns translation quality from
complete documents.

Format all pairs as instruction-tuned chat messages:

```json
{
  "messages": [
    {
      "role": "system",
      "content": "You are an expert translator of Old Assyrian Akkadian cuneiform texts into English. Determinatives in curly brackets classify nouns: {d} = deity, {ki} = place, {m} = masculine name, {mi} = feminine name. Words in ALL CAPS are Sumerian logograms. Words with a capitalized first letter are proper nouns. Translate the transliterated Akkadian into fluent English."
    },
    {
      "role": "user",
      "content": "Translate: um-ma A-šùr-na-da ma-la a-Hi-a"
    },
    {
      "role": "assistant",
      "content": "Thus (says) Aššur-nādā: as much as my brother"
    }
  ]
}
```

Output: `data/processed/train.jsonl`, `data/processed/val.jsonl`.

**Validation split:** Hold out ~10% of **complete documents** (not random
sentences) so there's no leakage. This gives ~150 docs for validation.

### A3. Training (`train.py`)

QLoRA fine-tuning with PEFT + TRL (not Unsloth — avoid MPS compatibility issues):

```python
from peft import LoraConfig, get_peft_model, TaskType
from trl import SFTTrainer, SFTConfig
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from datasets import load_dataset
import yaml, sys

def main():
    cfg = yaml.safe_load(open(sys.argv[1]))

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype="bfloat16",
    )

    model = AutoModelForCausalLM.from_pretrained(
        cfg["model_name"],
        quantization_config=bnb_config,
        device_map="auto",
    )
    tokenizer = AutoTokenizer.from_pretrained(cfg["model_name"])

    lora_config = LoraConfig(
        r=cfg.get("lora_r", 32),
        lora_alpha=cfg.get("lora_alpha", 64),
        target_modules=[
            "q_proj", "k_proj", "v_proj", "o_proj",
            "gate_proj", "up_proj", "down_proj",
        ],
        lora_dropout=0.05,
        task_type=TaskType.CAUSAL_LM,
    )
    model = get_peft_model(model, lora_config)

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

**Config: `configs/local.yaml`** — Mac M3 quick iteration:
```yaml
model_name: "Qwen/Qwen3-1.5B-Instruct-2507"
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

**Config: `configs/cloud.yaml`** — full CUDA training:
```yaml
model_name: "Qwen/Qwen3-1.5B-Instruct-2507"
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

### A4. Inference (`inference.py`)

Batched inference with Qwen3 thinking mode explicitly disabled:

```python
def translate_batch(model, tokenizer, texts, cfg):
    """Batched translation with left-padding for efficiency."""
    tokenizer.padding_side = "left"
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    results = []
    batch_size = cfg.get("inference_batch_size", 8)

    # Sort by length for better batching, track original indices
    indexed = sorted(enumerate(texts), key=lambda x: len(x[1]))

    for i in range(0, len(indexed), batch_size):
        batch = indexed[i:i + batch_size]
        prompts = []
        for _, text in batch:
            messages = [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": f"Translate: {text}"},
            ]
            # Disable Qwen3 thinking mode
            prompt = tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
                enable_thinking=False,
            )
            prompts.append(prompt)

        inputs = tokenizer(
            prompts, return_tensors="pt", padding=True, truncation=True
        ).to(model.device)

        outputs = model.generate(
            **inputs,
            max_new_tokens=cfg.get("max_new_tokens", 256),
            do_sample=False,      # greedy for speed + reproducibility
            temperature=None,
            top_p=None,
        )

        for j, (orig_idx, _) in enumerate(batch):
            decoded = tokenizer.decode(
                outputs[j][inputs["input_ids"].shape[1]:],
                skip_special_tokens=True,
            )
            results.append((orig_idx, decoded))

    # Restore original order
    results.sort(key=lambda x: x[0])
    return [r[1] for r in results]
```

### A5. Post-processing (`postprocess.py`)

Clean model output before writing submission:

```python
import re

def postprocess(text: str) -> str:
    """Clean up a single model output for submission."""
    # Strip any residual thinking tokens or chat template artifacts
    text = re.sub(r"<\|.*?\|>", "", text)
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL)

    # Remove repeated phrases (common LLM failure mode)
    text = remove_repetitions(text)

    # Collapse whitespace, strip
    text = re.sub(r"\s+", " ", text).strip()

    # Ensure single sentence (take first sentence if multiple)
    if not text:
        text = "..."
    return text

def remove_repetitions(text: str) -> str:
    """Remove consecutive duplicate phrases."""
    words = text.split()
    result = []
    i = 0
    while i < len(words):
        # Check for repeated n-grams (length 3-10)
        found_repeat = False
        for n in range(10, 2, -1):
            if i + 2 * n <= len(words):
                chunk = words[i:i+n]
                next_chunk = words[i+n:i+2*n]
                if chunk == next_chunk:
                    result.extend(chunk)
                    i += 2 * n
                    found_repeat = True
                    break
        if not found_repeat:
            result.append(words[i])
            i += 1
    return " ".join(result)
```

### A6. Evaluation (`evaluate.py`)

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

### A7. Submission Pipeline

1. **Export**: Merge LoRA → base model (see `scripts/export_model.sh`)
2. **Upload**: Push merged model to Kaggle Datasets
3. **Notebook**: Load model, preprocess test.csv, generate, postprocess, write `submission.csv`

**How `src/` code gets into the notebook:**

The Kaggle notebook runs offline with no access to the repo or `pip install`.
The submission notebook must be self-contained. Strategy:

1. The notebook **inlines** the needed functions from `src/` directly into
   notebook cells. Only three modules are needed at inference time:
   `preprocess.py`, `inference.py`, and `postprocess.py`. These are small,
   pure-Python files with minimal dependencies.

2. A script `scripts/build_notebook.py` **auto-generates** the submission
   notebook by reading the source files and inserting them as cells. This
   avoids manual copy-paste drift. Run it before each Kaggle submission:

   ```bash
   uv run python scripts/build_notebook.py    # → notebooks/submission.ipynb
   ```

3. The merged model and any data files (e.g., lexicon CSV for proper noun
   correction) are uploaded as a **Kaggle Dataset** and accessed via
   `/kaggle/input/...`.

The generated notebook skeleton:

```python
# Cell 1: Inline src/preprocess.py (cleaning functions)
# Cell 2: Inline src/inference.py (batched translation)
# Cell 3: Inline src/postprocess.py (output cleaning)
# Cell 4: Load merged Qwen3-1.5B from /kaggle/input/...
# Cell 5: Load + preprocess test.csv
# Cell 6: Generate translations (batched, greedy, thinking disabled)
# Cell 7: Post-process all outputs
# Cell 8: Write submission.csv
```

At 1.5B parameters with 4-bit quantization and greedy decoding, expect ~1-3
seconds per translation → 4000 translations in ~1-3 hours, well within the 9h
budget. This leaves room for ensembling later.

---

## 5. Part B — Iterative Improvements

Each stage below is independent and can be pursued in any order based on
expected impact. Each produces a new model version to submit.

### B1. Lexicon Integration (`lexicon.py`)

The competition provides `OA_Lexicon_eBL.csv` (all Old Assyrian words with
dictionary entries) and `eBL_Dictionary.csv` (full Akkadian dictionary). This is
free structured knowledge that the baseline ignores entirely.

**Proper noun table:** Build a lookup from transliterated form → normalized
English form for all entries where `type` is `PN` (person name), `GN`
(geographic name), `DN` (divine name), etc. Use this in post-processing to
correct proper noun spelling and capitalization in model output.

**Prompt-time lexicon injection:** For each test transliteration, look up rare
or ambiguous words in the lexicon and append their dictionary meanings to the
user prompt:

```json
{
  "role": "user",
  "content": "Translate: um-ma A-šùr-na-da ma-la a-Hi-a\n\nLexicon hints:\n- um-ma: saying, thus\n- ma-la: as much as\n- a-Hi: brother"
}
```

This is retrieval-augmented generation for free — no model retraining needed,
just an inference-time enhancement. Can be tested immediately on the val set.

**Synthetic lexicon pairs:** Create word-level and phrase-level training pairs
from lexicon entries as a form of curriculum learning. Train the model on these
first (or mix them in) so it learns the basic vocabulary before tackling full
sentences.

### B2. Sentence Alignment

Training data is document-level (~1500 docs), test data is sentence-level
(~4000 sentences from ~400 docs). Two complementary strategies:

**Strategy A — Document context at inference:** Keep training on doc-level
pairs. At inference, the test data groups sentences by `text_id` and orders them
by `line_start`/`line_end`. For each sentence, include the surrounding
transliteration lines as context in the prompt:

```json
{
  "role": "user",
  "content": "[Context: preceding lines...]\nTranslate this sentence: [target lines]\n[Context: following lines...]"
}
```

This lets the model leverage document context without requiring aligned
sentence-level training data.

**Strategy B — Sentence-level training pairs:** Use
`Sentences_Oare_FirstWord_LinNum.csv` to split transliterations at sentence
boundaries. For the English side, split on sentence-ending punctuation (`.`,
`!`, `?`) and align by:
1. Matching sentence counts between transliteration and translation
2. Using length ratios as a soft constraint
3. Anchoring on proper nouns that appear in both sides

Only use high-confidence alignments (where sentence counts match and proper
nouns anchor correctly). Discard ambiguous cases rather than introducing noise.
Validate on a manually-checked sample of ~50 documents.

### B3. Improved System Prompt & Few-Shot

The baseline system prompt is minimal. Improve it by:

- Including a brief glossary of common determinatives and their meanings
- Adding 2-3 example translation pairs directly in the system prompt (few-shot)
- Describing the expected output style (academic translation conventions)

Test variants on the validation set to measure impact:

```
Variant 1: Baseline prompt (current)
Variant 2: + determinative glossary
Variant 3: + 2 few-shot examples
Variant 4: + lexicon hints (from B1)
Variant 5: Combined best
```

### B4. Hyperparameter Sweeps

Sweep these on the cloud, evaluating each on the val set:

| Parameter | Values to try |
|---|---|
| LoRA rank (r) | 16, 32, 64 |
| Learning rate | 1e-4, 2e-4, 5e-4 |
| Epochs | 3, 5, 8 (watch val loss for overfitting) |
| Max sequence length | 1024, 2048 |
| LoRA alpha | 2×r (standard), 4×r |

Track everything in W&B. Pick the best by geomean on the val set.

### B5. Ensembling

With Qwen3-1.5B, inference for 4000 translations takes ~1-3 hours. This leaves
budget for multiple passes within the 9h Kaggle limit.

**MBR Decoding (Minimum Bayes Risk):** For each input, generate N translations
(e.g., N=5 with different temperatures or from different checkpoints). Score
each candidate against all others using chrF++. Pick the candidate with the
highest average similarity to the others — this is the one most likely to be
correct.

**Multi-checkpoint:** Save checkpoints at epochs 3, 5, 8. Generate from each,
then select via MBR.

**Multi-seed:** Train 2-3 models with different random seeds. Generate from
each, select via MBR.

### B6. Proper Noun Post-Processing

The competition explicitly warns that "proper nouns in general are where most ML
tasks underperform." Build a targeted fix:

1. Extract all proper nouns from training translations (capitalized words)
2. Cross-reference with `OA_Lexicon_eBL.csv` (type = PN, GN, DN, etc.)
3. Build a fuzzy matching table: transliterated form → canonical English form
4. In post-processing, find proper nouns in model output and replace with
   canonical forms when fuzzy match confidence is high

---

## 6. Helper Scripts

### `scripts/download_data.sh`

```bash
#!/usr/bin/env bash
set -euo pipefail

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
```

**`.env` file** (local, `.gitignore`'d — create once on the dev machine):

```bash
WANDB_API_KEY=your-key-here
KAGGLE_USERNAME=your-username
KAGGLE_KEY=your-kaggle-api-key
```

### `scripts/sync_up.sh`

```bash
#!/usr/bin/env bash
# Sync code, data, and .env (for WANDB_API_KEY etc.) to cloud machine.
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

Note: `.env` is synced intentionally so that `WANDB_API_KEY` and
`KAGGLE_USERNAME`/`KAGGLE_KEY` are available on the cloud machine. The
`.env` file is `.gitignore`'d so it never enters version control.

### `scripts/sync_down.sh`

```bash
#!/usr/bin/env bash
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
set -euo pipefail

REMOTE="${1:?Usage: run_remote.sh user@host [config]}"
CONFIG="${2:-configs/cloud.yaml}"
SESSION="training"

echo "Starting training on ${REMOTE} with ${CONFIG}..."
ssh "${REMOTE}" "bash -lc '
    cd ~/deep-past
    tmux kill-session -t ${SESSION} 2>/dev/null || true
    tmux new-session -d -s ${SESSION} \
        \"source \\\"$HOME/.local/bin/env\\\" && set -a && source .env && set +a && uv run python src/train.py ${CONFIG} 2>&1 | tee outputs/train.log\"
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
    'Qwen/Qwen3-1.5B-Instruct-2507', torch_dtype='auto', device_map='cpu'
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

## 7. Development Workflow

```
┌───────────────────────────────────────────────────────────────┐
│  MacBook Pro M3                                               │
│                                                               │
│  1. Edit code in src/                                         │
│  2. uv run python src/preprocess.py            # build data   │
│  3. uv run python src/train.py configs/local.yaml  # smoke    │
│  4. uv run python src/inference.py ...         # sanity check │
│  5. git commit + push                                         │
│                                                               │
│  6. ./scripts/sync_up.sh user@gpu-box   # code + data + .env │
│  7. ./scripts/run_remote.sh user@gpu-box                      │
│     ↕ (monitor via wandb dashboard or tmux attach)            │
│  8. ./scripts/sync_down.sh user@gpu-box                       │
│                                                               │
│  9. uv run python src/evaluate.py              # score        │
│ 10. uv run python src/submit.py                # CSV          │
│                                                               │
│ 11. ./scripts/export_model.sh                  # merge LoRA   │
│ 12. ./scripts/upload_kaggle_dataset.sh user/model             │
│ 13. uv run python scripts/build_notebook.py    # gen notebook │
│ 14. Submit notebook on Kaggle                                 │
└───────────────────────────────────────────────────────────────┘
```

---

## 8. Risk Assessment & Mitigations

| Risk | Impact | Mitigation |
|---|---|---|
| 18 GB RAM too tight for QLoRA | Training OOMs on Mac | 1.5B is much easier on RAM. Use `max_seq_length=512`, `batch_size=1`. Fallback: CPU-only dry runs locally, train exclusively on cloud. |
| PEFT/TRL MPS issues | Can't fine-tune locally | Use MLX-LM LoRA as fallback for local dev. Cloud for real training. |
| Qwen3 thinking mode wastes tokens | Slow inference, garbled output | Explicitly disable via `enable_thinking=False` in `apply_chat_template`. Strip `<think>` tags in post-processing as safety net. |
| Kaggle notebook 9h limit | Not enough time for inference | 1.5B at 4-bit with greedy decoding → ~1-3h for 4000 translations. Plenty of headroom for ensembling. |
| Sentence alignment is imperfect | Misaligned pairs degrade training | Start by training on doc-level pairs (no alignment needed). Add sentence-level pairs only when high-confidence. |
| Proper nouns misspelled by model | Hurts both BLEU and chrF++ | Post-process with lexicon-based proper noun correction (B6). |
