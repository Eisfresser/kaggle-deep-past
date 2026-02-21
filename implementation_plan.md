# Deep Past Challenge — Implementation Plan

**Goal:** Fine-tune Qwen3-1.5B-Instruct-2507 for Akkadian → English translation
**Dev:** MacBook Pro M3 18 GB · **Train:** RunPod cloud GPU
**Tooling:** `uv`, public GitHub repo, SSH-based RunPod workflow
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
│   ├── setup_cloud.sh              # bootstrap RunPod machine
│   ├── sync_up.sh                  # rsync code + data to RunPod
│   ├── sync_down.sh                # rsync checkpoints + logs from RunPod
│   ├── run_remote.sh               # SSH into RunPod, run training
│   ├── watch_remote.sh             # poll for completion, sync down, shutdown
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
```

---

## 4. Part A — First Submission (End-to-End Baseline)

The goal is to get a score on the leaderboard as fast as possible. Train on
document-level pairs directly (no sentence alignment yet), submit, then iterate.

### A0. Zero-Shot Baseline (no training)

Running inference with `--no-lora` flag skips loading a LoRA. 


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

### A4. Inference (`inference.py`)

Batched inference with Qwen3 thinking mode explicitly disabled:


### A5. Post-processing (`postprocess.py`)

Clean model output before writing submission:

### A6. Evaluation (`evaluate.py`)

Score against validation set using SacreBLEU:


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


--- 

## Results of Part A

Implementation of part A yielded these results.

| Config | BLEU | chrF++ | Geomean | Kaggle Score |
|---|---|---|---|---|
| Zero-shot (no training) | 0.03 | 6.56 | 0.42 |  |
| qwen3-1.7b_r64-a128_lr2e-4_ep5 | 29.53 | 51.27 | 38.91 | 26.5 |
| qwen3-1.7b_r32-a64_lr5e-5_ep5 | 12.75 | 33.02 | 20.52 |  |
| qwen3-1.7b_r32-a64_lr2e-4_ep5 | 27.99 | 49.42 | 37.20 |  |
| qwen3-1.7b_r16-a32_lr1e-4_ep5 | 19.19 | 40.29 | 27.81 |


---

## 5. Part B — Iterative Improvements

Each stage below is independent and can be pursued in any order based on
expected impact. Each produces a new model version to submit.

### B1. Improved System Prompt & Few-Shot

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

### B2. Lexicon Integration (`lexicon.py`)

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

### B3. Sentence Alignment Strategy A

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


---

## Results of Part B

Implementation of part B1, B2 and B3 (without new fine-tuning) yielded these results.

| Config | BLEU | chrF++ | Geomean | Kaggle Score |
|---|---|---|---|---|
| Zero-shot (no training) | 0.03 | 6.56 | 0.42 |  |
| qwen3-1.7b_r64-a128_lr2e-4_ep5 part A | 29.53 | 51.27 | 38.91 | 26.5 |
| qwen3-1.7b_r64-a128_lr2e-4_ep5 part B1,2,3 |  |  |  | 19.6 |

Changes in part B contributed negatively and will summarily be undone. 

---

## 6. Part C - More Improvements

### C1. Sentence Alignment Strategy B

**Strategy B — Sentence-level training pairs:** Use
`Sentences_Oare_FirstWord_LinNum.csv` to split transliterations at sentence
boundaries. For the English side, split on sentence-ending punctuation (`.`,
`!`, `?`) and align by:
1. Matching sentence counts between transliteration and translation
2. Using length ratios as a soft constraint
3. Anchoring on proper nouns that appear in both sides

Only use high-confidence alignments (where sentence counts match and proper
nouns anchor correctly). Discard ambiguous cases rather than introducing noise.

Validatation:
- prepare sample of ~50 documents for manual checking
- present a notebook with visual data analysis of how well sentense splitting worked. 
- create an excel with complete original and translated sentenses plus some reference columns for visual inspection

### C2. Ensembling

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

### C3. Proper Noun Post-Processing

The competition explicitly warns that "proper nouns in general are where most ML
tasks underperform." Build a targeted fix:

1. Extract all proper nouns from training translations (capitalized words)
2. Cross-reference with `OA_Lexicon_eBL.csv` (type = PN, GN, DN, etc.)
3. Build a fuzzy matching table: transliterated form → canonical English form
4. In post-processing, find proper nouns in model output and replace with
   canonical forms when fuzzy match confidence is high

---
