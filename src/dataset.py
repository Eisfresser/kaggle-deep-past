"""A2. HF Dataset construction + chat template formatting.

Reads cleaned CSVs from data/processed/, builds JSONL files formatted as
instruction-tuned chat messages for SFT training.

Usage:
    uv run python src/dataset.py
"""

import json
import random
from pathlib import Path

import pandas as pd

PROC_DIR = Path("data/processed")

SYSTEM_PROMPT = (
    "You are an expert translator of Old Assyrian Akkadian cuneiform texts "
    "into English. Determinatives in curly brackets classify nouns: "
    "{d} = deity, {ki} = place, {m} = masculine name, {mi} = feminine name. "
    "Words in ALL CAPS are Sumerian logograms. Words with a capitalized first "
    "letter are proper nouns. Translate the transliterated Akkadian into "
    "fluent English."
)

VAL_FRACTION = 0.10  # hold out ~10% of documents for validation
SEED = 42


def build_message(transliteration: str, translation: str) -> dict:
    """Build a single chat-format training example."""
    return {
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": f"Translate: {transliteration}"},
            {"role": "assistant", "content": translation},
        ]
    }


def write_jsonl(records: list[dict], path: Path):
    """Write a list of dicts as JSONL."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for rec in records:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
    print(f"  Wrote {len(records)} examples → {path}")


def main():
    print("=== Building datasets ===")

    train_path = PROC_DIR / "train_clean.csv"
    if not train_path.exists():
        print(f"ERROR: {train_path} not found. Run src/preprocess.py first.")
        return

    df = pd.read_csv(train_path)

    # Find the right column names
    cols = {c.lower(): c for c in df.columns}
    trans_col = "transliteration_clean"
    target_col = "translation_clean"

    if trans_col not in df.columns or target_col not in df.columns:
        raise ValueError(
            f"Expected columns {trans_col}, {target_col}. "
            f"Found: {list(df.columns)}"
        )

    # Split by document (text_id) to prevent leakage
    id_col = cols.get("text_id", cols.get("id", ""))

    if id_col and id_col in df.columns:
        # Group by document, split at document level
        doc_ids = df[id_col].unique().tolist()
        random.seed(SEED)
        random.shuffle(doc_ids)
        n_val = max(1, int(len(doc_ids) * VAL_FRACTION))
        val_ids = set(doc_ids[:n_val])
        train_ids = set(doc_ids[n_val:])

        train_df = df[df[id_col].isin(train_ids)]
        val_df = df[df[id_col].isin(val_ids)]
        print(f"  Split by {id_col}: {len(train_ids)} train docs, {len(val_ids)} val docs")
    else:
        # No document ID column — random row-level split
        print("  WARNING: No text_id/id column found, using random row split")
        df_shuffled = df.sample(frac=1, random_state=SEED)
        n_val = max(1, int(len(df_shuffled) * VAL_FRACTION))
        val_df = df_shuffled.iloc[:n_val]
        train_df = df_shuffled.iloc[n_val:]

    # Build chat-format examples
    train_records = [
        build_message(row[trans_col], row[target_col])
        for _, row in train_df.iterrows()
    ]
    val_records = [
        build_message(row[trans_col], row[target_col])
        for _, row in val_df.iterrows()
    ]

    # Write full datasets
    write_jsonl(train_records, PROC_DIR / "train_full.jsonl")
    write_jsonl(val_records, PROC_DIR / "val.jsonl")

    # Write a small subset for local smoke testing
    small_n = min(100, len(train_records))
    write_jsonl(train_records[:small_n], PROC_DIR / "train_small.jsonl")
    print(f"  Wrote {small_n}-example subset → {PROC_DIR / 'train_small.jsonl'}")

    print(f"\n  Train: {len(train_records)} examples")
    print(f"  Val:   {len(val_records)} examples")
    print("Done.")


if __name__ == "__main__":
    main()
