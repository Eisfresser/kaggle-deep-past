"""A1. Akkadian transliteration cleaning pipeline.

Reads raw CSVs from data/raw/, cleans transliterations (and lightly cleans
translations), writes cleaned CSVs to data/processed/.

Usage:
    uv run python src/preprocess.py
"""

import re
import unicodedata
from pathlib import Path

import pandas as pd

RAW_DIR = Path("data/raw")
PROC_DIR = Path("data/processed")

# ── Unicode normalization maps ──────────────────────────────────────────────

# Ḫ / ḫ  →  H / h
_SPECIAL_CHARS = str.maketrans({"Ḫ": "H", "ḫ": "h"})

# Unicode subscript / superscript digits → ASCII
_SUB_DIGITS = str.maketrans("₀₁₂₃₄₅₆₇₈₉", "0123456789")
_SUP_DIGITS = str.maketrans("⁰¹²³⁴⁵⁶⁷⁸⁹", "0123456789")

# Half-brackets (damaged text markers)
_HALF_BRACKETS = str.maketrans({"˹": "", "˺": ""})


def clean_transliteration(text: str) -> str:
    """Clean a single transliteration string per competition instructions."""
    if not isinstance(text, str) or not text.strip():
        return ""

    s = text

    # 1. Normalize Unicode (NFC) for consistent handling
    s = unicodedata.normalize("NFC", s)

    # 2. Ḫ / ḫ  →  H / h
    s = s.translate(_SPECIAL_CHARS)

    # 3. Remove half-brackets ˹ ˺ (damaged but readable signs)
    s = s.translate(_HALF_BRACKETS)

    # 4. Handle double angle brackets << >> — remove entirely
    s = re.sub(r"<<.*?>>", "", s)

    # 5. Handle single angle brackets < > — keep text, remove brackets
    s = re.sub(r"<(.*?)>", r"\1", s)

    # 6. Handle square brackets
    #    [... ...] or [...]  →  <big_gap>
    s = re.sub(r"\[\.\.\.\s*\.\.\.?\]", "<big_gap>", s)
    #    [x] or [x x] etc  →  <gap>
    s = re.sub(r"\[x(?:\s+x)*\]", "<gap>", s)
    #    [text]  →  text  (keep content, remove brackets)
    s = re.sub(r"\[(.*?)\]", r"\1", s)

    # 7. Strip scribal notations: ! ? (certainty markers)
    s = re.sub(r"[!?]", "", s)

    # 8. Normalize line dividers: / and : used as line breaks → space
    s = re.sub(r"\s*/\s*", " ", s)
    s = re.sub(r"\s*:\s*", " ", s)

    # 9. Subscript / superscript digits → ASCII
    s = s.translate(_SUB_DIGITS)
    s = s.translate(_SUP_DIGITS)

    # 10. Strip line numbers at start (e.g. "1. " or "1' " or "r. 1 ")
    s = re.sub(r"^(?:(?:o|r|rev|obv|lo\.?e\.?|u\.?e\.?)\.?\s+)?(?:\d+['′]?\.\s*)", "", s)

    # 11. Collapse whitespace
    s = re.sub(r"\s+", " ", s).strip()

    return s


def clean_translation(text: str) -> str:
    """Lightly clean an English translation string."""
    if not isinstance(text, str) or not text.strip():
        return ""

    s = text
    # Just whitespace normalization
    s = re.sub(r"\s+", " ", s).strip()
    return s


def process_train(raw_path: Path, out_path: Path) -> pd.DataFrame:
    """Process train.csv: clean both transliteration and translation."""
    df = pd.read_csv(raw_path)

    # Auto-detect column names (case-insensitive)
    cols = {c.lower(): c for c in df.columns}

    trans_col = cols.get("transliteration", cols.get("source", ""))
    target_col = cols.get("translation", cols.get("target", ""))

    if not trans_col or not target_col:
        raise ValueError(
            f"Cannot find transliteration/translation columns in {raw_path}. "
            f"Found: {list(df.columns)}"
        )

    df["transliteration_clean"] = df[trans_col].apply(clean_transliteration)
    df["translation_clean"] = df[target_col].apply(clean_translation)

    # Drop rows where cleaning produced empty strings
    before = len(df)
    df = df[
        (df["transliteration_clean"].str.len() > 0)
        & (df["translation_clean"].str.len() > 0)
    ].copy()
    after = len(df)
    if before != after:
        print(f"  Dropped {before - after} empty rows ({before} → {after})")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_path, index=False)
    print(f"  Saved {len(df)} rows → {out_path}")
    return df


def process_test(raw_path: Path, out_path: Path) -> pd.DataFrame:
    """Process test.csv: clean transliteration only."""
    df = pd.read_csv(raw_path)

    cols = {c.lower(): c for c in df.columns}
    trans_col = cols.get("transliteration", cols.get("source", ""))
    if not trans_col:
        raise ValueError(
            f"Cannot find transliteration column in {raw_path}. "
            f"Found: {list(df.columns)}"
        )

    df["transliteration_clean"] = df[trans_col].apply(clean_transliteration)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_path, index=False)
    print(f"  Saved {len(df)} rows → {out_path}")
    return df


def main():
    print("=== Preprocessing ===")

    train_raw = RAW_DIR / "train.csv"
    test_raw = RAW_DIR / "test.csv"

    if train_raw.exists():
        print(f"Processing {train_raw}")
        process_train(train_raw, PROC_DIR / "train_clean.csv")
    else:
        print(f"WARNING: {train_raw} not found. Run scripts/download_data.sh first.")

    if test_raw.exists():
        print(f"Processing {test_raw}")
        process_test(test_raw, PROC_DIR / "test_clean.csv")
    else:
        print(f"WARNING: {test_raw} not found. Run scripts/download_data.sh first.")

    print("Done.")


if __name__ == "__main__":
    main()
