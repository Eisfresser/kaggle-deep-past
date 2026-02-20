"""A6. Local BLEU + chrF++ scoring.

Scores predictions against references using SacreBLEU.
Metric: geometric mean of BLEU and chrF++ (competition metric).

Usage:
    uv run python src/evaluate.py                                    # auto-find latest
    uv run python src/evaluate.py outputs/predictions/val_lora.csv   # specific file
"""

import math
import sys
from pathlib import Path

import pandas as pd
import sacrebleu


def score(predictions: list[str], references: list[str]) -> dict:
    """Compute BLEU, chrF++, and their geometric mean."""
    bleu = sacrebleu.corpus_bleu(predictions, [references])
    chrf = sacrebleu.corpus_chrf(predictions, [references], word_order=2)
    geomean = math.sqrt(bleu.score * chrf.score) if bleu.score > 0 and chrf.score > 0 else 0.0
    return {
        "bleu": bleu.score,
        "chrfpp": chrf.score,
        "geomean": geomean,
    }


def main():
    pred_dir = Path("outputs/predictions")

    if len(sys.argv) >= 2:
        pred_path = Path(sys.argv[1])
    else:
        # Auto-find the most recent val prediction file
        candidates = sorted(pred_dir.glob("val_*.csv"), key=lambda p: p.stat().st_mtime)
        if not candidates:
            print("ERROR: No val prediction files found in outputs/predictions/")
            print("Run: uv run python src/inference.py configs/local.yaml --split val")
            sys.exit(1)
        pred_path = candidates[-1]

    print(f"=== Evaluating {pred_path} ===")
    df = pd.read_csv(pred_path)

    # Determine prediction and reference columns
    pred_col = "prediction_clean" if "prediction_clean" in df.columns else "prediction"
    if pred_col not in df.columns:
        print(f"ERROR: No prediction column in {pred_path}")
        sys.exit(1)
    if "reference" not in df.columns:
        print(f"ERROR: No reference column in {pred_path}. Use --split val for inference.")
        sys.exit(1)

    predictions = df[pred_col].fillna("...").tolist()
    references = df["reference"].fillna("").tolist()

    results = score(predictions, references)

    print(f"\n  BLEU:     {results['bleu']:.2f}")
    print(f"  chrF++:   {results['chrfpp']:.2f}")
    print(f"  Geomean:  {results['geomean']:.2f}")
    print()


if __name__ == "__main__":
    main()
