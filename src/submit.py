"""A7. Produce submission.csv from predictions.

Reads test predictions, applies post-processing, and writes a submission
file in the format expected by the Kaggle competition.

Usage:
    uv run python src/submit.py
    uv run python src/submit.py outputs/predictions/test_lora.csv
"""

import sys
from pathlib import Path

import pandas as pd

# Import postprocess from the same src/ directory
sys.path.insert(0, str(Path(__file__).parent))
from postprocess import postprocess


def main():
    pred_dir = Path("outputs/predictions")
    sub_dir = Path("outputs/submissions")
    sub_dir.mkdir(parents=True, exist_ok=True)

    if len(sys.argv) >= 2:
        pred_path = Path(sys.argv[1])
    else:
        # Auto-find the most recent test prediction file
        candidates = sorted(pred_dir.glob("test_*.csv"), key=lambda p: p.stat().st_mtime)
        if not candidates:
            print("ERROR: No test prediction files found in outputs/predictions/")
            print("Run: uv run python src/inference.py configs/local.yaml")
            sys.exit(1)
        pred_path = candidates[-1]

    print(f"=== Building submission from {pred_path} ===")

    # Load predictions
    pred_df = pd.read_csv(pred_path)
    pred_col = "prediction_clean" if "prediction_clean" in pred_df.columns else "prediction"
    if pred_col not in pred_df.columns:
        print(f"ERROR: No prediction column found in {pred_path}")
        sys.exit(1)

    # Load original test.csv for IDs
    test_path = Path("data/processed/test_clean.csv")
    if not test_path.exists():
        test_path = Path("data/raw/test.csv")
    if not test_path.exists():
        print("ERROR: Cannot find test.csv for IDs")
        sys.exit(1)

    test_df = pd.read_csv(test_path)

    # Find ID column
    cols = {c.lower(): c for c in test_df.columns}
    id_col = cols.get("id", cols.get("text_id", ""))
    if not id_col:
        print(f"WARNING: No id column found in {test_path}, using row index")
        test_df["id"] = range(len(test_df))
        id_col = "id"

    # Post-process predictions
    predictions = pred_df[pred_col].apply(postprocess).tolist()

    if len(predictions) != len(test_df):
        print(
            f"WARNING: Prediction count ({len(predictions)}) != "
            f"test count ({len(test_df)})"
        )

    # Build submission
    submission = pd.DataFrame({
        "id": test_df[id_col].values[: len(predictions)],
        "translation": predictions,
    })

    out_path = sub_dir / "submission.csv"
    submission.to_csv(out_path, index=False)
    print(f"  Submission saved → {out_path}")
    print(f"  {len(submission)} rows")


if __name__ == "__main__":
    main()
