"""A5. Clean model output for submission.

Can be used standalone or imported by other modules.

Usage:
    uv run python src/postprocess.py outputs/predictions/test_lora.csv
"""

import re
import sys
from pathlib import Path

import pandas as pd


def postprocess(text: str) -> str:
    """Clean up a single model output for submission."""
    if not isinstance(text, str):
        return "..."

    # Strip any residual thinking tokens or chat template artifacts
    text = re.sub(r"<\|.*?\|>", "", text)
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL)

    # Remove repeated phrases (common LLM failure mode)
    text = remove_repetitions(text)

    # Collapse whitespace, strip
    text = re.sub(r"\s+", " ", text).strip()

    # Ensure non-empty
    if not text:
        text = "..."
    return text


def remove_repetitions(text: str) -> str:
    """Remove consecutive duplicate phrases."""
    words = text.split()
    result = []
    i = 0
    while i < len(words):
        found_repeat = False
        for n in range(10, 2, -1):
            if i + 2 * n <= len(words):
                chunk = words[i : i + n]
                next_chunk = words[i + n : i + 2 * n]
                if chunk == next_chunk:
                    result.extend(chunk)
                    i += 2 * n
                    found_repeat = True
                    break
        if not found_repeat:
            result.append(words[i])
            i += 1
    return " ".join(result)


def main():
    if len(sys.argv) < 2:
        print("Usage: python src/postprocess.py <predictions.csv>")
        sys.exit(1)

    in_path = Path(sys.argv[1])
    df = pd.read_csv(in_path)

    if "prediction" not in df.columns:
        print(f"ERROR: No 'prediction' column in {in_path}")
        sys.exit(1)

    df["prediction_clean"] = df["prediction"].apply(postprocess)

    out_path = in_path.with_name(in_path.stem + "_clean.csv")
    df.to_csv(out_path, index=False)
    print(f"Post-processed {len(df)} predictions → {out_path}")


if __name__ == "__main__":
    main()
