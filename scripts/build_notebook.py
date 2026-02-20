"""Auto-generate notebooks/submission.ipynb from src/ modules.

Inlines preprocess.py, inference.py, and postprocess.py into notebook cells
so the Kaggle notebook is self-contained (no pip install, no internet).

Usage:
    uv run python scripts/build_notebook.py
"""

import json
from pathlib import Path

SRC_DIR = Path("src")
OUT_PATH = Path("notebooks/submission.ipynb")

# Kaggle dataset path where the merged model is uploaded
MODEL_DATASET = "/kaggle/input/deep-past-model"


def read_source(filename: str, drop_funcs: set[str] | None = None,
                drop_imports: set[str] | None = None) -> str:
    """Read a source file, stripping docstring, __main__, and optionally
    specific top-level functions and import lines."""
    path = SRC_DIR / filename
    content = path.read_text()

    lines = content.split("\n")
    result = []
    skip_main = False
    skip_func = False
    in_docstring = False
    docstring_done = False
    drop_funcs = drop_funcs or set()
    drop_imports = drop_imports or set()

    for line in lines:
        # Skip module-level docstrings (triple-quoted at the top)
        if not docstring_done:
            if line.strip().startswith('"""'):
                if in_docstring:
                    in_docstring = False
                    docstring_done = True
                    continue
                elif line.strip().endswith('"""') and line.strip() != '"""':
                    docstring_done = True
                    continue
                else:
                    in_docstring = True
                    continue
            if in_docstring:
                continue
            if not line.strip():
                continue
            docstring_done = True

        # Skip if __name__ == "__main__" block and everything after
        if line.strip().startswith('if __name__'):
            skip_main = True
            continue
        if skip_main:
            continue

        # Skip top-level functions by name (and their entire body)
        if any(line.startswith(f"def {fn}(") for fn in drop_funcs):
            skip_func = True
            continue
        if skip_func:
            if line and not line[0].isspace() and line.strip():
                skip_func = False
            else:
                continue

        # Skip specific imports
        if drop_imports and any(tok in line for tok in drop_imports):
            continue

        # Skip sys.path manipulation
        if "sys.path.insert" in line:
            continue

        result.append(line)

    # Remove trailing empty lines
    while result and not result[-1].strip():
        result.pop()

    return "\n".join(result)


def make_cell(cell_type: str, source: str) -> dict:
    """Create a notebook cell."""
    cell = {
        "cell_type": cell_type,
        "metadata": {},
        "source": source.split("\n"),
    }
    if cell_type == "code":
        cell["execution_count"] = None
        cell["outputs"] = []
    return cell


def build_notebook():
    """Build the submission notebook."""
    cells = []

    # Cell 0: Title
    cells.append(make_cell("markdown", "# Deep Past Challenge — Akkadian → English\n\nAuto-generated submission notebook. Do not edit manually.\nRegenerate with: `uv run python scripts/build_notebook.py`"))

    # Cell 1: Inline preprocess.py (only cleaning functions needed)
    cells.append(make_cell("markdown", "## Preprocessing (from src/preprocess.py)"))
    cells.append(make_cell("code", read_source(
        "preprocess.py",
        drop_funcs={"main", "process_train", "process_test"},
        drop_imports={"from pathlib", "import pandas", "RAW_DIR", "PROC_DIR"},
    )))

    # Cell 2: Inline postprocess.py
    cells.append(make_cell("markdown", "## Post-processing (from src/postprocess.py)"))
    cells.append(make_cell("code", read_source(
        "postprocess.py",
        drop_funcs={"main"},
        drop_imports={"import re", "import sys", "from pathlib", "import pandas"},
    )))

    # Cell 3: Load model (before inference cell so torch is available)
    cells.append(make_cell("markdown", "## Load Model"))
    cells.append(make_cell("code", f"""import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL_PATH = "{MODEL_DATASET}"

tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
model = AutoModelForCausalLM.from_pretrained(
    MODEL_PATH,
    torch_dtype=torch.float16,
    device_map="auto",
)
model.eval()
print(f"Model loaded from {{MODEL_PATH}}")"""))

    # Cell 4: Inline inference functions (only translate_batch + SYSTEM_PROMPT)
    cells.append(make_cell("markdown", "## Inference (from src/inference.py)"))
    cells.append(make_cell("code", read_source(
        "inference.py",
        drop_funcs={"main", "load_model"},
        drop_imports={"import argparse", "import sys", "from pathlib",
                      "import pandas", "import torch", "import yaml",
                      "from dotenv", "load_dotenv", "from peft",
                      "BitsAndBytesConfig", "from transformers"},
    )))

    # Cell 5: Load + preprocess test data
    cells.append(make_cell("markdown", "## Load & Preprocess Test Data"))
    cells.append(make_cell("code", """import pandas as pd

test_df = pd.read_csv("/kaggle/input/deep-past-initiative-machine-translation/test.csv")
print(f"Test set: {len(test_df)} rows")
print(test_df.head())

# Find transliteration column
cols = {c.lower(): c for c in test_df.columns}
trans_col = cols.get("transliteration", cols.get("source", ""))
if not trans_col:
    raise ValueError(f"No transliteration column found: {list(test_df.columns)}")

test_df["transliteration_clean"] = test_df[trans_col].apply(clean_transliteration)
print(f"Cleaned {len(test_df)} transliterations")"""))

    # Cell 6: Generate translations
    cells.append(make_cell("markdown", "## Generate Translations"))
    cells.append(make_cell("code", """cfg = {
    "inference_batch_size": 8,
    "max_new_tokens": 256,
    "max_seq_length": 2048,
}

texts = test_df["transliteration_clean"].tolist()
predictions = translate_batch(model, tokenizer, texts, cfg)
print(f"Generated {len(predictions)} translations")"""))

    # Cell 7: Post-process
    cells.append(make_cell("markdown", "## Post-process"))
    cells.append(make_cell("code", """predictions_clean = [postprocess(p) for p in predictions]
print("Sample predictions:")
for i in range(min(5, len(predictions_clean))):
    print(f"  {i}: {predictions_clean[i][:100]}...")"""))

    # Cell 8: Write submission
    cells.append(make_cell("markdown", "## Write Submission"))
    cells.append(make_cell("code", """# Find ID column
id_col = cols.get("id", cols.get("text_id", ""))
if not id_col:
    test_df["id"] = range(len(test_df))
    id_col = "id"

submission = pd.DataFrame({
    "id": test_df[id_col],
    "translation": predictions_clean,
})

submission.to_csv("submission.csv", index=False)
print(f"Submission saved: {len(submission)} rows")
print(submission.head())"""))

    # Build notebook structure
    notebook = {
        "nbformat": 4,
        "nbformat_minor": 5,
        "metadata": {
            "kernelspec": {
                "display_name": "Python 3",
                "language": "python",
                "name": "python3",
            },
            "language_info": {
                "name": "python",
                "version": "3.11.0",
            },
        },
        "cells": cells,
    }

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT_PATH, "w", encoding="utf-8") as f:
        json.dump(notebook, f, indent=1, ensure_ascii=False)

    print(f"Notebook generated → {OUT_PATH}")
    print(f"  {len(cells)} cells ({sum(1 for c in cells if c['cell_type'] == 'code')} code, "
          f"{sum(1 for c in cells if c['cell_type'] == 'markdown')} markdown)")


if __name__ == "__main__":
    build_notebook()
