"""A4. Batched translation generation.

Loads model (optionally with LoRA), translates cleaned test transliterations,
writes predictions to outputs/predictions/.

Usage:
    uv run python src/inference.py configs/local.yaml               # with LoRA
    uv run python src/inference.py configs/local.yaml --no-lora      # base only
"""

import argparse
import sys
from pathlib import Path

import pandas as pd
import torch
import yaml
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

SYSTEM_PROMPT = (
    "You are an expert translator of Old Assyrian Akkadian cuneiform texts "
    "into English. Determinatives in curly brackets classify nouns: "
    "{d} = deity, {ki} = place, {m} = masculine name, {mi} = feminine name. "
    "Words in ALL CAPS are Sumerian logograms. Words with a capitalized first "
    "letter are proper nouns. Translate the transliterated Akkadian into "
    "fluent English."
)


def load_model(cfg: dict, no_lora: bool = False):
    """Load base model, optionally with LoRA adapter."""
    model_name = cfg["model_name"]

    bnb_config = None
    if cfg.get("load_in_4bit", False):
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype="bfloat16",
        )

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        quantization_config=bnb_config,
        device_map="auto",
        torch_dtype="auto",
    )
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    # Load LoRA if available and requested
    if not no_lora:
        lora_path = Path(cfg["output_dir"]) / "final"
        if lora_path.exists():
            print(f"Loading LoRA from {lora_path}")
            model = PeftModel.from_pretrained(model, str(lora_path))
        else:
            print(f"WARNING: LoRA path {lora_path} not found, using base model")

    model.eval()
    return model, tokenizer


def translate_batch(
    model, tokenizer, texts: list[str], cfg: dict
) -> list[str]:
    """Batched translation with left-padding for efficiency."""
    tokenizer.padding_side = "left"
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    results = []
    batch_size = cfg.get("inference_batch_size", 8)

    # Sort by length for better batching, track original indices
    indexed = sorted(enumerate(texts), key=lambda x: len(x[1]))

    for i in range(0, len(indexed), batch_size):
        batch = indexed[i : i + batch_size]
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
            prompts, return_tensors="pt", padding=True, truncation=True,
            max_length=cfg.get("max_seq_length", 2048),
        ).to(model.device)

        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=cfg.get("max_new_tokens", 256),
                do_sample=False,
                temperature=None,
                top_p=None,
            )

        for j, (orig_idx, _) in enumerate(batch):
            decoded = tokenizer.decode(
                outputs[j][inputs["input_ids"].shape[1] :],
                skip_special_tokens=True,
            )
            results.append((orig_idx, decoded))

        done = min(i + batch_size, len(indexed))
        print(f"  Translated {done}/{len(indexed)} examples", end="\r")

    print()

    # Restore original order
    results.sort(key=lambda x: x[0])
    return [r[1] for r in results]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("config", help="Path to config YAML")
    parser.add_argument("--no-lora", action="store_true", help="Skip LoRA, use base model")
    parser.add_argument("--split", default="test", choices=["test", "val"],
                        help="Which split to run inference on")
    args = parser.parse_args()

    cfg = yaml.safe_load(open(args.config))
    print(f"=== Inference ({args.split}) ===")

    # Load data
    if args.split == "val":
        data_path = Path("data/processed/val.jsonl")
        if not data_path.exists():
            print(f"ERROR: {data_path} not found. Run src/dataset.py first.")
            sys.exit(1)
        import json
        records = [json.loads(l) for l in open(data_path)]
        texts = [r["messages"][1]["content"].removeprefix("Translate: ") for r in records]
        references = [r["messages"][2]["content"] for r in records]
    else:
        data_path = Path("data/processed/test_clean.csv")
        if not data_path.exists():
            print(f"ERROR: {data_path} not found. Run src/preprocess.py first.")
            sys.exit(1)
        df = pd.read_csv(data_path)
        texts = df["transliteration_clean"].tolist()
        references = None

    print(f"  {len(texts)} examples from {data_path}")

    # Load model
    model, tokenizer = load_model(cfg, no_lora=args.no_lora)

    # Generate translations
    predictions = translate_batch(model, tokenizer, texts, cfg)

    # Save predictions
    out_dir = Path("outputs/predictions")
    out_dir.mkdir(parents=True, exist_ok=True)

    suffix = "base" if args.no_lora else "lora"
    out_path = out_dir / f"{args.split}_{suffix}.csv"

    out_df = pd.DataFrame({"transliteration": texts, "prediction": predictions})
    if references is not None:
        out_df["reference"] = references
    out_df.to_csv(out_path, index=False)
    print(f"  Predictions saved → {out_path}")


if __name__ == "__main__":
    main()
