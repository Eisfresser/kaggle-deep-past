"""A3. QLoRA fine-tuning entry point.

Usage:
    uv run python src/train.py configs/local.yaml
    uv run python src/train.py configs/cloud.yaml
"""

import sys
from pathlib import Path

import yaml
from datasets import load_dataset
from peft import LoraConfig, TaskType, get_peft_model
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from trl import SFTConfig, SFTTrainer


def main():
    if len(sys.argv) < 2:
        print("Usage: python src/train.py <config.yaml>")
        sys.exit(1)

    cfg = yaml.safe_load(open(sys.argv[1]))

    model_name = cfg["model_name"]
    print(f"=== Training with {model_name} ===")
    print(f"Config: {sys.argv[1]}")

    # Quantization config
    bnb_config = None
    if cfg.get("load_in_4bit", False):
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype="bfloat16",
        )

    # Load model
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        quantization_config=bnb_config,
        device_map="auto",
        torch_dtype="auto",
    )
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # LoRA config
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
    model.print_trainable_parameters()

    # Load dataset
    train_data = cfg["train_data"]
    dataset = load_dataset("json", data_files=train_data, split="train")
    print(f"Training on {len(dataset)} examples from {train_data}")

    # Eval dataset (optional)
    eval_dataset = None
    val_data = cfg.get("val_data")
    if val_data and Path(val_data).exists():
        eval_dataset = load_dataset("json", data_files=val_data, split="train")
        print(f"Evaluating on {len(eval_dataset)} examples from {val_data}")

    # Output dir
    output_dir = cfg["output_dir"]
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    # Training args
    max_seq_length = cfg.get("max_seq_length", 2048)
    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=dataset,
        eval_dataset=eval_dataset,
        args=SFTConfig(
            output_dir=output_dir,
            max_seq_length=max_seq_length,
            per_device_train_batch_size=cfg.get("batch_size", 2),
            gradient_accumulation_steps=cfg.get("grad_accum", 4),
            num_train_epochs=cfg.get("epochs", 3),
            learning_rate=cfg.get("lr", 2e-4),
            warmup_ratio=0.1,
            logging_steps=10,
            save_strategy="epoch",
            eval_strategy="epoch" if eval_dataset is not None else "no",
            bf16=True,
            report_to=cfg.get("report_to", "wandb"),
            run_name=cfg.get("run_name", "deep-past"),
        ),
    )

    trainer.train()

    # Save final checkpoint
    final_dir = f"{output_dir}/final"
    model.save_pretrained(final_dir)
    tokenizer.save_pretrained(final_dir)
    print(f"Final model saved to {final_dir}")


if __name__ == "__main__":
    main()
