"""A3. QLoRA fine-tuning entry point.

Usage:
    uv run python src/train.py configs/local.yaml
    uv run python src/train.py configs/cloud.yaml
    uv run python src/train.py configs/sweep_r16_lr1e-4.yaml
"""

import json
import shutil
import sys
from pathlib import Path

import yaml
from datasets import load_dataset
from dotenv import load_dotenv
from peft import LoraConfig, TaskType, get_peft_model
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from trl.trainer.sft_config import SFTConfig
from trl.trainer.sft_trainer import SFTTrainer

from callbacks import GenerationEvalCallback

load_dotenv()


def make_run_name(cfg: dict) -> str:
    """Build a run name from key config fields, e.g. qwen3-1.5b_r32-a64_lr2e-4_ep5."""
    # Short model tag: "Qwen/Qwen3-1.5B-Instruct-2507" -> "qwen3-1.5b"
    model_tag = cfg["model_name"].split("/")[-1].split("-Instruct")[0].lower()
    r = cfg.get("lora_r", 32)
    a = cfg.get("lora_alpha", 64)
    lr = cfg.get("lr", 2e-4)
    ep = cfg.get("epochs", 3)
    return f"{model_tag}_r{r}-a{a}_lr{lr}_ep{ep}"


def main():
    if len(sys.argv) < 2:
        print("Usage: python src/train.py <config.yaml>")
        sys.exit(1)

    cfg = yaml.safe_load(open(sys.argv[1]))

    # Auto-generate run_name if not explicitly set
    if "run_name" not in cfg:
        cfg["run_name"] = make_run_name(cfg)

    model_name = cfg["model_name"]
    print(f"=== Training with {model_name} ===")
    print(f"Config: {sys.argv[1]}")
    print(f"Run name: {cfg['run_name']}")

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
        tie_word_embeddings=False,
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

    # Output dir — sweep configs use sweep_dir/<run_name>, others use output_dir
    if "sweep_dir" in cfg:
        output_dir = str(Path(cfg["sweep_dir"]) / cfg["run_name"])
    else:
        output_dir = cfg["output_dir"]
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    # Generation-eval callback (requires val_data)
    callbacks = []
    if val_data and Path(val_data).exists():
        gen_eval_cb = GenerationEvalCallback(cfg, tokenizer)
        callbacks.append(gen_eval_cb)

    # Training args
    max_seq_length = cfg.get("max_seq_length", 2048)
    trainer = SFTTrainer(
        model=model,  # type: ignore[arg-type]
        processing_class=tokenizer,
        train_dataset=dataset,
        eval_dataset=eval_dataset,
        callbacks=callbacks,
        args=SFTConfig(
            output_dir=output_dir,
            max_length=max_seq_length,
            per_device_train_batch_size=cfg.get("batch_size", 2),
            per_device_eval_batch_size=cfg.get("eval_batch_size", 1),
            gradient_accumulation_steps=cfg.get("grad_accum", 4),
            num_train_epochs=cfg.get("epochs", 3),
            learning_rate=float(cfg.get("lr", 2e-4)),
            warmup_steps=10,
            logging_steps=10,
            save_strategy="epoch",
            eval_strategy="epoch" if eval_dataset is not None else "no",
            bf16=True,
            dataloader_pin_memory=False,
            report_to=cfg.get("report_to", "wandb"),
            run_name=cfg.get("run_name", "deep-past"),
        ),
    )

    # Give callback access to trainer.log() for wandb/tensorboard reporting
    if callbacks:
        gen_eval_cb.trainer = trainer

    trainer.train()

    # Save final checkpoint
    final_dir = f"{output_dir}/final"
    model.save_pretrained(final_dir)
    tokenizer.save_pretrained(final_dir)
    print(f"Final model saved to {final_dir}")

    # Save config snapshot alongside the model for reproducibility
    shutil.copy2(sys.argv[1], f"{output_dir}/config.yaml")

    # Save final generation-eval scores if callback ran
    if callbacks and gen_eval_cb.last_scores is not None:
        scores_path = Path(output_dir) / "scores.json"
        json.dump(gen_eval_cb.last_scores, open(scores_path, "w"), indent=2)
        print(f"Final scores saved to {scores_path}")


if __name__ == "__main__":
    main()
