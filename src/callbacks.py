"""Trainer callback for generation-based evaluation (BLEU / chrF++ / geomean).

Runs model.generate() on the val set at the end of each epoch and logs
scores to the active reporter (wandb when configured).
"""

import json

import torch
from transformers import TrainerCallback

from evaluate import score

# Re-use the same system prompt as inference.py
SYSTEM_PROMPT = (
    "You are an expert translator of Old Assyrian Akkadian cuneiform texts "
    "into English. Determinatives in curly brackets classify nouns: "
    "{d} = deity, {ki} = place, {m} = masculine name, {mi} = feminine name. "
    "Words in ALL CAPS are Sumerian logograms. Words with a capitalized first "
    "letter are proper nouns. Translate the transliterated Akkadian into "
    "fluent English."
)


class GenerationEvalCallback(TrainerCallback):
    """Generate translations on the val set after each epoch and log scores."""

    def __init__(self, cfg: dict, tokenizer):
        self.cfg = cfg
        self.tokenizer = tokenizer
        self.texts, self.references = self._load_val(cfg["val_data"])
        # Set by train.py after trainer is created
        self.trainer = None
        print(f"GenerationEvalCallback: {len(self.texts)} val examples loaded")

    # ------------------------------------------------------------------
    @staticmethod
    def _load_val(val_path: str):
        records = [json.loads(line) for line in open(val_path)]
        texts = [r["messages"][1]["content"].removeprefix("Translate: ") for r in records]
        references = [r["messages"][2]["content"] for r in records]
        return texts, references

    # ------------------------------------------------------------------
    def _generate(self, model) -> list[str]:
        """Batched greedy generation, mirrors inference.translate_batch."""
        self.tokenizer.padding_side = "left"
        batch_size = self.cfg.get("inference_batch_size", 8)
        indexed = sorted(enumerate(self.texts), key=lambda x: len(x[1]))
        results = []

        for i in range(0, len(indexed), batch_size):
            batch = indexed[i : i + batch_size]
            prompts = []
            for _, text in batch:
                messages = [
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": f"Translate: {text}"},
                ]
                prompt = self.tokenizer.apply_chat_template(
                    messages,
                    tokenize=False,
                    add_generation_prompt=True,
                    enable_thinking=False,
                )
                prompts.append(prompt)

            inputs = self.tokenizer(
                prompts, return_tensors="pt", padding=True, truncation=True,
                max_length=self.cfg.get("max_seq_length", 2048),
            ).to(model.device)

            with torch.no_grad():
                outputs = model.generate(
                    **inputs,
                    max_new_tokens=self.cfg.get("max_new_tokens", 256),
                    do_sample=False,
                    temperature=None,
                    top_p=None,
                    top_k=None,
                )

            for j, (orig_idx, _) in enumerate(batch):
                decoded = self.tokenizer.decode(
                    outputs[j][inputs["input_ids"].shape[1] :],
                    skip_special_tokens=True,
                )
                results.append((orig_idx, decoded))

        results.sort(key=lambda x: x[0])
        return [r[1] for r in results]

    # ------------------------------------------------------------------
    def on_epoch_end(self, args, state, control, model=None, **kwargs):
        epoch = int(state.epoch)
        print(f"\n=== Generation eval (epoch {epoch}) ===")

        was_training = model.training
        model.eval()

        predictions = self._generate(model)
        results = score(predictions, self.references)

        print(f"  BLEU:    {results['bleu']:.2f}")
        print(f"  chrF++:  {results['chrfpp']:.2f}")
        print(f"  Geomean: {results['geomean']:.2f}")

        # Log via the trainer's built-in logging, which routes to all active
        # reporters (wandb, tensorboard, etc.) based on report_to config.
        if self.trainer is not None:
            self.trainer.log({
                "val_gen/bleu": results["bleu"],
                "val_gen/chrfpp": results["chrfpp"],
                "val_gen/geomean": results["geomean"],
            })

        if was_training:
            model.train()
