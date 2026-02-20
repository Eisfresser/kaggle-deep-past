#!/usr/bin/env bash
set -euo pipefail

CHECKPOINT="${1:-outputs/checkpoints/cloud/final}"
OUTPUT="${2:-outputs/merged}"

echo "Merging LoRA from ${CHECKPOINT} → ${OUTPUT}"
uv run python -c "
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

base = AutoModelForCausalLM.from_pretrained(
    'Qwen/Qwen3-1.5B-Instruct-2507', torch_dtype='auto', device_map='cpu'
)
model = PeftModel.from_pretrained(base, '${CHECKPOINT}')
merged = model.merge_and_unload()
merged.save_pretrained('${OUTPUT}')

tokenizer = AutoTokenizer.from_pretrained('${CHECKPOINT}')
tokenizer.save_pretrained('${OUTPUT}')
print('Merged model saved to ${OUTPUT}')
"
