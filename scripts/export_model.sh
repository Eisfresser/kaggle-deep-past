#!/usr/bin/env bash
set -euo pipefail

CHECKPOINT="${1:-outputs/checkpoints/cloud/final}"
OUTPUT="${2:-outputs/merged}"

# Load HF_TOKEN from .env if present
ENV_FILE="$(cd "$(dirname "$0")/.." && pwd)/.env"
if [ -f "${ENV_FILE}" ]; then
    HF_TOKEN=$(grep -E '^HF_TOKEN=' "${ENV_FILE}" | cut -d= -f2- | tr -d '[:space:]')
    export HF_TOKEN
fi

echo "Merging LoRA from ${CHECKPOINT} → ${OUTPUT}"
uv run python -c "
import json, pathlib
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

adapter_cfg = json.loads((pathlib.Path('${CHECKPOINT}') / 'adapter_config.json').read_text())
base_model = adapter_cfg['base_model_name_or_path']
print(f'Base model: {base_model}')

base = AutoModelForCausalLM.from_pretrained(base_model, torch_dtype='auto', device_map='cpu')
model = PeftModel.from_pretrained(base, '${CHECKPOINT}')
merged = model.merge_and_unload()
merged.save_pretrained('${OUTPUT}')

tokenizer = AutoTokenizer.from_pretrained('${CHECKPOINT}')
tokenizer.save_pretrained('${OUTPUT}')
print('Merged model saved to ${OUTPUT}')
"
