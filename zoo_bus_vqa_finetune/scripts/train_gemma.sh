#!/usr/bin/env bash
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
OUTPUT_DIR="${1:-${PROJECT_ROOT}/outputs/gemma/default_run}"
PYTHON_BIN="${PYTHON_BIN:-python}"

if ! command -v "${PYTHON_BIN}" >/dev/null 2>&1; then
  PYTHON_BIN="python3"
fi

cd "${PROJECT_ROOT}"

"${PYTHON_BIN}" src/train.py \
  --output_dir "${OUTPUT_DIR}" \
  --dataset_name "aprilavrilivan/zoo-bus-vqa" \
  --model_name "google/gemma-3-4b-it" \
  --max_seq_length 512 \
  --max_new_tokens_eval 16 \
  --epochs 3 \
  --lr 1e-4 \
  --per_device_train_batch_size 32 \
  --per_device_eval_batch_size 32 \
  --gradient_accumulation_steps 2 \
  --eval_steps 300 \
  --save_steps 300 \
  --logging_steps 10 \
  --lora_r 32 \
  --lora_alpha 64 \
  --lora_dropout 0.05 \
  --report_to "wandb" \
  --wandb_project "zoo-bus-vqa-gemma-finetune"
