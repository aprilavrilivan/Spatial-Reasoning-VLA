#!/usr/bin/env bash
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
OUTPUT_DIR="${1:-${PROJECT_ROOT}/outputs/smol/default_run}"

if [ -f "${SCRIPT_DIR}/remote_env.sh" ]; then
  source "${SCRIPT_DIR}/remote_env.sh"
fi

DEFAULT_PYTHON="${PROJECT_ROOT}/.venv/bin/python"
if [ -x "${DEFAULT_PYTHON}" ]; then
  PYTHON_BIN="${PYTHON_BIN:-${DEFAULT_PYTHON}}"
else
  PYTHON_BIN="${PYTHON_BIN:-python}"
fi

if ! command -v "${PYTHON_BIN}" >/dev/null 2>&1; then
  PYTHON_BIN="python3"
fi

REPORT_TO="${REPORT_TO:-wandb}"
WANDB_PROJECT="${WANDB_PROJECT:-zoo-bus-vqa-smol-finetune}"

cd "${PROJECT_ROOT}"

"${PYTHON_BIN}" src/train.py \
  --output_dir "${OUTPUT_DIR}" \
  --dataset_name "aprilavrilivan/zoo-bus-vqa" \
  --model_name "HuggingFaceTB/SmolVLM2-2.2B-Instruct" \
  --max_seq_length 2048 \
  --max_new_tokens_eval 16 \
  --epochs 3 \
  --lr 1e-4 \
  --per_device_train_batch_size 32 \
  --per_device_eval_batch_size 80 \
  --gradient_accumulation_steps 2 \
  --eval_steps 300 \
  --save_steps 300 \
  --logging_steps 10 \
  --lora_r 32 \
  --lora_alpha 64 \
  --lora_dropout 0.05 \
  --report_to "${REPORT_TO}" \
  --wandb_project "${WANDB_PROJECT}"
