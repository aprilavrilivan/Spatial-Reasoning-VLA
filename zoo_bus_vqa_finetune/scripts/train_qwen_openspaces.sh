#!/usr/bin/env bash
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
OUTPUT_DIR="${1:-${PROJECT_ROOT}/outputs/qwen_openspaces/default_run}"

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
WANDB_PROJECT="${WANDB_PROJECT:-openspaces-qwen-finetune}"

cd "${PROJECT_ROOT}"

"${PYTHON_BIN}" src/train_qwen_openspaces.py \
  --output_dir "${OUTPUT_DIR}" \
  --dataset_name "remyxai/OpenSpaces" \
  --model_name "Qwen/Qwen3-VL-4B-Instruct" \
  --train_split "train" \
  --test_split "test" \
  --eval_fraction 0.1 \
  --max_eval_examples 1024 \
  --max_test_examples 1024 \
  --max_seq_length 2048 \
  --max_new_tokens_eval 64 \
  --image_min_pixels 200704 \
  --image_max_pixels 602112 \
  --epochs 3 \
  --max_steps 2949 \
  --lr 5e-6 \
  --max_grad_norm 0.3 \
  --warmup_steps 300 \
  --per_device_train_batch_size 16 \
  --per_device_eval_batch_size 64 \
  --gradient_accumulation_steps 4 \
  --eval_steps 300 \
  --save_steps 300 \
  --logging_steps 10 \
  --lora_r 32 \
  --lora_alpha 64 \
  --lora_dropout 0.05 \
  --report_to "${REPORT_TO}" \
  --wandb_project "${WANDB_PROJECT}" \
  --skip_pre_finetune_baseline
