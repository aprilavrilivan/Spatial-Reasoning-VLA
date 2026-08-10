#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${PROJECT_ROOT}"

if [ -f "${SCRIPT_DIR}/remote_env.sh" ]; then
  source "${SCRIPT_DIR}/remote_env.sh"
fi
if [ -f ".venv/bin/activate" ]; then
  source .venv/bin/activate
fi

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
REPORT_TO="${REPORT_TO:-none}"
WANDB_PROJECT="${WANDB_PROJECT:-zoo-bus-vqa-robotic-spatial-eval}"
OUTPUT_DIR="${OUTPUT_DIR:-${PROJECT_ROOT}/external_eval/robotic_spatial_results}"
COMMON=(
  --dataset all
  --state both
  --report_to "${REPORT_TO}"
  --wandb_project "${WANDB_PROJECT}"
  --output_dir "${OUTPUT_DIR}"
  --max_new_tokens 128
)

python scripts/evaluate_robotic_spatial.py \
  --model_family smol \
  --model_name HuggingFaceTB/SmolVLM2-2.2B-Instruct \
  --adapter_path outputs/smol/smol_final_20260502_120924/adapter/best_checkpoint \
  --batch_size "${SMOL_BATCH_SIZE:-8}" \
  --max_seq_length 2048 \
  --wandb_run_name smol_robotic_spatial \
  "${COMMON[@]}"

python scripts/evaluate_robotic_spatial.py \
  --model_family qwen \
  --model_name Qwen/Qwen3-VL-4B-Instruct \
  --adapter_path outputs/qwen/qwen_final_evalfix_20260503_073400/adapter/best_checkpoint \
  --batch_size "${QWEN_BATCH_SIZE:-2}" \
  --max_seq_length 2048 \
  --wandb_run_name qwen_robotic_spatial \
  "${COMMON[@]}"

python scripts/evaluate_robotic_spatial.py \
  --model_family gemma \
  --model_name google/gemma-3-4b-it \
  --adapter_path outputs/gemma/gemma_final_20260503_214729/adapter/best_checkpoint \
  --batch_size "${GEMMA_BATCH_SIZE:-2}" \
  --max_seq_length 2048 \
  --wandb_run_name gemma_robotic_spatial \
  "${COMMON[@]}"

python scripts/evaluate_robotic_spatial.py \
  --model_family internvl \
  --model_name OpenGVLab/InternVL3_5-4B-HF \
  --adapter_path outputs/internvl/internvl_final_20260504_073322/adapter/best_checkpoint \
  --batch_size "${INTERNVL_BATCH_SIZE:-1}" \
  --max_seq_length 4096 \
  --wandb_run_name internvl_robotic_spatial \
  "${COMMON[@]}"
