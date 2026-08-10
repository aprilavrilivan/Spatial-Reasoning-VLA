#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${PROJECT_ROOT}"

if [ -f "${SCRIPT_DIR}/remote_env.sh" ]; then
  source "${SCRIPT_DIR}/remote_env.sh"
fi
source .venv/bin/activate

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
REPORT_TO="${REPORT_TO:-wandb}"
WANDB_PROJECT="${WANDB_PROJECT:-zoo-bus-vqa-external-spatial-eval}"
OUTPUT_DIR="${OUTPUT_DIR:-${PROJECT_ROOT}/external_eval/results}"
COMMON=(--dataset all --state both --download_whatsup --report_to "${REPORT_TO}" --wandb_project "${WANDB_PROJECT}" --output_dir "${OUTPUT_DIR}")

python scripts/evaluate_external_spatial.py \
  --model_family smol \
  --model_name HuggingFaceTB/SmolVLM2-2.2B-Instruct \
  --adapter_path outputs/smol/smol_final_20260502_120924/checkpoints/checkpoint-2949 \
  --batch_size 64 \
  --max_seq_length 2048 \
  --wandb_run_name smol_external_spatial \
  "${COMMON[@]}"

python scripts/evaluate_external_spatial.py \
  --model_family qwen \
  --model_name Qwen/Qwen3-VL-4B-Instruct \
  --adapter_path outputs/qwen/qwen_final_evalfix_20260503_073400/checkpoints/checkpoint-2949 \
  --batch_size 64 \
  --max_seq_length 2048 \
  --wandb_run_name qwen_external_spatial \
  "${COMMON[@]}"

python scripts/evaluate_external_spatial.py \
  --model_family gemma \
  --model_name google/gemma-3-4b-it \
  --adapter_path outputs/gemma/gemma_final_20260503_214729/checkpoints/checkpoint-2949 \
  --batch_size 64 \
  --max_seq_length 512 \
  --wandb_run_name gemma_external_spatial \
  "${COMMON[@]}"

python scripts/evaluate_external_spatial.py \
  --model_family internvl \
  --model_name OpenGVLab/InternVL3_5-4B-HF \
  --adapter_path outputs/internvl/internvl_final_20260504_073322/checkpoints/checkpoint-2949 \
  --batch_size 32 \
  --max_seq_length 4096 \
  --wandb_run_name internvl_external_spatial \
  "${COMMON[@]}"
