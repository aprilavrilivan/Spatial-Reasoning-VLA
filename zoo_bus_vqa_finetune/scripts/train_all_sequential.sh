#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

if [ -f "${SCRIPT_DIR}/remote_env.sh" ]; then
  source "${SCRIPT_DIR}/remote_env.sh"
fi

RUN_NAME="${1:-sequential_$(date +%Y%m%d_%H%M%S)}"
MODEL_ORDER="${MODEL_ORDER:-smol qwen gemma internvl}"
REPORT_TO="${REPORT_TO:-wandb}"
CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
LOG_DIR="${PROJECT_ROOT}/logs"
SUMMARY_FILE="${PROJECT_ROOT}/outputs/sequential_${RUN_NAME}_summary.tsv"

mkdir -p "${LOG_DIR}" "${PROJECT_ROOT}/outputs"

echo -e "model\tstatus\tstarted_at\tfinished_at\toutput_dir\tlog_file" > "${SUMMARY_FILE}"
echo "Sequential run: ${RUN_NAME}"
echo "Model order: ${MODEL_ORDER}"
echo "CUDA_VISIBLE_DEVICES: ${CUDA_VISIBLE_DEVICES}"
echo "REPORT_TO: ${REPORT_TO}"
echo "Summary: ${SUMMARY_FILE}"

for MODEL in ${MODEL_ORDER}; do
  TRAIN_SCRIPT="${SCRIPT_DIR}/train_${MODEL}.sh"
  if [ ! -x "${TRAIN_SCRIPT}" ]; then
    echo "Missing executable train script: ${TRAIN_SCRIPT}" >&2
    exit 1
  fi

  STARTED_AT="$(date -Iseconds)"
  OUTPUT_DIR="${PROJECT_ROOT}/outputs/${MODEL}/${RUN_NAME}"
  LOG_FILE="${LOG_DIR}/${MODEL}_${RUN_NAME}.log"

  echo "============================================================"
  echo "Starting ${MODEL} at ${STARTED_AT}"
  echo "Output dir: ${OUTPUT_DIR}"
  echo "Log file: ${LOG_FILE}"
  echo "============================================================"

  set +e
  CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES}" \
  REPORT_TO="${REPORT_TO}" \
  WANDB_PROJECT="${WANDB_PROJECT:-}" \
  "${TRAIN_SCRIPT}" "${OUTPUT_DIR}" 2>&1 | tee "${LOG_FILE}"
  STATUS="${PIPESTATUS[0]}"
  set -e

  FINISHED_AT="$(date -Iseconds)"
  if [ "${STATUS}" -eq 0 ]; then
    echo -e "${MODEL}\tsuccess\t${STARTED_AT}\t${FINISHED_AT}\t${OUTPUT_DIR}\t${LOG_FILE}" >> "${SUMMARY_FILE}"
    echo "Finished ${MODEL} successfully at ${FINISHED_AT}"
  else
    echo -e "${MODEL}\tfailed_${STATUS}\t${STARTED_AT}\t${FINISHED_AT}\t${OUTPUT_DIR}\t${LOG_FILE}" >> "${SUMMARY_FILE}"
    echo "Model ${MODEL} failed with exit code ${STATUS}. Stopping sequential run." >&2
    exit "${STATUS}"
  fi
done

echo "All sequential fine-tuning jobs completed."
echo "Summary: ${SUMMARY_FILE}"
