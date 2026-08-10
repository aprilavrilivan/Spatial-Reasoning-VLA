#!/usr/bin/env bash
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

MODEL="${1:-}"
RUN_NAME="${2:-default_run}"

if [ -z "${MODEL}" ]; then
  echo "Usage: $0 {gemma|qwen|smol|internvl} [run_name]" >&2
  exit 1
fi

case "${MODEL}" in
  gemma|qwen|smol|internvl)
    ;;
  *)
    echo "Unknown model '${MODEL}'. Use one of: gemma, qwen, smol, internvl." >&2
    exit 1
    ;;
esac

TRAIN_SCRIPT="${SCRIPT_DIR}/train_${MODEL}.sh"
OUTPUT_DIR="${PROJECT_ROOT}/outputs/${MODEL}/${RUN_NAME}"
LOG_DIR="${PROJECT_ROOT}/logs"
LOG_FILE="${LOG_DIR}/${MODEL}_${RUN_NAME}.log"
SESSION_NAME="zoo_${MODEL}_${RUN_NAME}"
REPORT_TO="${REPORT_TO:-wandb}"
CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"

mkdir -p "${LOG_DIR}" "${PROJECT_ROOT}/outputs/${MODEL}"

if tmux has-session -t "${SESSION_NAME}" 2>/dev/null; then
  echo "tmux session '${SESSION_NAME}' already exists." >&2
  echo "Attach with: tmux attach -t ${SESSION_NAME}" >&2
  exit 1
fi

tmux new-session -d -s "${SESSION_NAME}" \
  "cd '${PROJECT_ROOT}' && CUDA_VISIBLE_DEVICES='${CUDA_VISIBLE_DEVICES}' REPORT_TO='${REPORT_TO}' WANDB_PROJECT='${WANDB_PROJECT:-}' '${TRAIN_SCRIPT}' '${OUTPUT_DIR}' 2>&1 | tee '${LOG_FILE}'"

echo "Started ${MODEL} training in tmux session: ${SESSION_NAME}"
echo "Output dir: ${OUTPUT_DIR}"
echo "Log file: ${LOG_FILE}"
echo "CUDA_VISIBLE_DEVICES: ${CUDA_VISIBLE_DEVICES}"
echo "REPORT_TO: ${REPORT_TO}"
echo "Attach: tmux attach -t ${SESSION_NAME}"
