#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

RUN_NAME="${1:-sequential_$(date +%Y%m%d_%H%M%S)}"
SESSION_NAME="${SESSION_NAME:-zoo_all_${RUN_NAME}}"
LOG_DIR="${PROJECT_ROOT}/logs"
MASTER_LOG="${LOG_DIR}/all_${RUN_NAME}.log"
REPORT_TO="${REPORT_TO:-wandb}"
CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
MODEL_ORDER="${MODEL_ORDER:-smol qwen gemma internvl}"

mkdir -p "${LOG_DIR}"

if tmux has-session -t "${SESSION_NAME}" 2>/dev/null; then
  echo "tmux session '${SESSION_NAME}' already exists." >&2
  echo "Attach with: tmux attach -t ${SESSION_NAME}" >&2
  exit 1
fi

tmux new-session -d -s "${SESSION_NAME}" \
  "cd '${PROJECT_ROOT}' && CUDA_VISIBLE_DEVICES='${CUDA_VISIBLE_DEVICES}' REPORT_TO='${REPORT_TO}' MODEL_ORDER='${MODEL_ORDER}' '${SCRIPT_DIR}/train_all_sequential.sh' '${RUN_NAME}' 2>&1 | tee '${MASTER_LOG}'"

echo "Started sequential fine-tuning in tmux session: ${SESSION_NAME}"
echo "Run name: ${RUN_NAME}"
echo "Model order: ${MODEL_ORDER}"
echo "CUDA_VISIBLE_DEVICES: ${CUDA_VISIBLE_DEVICES}"
echo "REPORT_TO: ${REPORT_TO}"
echo "Master log: ${MASTER_LOG}"
echo "Attach: tmux attach -t ${SESSION_NAME}"
