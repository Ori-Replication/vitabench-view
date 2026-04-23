#!/usr/bin/env bash
set -euo pipefail

# 用法：
#   bash run_single.sh
#   bash run_single.sh <model_name> <max_concurrency> <num_trials> <language>
# 示例：
#   bash run_single.sh "zai-org/glm-5.1" 16 1 chinese

MODEL_NAME="${1:-zai-org/glm-5.1}"
MAX_CONCURRENCY="${2:-16}"
NUM_TRIALS="${3:-1}"
LANGUAGE="${4:-chinese}"
LOG_LEVEL="${LOG_LEVEL:-INFO}"

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
STAMP="$(date +%Y%m%d_%H%M%S)"
LOG_DIR="${ROOT_DIR}/data/logs/single_${STAMP}"
mkdir -p "${LOG_DIR}"

echo "[single] model=${MODEL_NAME} concurrency=${MAX_CONCURRENCY} trials=${NUM_TRIALS} language=${LANGUAGE}"
echo "[single] logs -> ${LOG_DIR}"

for DOMAIN in delivery instore ota; do
  OUT_LOG="${LOG_DIR}/${STAMP}_${DOMAIN}.log"
  SAVE_TO="${STAMP}_${DOMAIN}_single_glm51.json"
  echo "[single] running domain=${DOMAIN} ..."
  (
    cd "${ROOT_DIR}"
    vita run \
      --domain "${DOMAIN}" \
      --user-llm "${MODEL_NAME}" \
      --agent-llm "${MODEL_NAME}" \
      --evaluator-llm "${MODEL_NAME}" \
      --num-trials "${NUM_TRIALS}" \
      --max-concurrency "${MAX_CONCURRENCY}" \
      --language "${LANGUAGE}" \
      --log-level "${LOG_LEVEL}" \
      --save-to "${SAVE_TO}"
  ) > "${OUT_LOG}" 2>&1
  echo "[single] done domain=${DOMAIN}, log=${OUT_LOG}, save_to=data/simulations/${SAVE_TO}"
done

echo "[single] all single-scenario runs completed."
