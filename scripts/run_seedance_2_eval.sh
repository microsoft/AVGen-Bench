#!/usr/bin/env bash

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

RUN_TAG="${RUN_TAG:-seedance_2}"
VIDEOS_DIR="${VIDEOS_DIR:-/home/v-wangrui5/AVGen-Bench-Videos/seedance_2}"
PROMPTS_DIR="${PROMPTS_DIR:-prompts}"
OUTPUT_DIR="${OUTPUT_DIR:-user_study_eval/${RUN_TAG}}"
LOG_DIR="${LOG_DIR:-logs}"
ENV_FILE="${ENV_FILE:-seedance_2.env}"

WORKERS="${WORKERS:-64}"
SYNCNET_WORKERS="${SYNCNET_WORKERS:-24}"
VIDEOPHY_CHECKPOINT="${VIDEOPHY_CHECKPOINT:-videophy_2_auto}"
ONLY_MODULES="${ONLY_MODULES:-q_align,audiobox,syncformer,ocr,syncnet,videophy,gemini_phy,facial,plot_matching,music,speech,aggregate}"

if [[ -f "$ENV_FILE" ]]; then
  set -a
  # shellcheck disable=SC1090
  source "$ENV_FILE"
  set +a
fi

: "${GEMINI_API_KEY:?Set GEMINI_API_KEY in the environment or in ${ENV_FILE}}"

export GEMINI_API_KEY
export GOOGLE_API_KEY="${GOOGLE_API_KEY:-$GEMINI_API_KEY}"

mkdir -p "$LOG_DIR" "$OUTPUT_DIR"
LOG_FILE="${LOG_DIR}/${RUN_TAG}_full_eval_$(date -u +%Y%m%dT%H%M%SZ).log"

if pgrep -af "run_full_evaluation.sh .*--run-tag ${RUN_TAG}" >/tmp/"${RUN_TAG}".eval.pids; then
  echo "[ERROR] Another ${RUN_TAG} evaluation appears to be running:"
  cat /tmp/"${RUN_TAG}".eval.pids
  echo "Stop it first, or wait for it to finish, then rerun this script."
  exit 1
fi

echo "[INFO] run_tag     : ${RUN_TAG}"
echo "[INFO] videos_dir  : ${VIDEOS_DIR}"
echo "[INFO] output_dir  : ${OUTPUT_DIR}"
echo "[INFO] modules     : ${ONLY_MODULES}"
echo "[INFO] log_file    : ${LOG_FILE}"
echo "[INFO] gemini_api  : official Google Gemini API"

bash run_full_evaluation.sh \
  --prompts-dir "$PROMPTS_DIR" \
  --videos-dir "$VIDEOS_DIR" \
  --output-dir "$OUTPUT_DIR" \
  --run-tag "$RUN_TAG" \
  --workers "$WORKERS" \
  --syncnet-workers "$SYNCNET_WORKERS" \
  --videophy-checkpoint "$VIDEOPHY_CHECKPOINT" \
  --only_modules "$ONLY_MODULES" \
  --auto_skip \
  2>&1 | tee "$LOG_FILE"

echo "[DONE] Evaluation finished. Log: ${LOG_FILE}"
