#!/usr/bin/env bash

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

PROMPTS_DIR="${PROMPTS_DIR:-$ROOT_DIR/prompts}"
VIDEOS_ROOT="${VIDEOS_ROOT:-/home/v-wangrui5/AVGen-Bench-Videos}"
OUTPUT_ROOT="${OUTPUT_ROOT:-$ROOT_DIR/user_study_eval}"
WORKERS="${WORKERS:-64}"
RUN_MODELS="${RUN_MODELS:-all}"

ONLY_MODULES="${ONLY_MODULES:-ocr,facial,music,speech,videophy,gemini_phy,plot_matching,aggregate}"

if [[ -n "${GEMINI_API_KEY:-}" && -z "${GOOGLE_API_KEY:-}" ]]; then
  export GOOGLE_API_KEY="$GEMINI_API_KEY"
fi
if [[ -n "${GOOGLE_API_KEY:-}" && -z "${GEMINI_API_KEY:-}" ]]; then
  export GEMINI_API_KEY="$GOOGLE_API_KEY"
fi

if [[ -z "${GEMINI_API_KEY:-}" && -z "${GOOGLE_API_KEY:-}" ]]; then
  echo "[ERROR] GEMINI_API_KEY / GOOGLE_API_KEY is not set."
  echo "Example:"
  echo "  export GEMINI_API_KEY='your_google_gemini_api_key'"
  exit 1
fi

mkdir -p "$OUTPUT_ROOT"

run_one() {
  local run_tag="$1"
  local videos_dir="$VIDEOS_ROOT/$run_tag"
  local output_dir="$OUTPUT_ROOT/$run_tag"

  if [[ ! -d "$videos_dir" ]]; then
    echo "[ERROR] videos dir not found: $videos_dir"
    exit 1
  fi

  echo
  echo "========== Running $run_tag =========="
  bash "$ROOT_DIR/run_full_evaluation.sh" \
    --prompts-dir "$PROMPTS_DIR" \
    --videos-dir "$videos_dir" \
    --output-dir "$output_dir" \
    --run-tag "$run_tag" \
    --only_modules "$ONLY_MODULES" \
    --workers "$WORKERS" \
    --auto_skip
}

case "$RUN_MODELS" in
  all)
    run_one "Ovi_11"
    run_one "Kling_2.6"
    ;;
  ovi)
    run_one "Ovi_11"
    ;;
  kling)
    run_one "Kling_2.6"
    ;;
  *)
    echo "[ERROR] RUN_MODELS must be one of: all, ovi, kling"
    exit 1
    ;;
esac

echo
echo "[DONE] requested user-study evaluations finished."
