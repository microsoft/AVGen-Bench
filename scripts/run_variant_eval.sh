#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="/home/v-wangrui5/AVGen-Bench"
SESSION_MODEL_DIR="/home/v-wangrui5/AVGen-Bench/stability_eval_runs/session_20260328T180932Z_4d0f84/Veo_3.1_fast"
PROMPTS_DIR="/home/v-wangrui5/AVGen-Bench/prompts"
WORKERS="${WORKERS:-16}"
FLASH_MODEL="${FLASH_MODEL:-gemini-3-flash-preview}"
VARIANTS=("v1" "v2")
ONLY_MODULES=""

usage() {
  cat <<'EOF'
Usage:
  bash scripts/run_variant_eval.sh [--session-model-dir DIR] [--prompts-dir DIR] [--workers N] [--variants "v1 v2"] [--only-modules "ocr speech plot_matching gemini_phy facial music"]

Examples:
  bash scripts/run_variant_eval.sh
  bash scripts/run_variant_eval.sh --only-modules "plot_matching gemini_phy"
  WORKERS=32 bash scripts/run_variant_eval.sh --variants "v1 v2"
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --session-model-dir)
      SESSION_MODEL_DIR="$2"
      shift 2
      ;;
    --prompts-dir)
      PROMPTS_DIR="$2"
      shift 2
      ;;
    --workers)
      WORKERS="$2"
      shift 2
      ;;
    --variants)
      IFS=' ' read -r -a VARIANTS <<< "$2"
      shift 2
      ;;
    --only-modules)
      ONLY_MODULES="$2"
      shift 2
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      echo "[ERROR] Unknown argument: $1" >&2
      usage
      exit 1
      ;;
  esac
done

if [[ ! -d "$ROOT_DIR" ]]; then
  echo "[ERROR] repo root not found: $ROOT_DIR" >&2
  exit 1
fi

if [[ ! -d "$SESSION_MODEL_DIR" ]]; then
  echo "[ERROR] session model dir not found: $SESSION_MODEL_DIR" >&2
  exit 1
fi

cd "$ROOT_DIR"

if ! command -v conda >/dev/null 2>&1; then
  echo "[ERROR] conda is not available in PATH" >&2
  exit 1
fi

eval "$(conda shell.bash hook)"

run_step() {
  local env_name="$1"
  shift
  local modules=("$@")
  local joined_modules
  joined_modules="${modules[*]}"

  if [[ -n "$ONLY_MODULES" ]]; then
    local requested=" $ONLY_MODULES "
    local selected=()
    local m
    for m in "${modules[@]}"; do
      if [[ "$requested" == *" $m "* ]]; then
        selected+=("$m")
      fi
    done
    if [[ ${#selected[@]} -eq 0 ]]; then
      echo "[SKIP] env=$env_name modules=$joined_modules"
      return 0
    fi
    modules=("${selected[@]}")
  fi

  echo "[RUN] env=$env_name modules=${modules[*]} variants=${VARIANTS[*]}"
  conda activate "$env_name"
  export OCR_GEMINI_MODEL="$FLASH_MODEL"
  export PLOT_MATCHING_GEMINI_MODEL="$FLASH_MODEL"
  export GEMINI_MODEL_NAME="$FLASH_MODEL"
  export GEMINI_MODEL="$FLASH_MODEL"
  python scripts/rerun_gemini_prompt_variants.py \
    --session_model_dir "$SESSION_MODEL_DIR" \
    --prompts_dir "$PROMPTS_DIR" \
    --workers "$WORKERS" \
    --variants "${VARIANTS[@]}" \
    --modules "${modules[@]}"
  conda deactivate
}

run_step ocr ocr
run_step whisper speech
run_step mllm plot_matching gemini_phy
run_step face facial
run_step music music

echo "[DONE] outputs under: $SESSION_MODEL_DIR/prompt_variants"
