#!/usr/bin/env bash

set -u
set -o pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

PROMPTS_DIR="prompts"
VIDEOS_DIR="generated_videos/veo3.1_fast"
OUTPUT_DIR="avgenbench"

WORKERS="64"
SYNCNET_WORKERS="24"
SYNCFORMER_EXP_NAME="24-01-04T16-39-21"
VIDEOPHY_CHECKPOINT="videophy_2_auto"
RUN_TAG=""
ONLY_MODULES=""
AUTO_SKIP="0"

usage() {
  cat <<'EOF'
Usage:
  bash run_full_evaluation.sh [--prompts-dir DIR] [--videos-dir DIR] [--output-dir DIR] [--run-tag TAG] [--only_modules LIST] [--auto_skip]

Defaults:
  --prompts-dir  prompts
  --videos-dir   generated_videos/veo3.1_fast
  --output-dir   avgenbench

Optional:
  --workers              32
  --syncnet-workers      2
  --syncformer-exp-name  24-01-04T16-39-21
  --videophy-checkpoint  videophy_2_auto
  --run-tag              basename(videos-dir)
  --only_modules         comma-separated module ids to run
                         available: q_align,audiobox,syncformer,ocr,syncnet,videophy,gemini_phy,facial,plot_matching,music,speech,aggregate
  --auto_skip            skip modules whose main output file already exists

Notes:
  - Run this script from anywhere; paths are resolved to absolute paths.
  - Commands are executed in each module subdirectory.
  - Outputs follow avgenbench-style structure, e.g. <output-dir>/q_align/<run-tag>.csv and <output-dir>/speech/<run-tag>/summary.json.
  - If neither GEMINI_API_KEY nor GOOGLE_API_KEY is set, Gemini-dependent modules are skipped.
  - If --only_modules is set, only the listed modules are executed. Include `aggregate` explicitly if you also want overall score aggregation.
  - If --auto_skip is set, modules with an existing non-empty primary output file are skipped automatically.
EOF
}

abs_path() {
  python -c 'import os,sys; print(os.path.abspath(sys.argv[1]))' "$1"
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --prompts-dir)
      PROMPTS_DIR="$2"
      shift 2
      ;;
    --videos-dir)
      VIDEOS_DIR="$2"
      shift 2
      ;;
    --output-dir)
      OUTPUT_DIR="$2"
      shift 2
      ;;
    --workers)
      WORKERS="$2"
      shift 2
      ;;
    --syncnet-workers)
      SYNCNET_WORKERS="$2"
      shift 2
      ;;
    --syncformer-exp-name)
      SYNCFORMER_EXP_NAME="$2"
      shift 2
      ;;
    --videophy-checkpoint)
      VIDEOPHY_CHECKPOINT="$2"
      shift 2
      ;;
    --run-tag)
      RUN_TAG="$2"
      shift 2
      ;;
    --only_modules)
      ONLY_MODULES="$2"
      shift 2
      ;;
    --auto_skip)
      AUTO_SKIP="1"
      shift 1
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      echo "[ERROR] Unknown argument: $1"
      usage
      exit 1
      ;;
  esac
done

PROMPTS_DIR="$(abs_path "$PROMPTS_DIR")"
VIDEOS_DIR="$(abs_path "$VIDEOS_DIR")"
OUTPUT_DIR="$(abs_path "$OUTPUT_DIR")"

if ! command -v conda >/dev/null 2>&1; then
  echo "[ERROR] conda not found in PATH."
  exit 1
fi

if [[ ! -d "$PROMPTS_DIR" ]]; then
  echo "[ERROR] prompts directory not found: $PROMPTS_DIR"
  exit 1
fi

if [[ ! -d "$VIDEOS_DIR" ]]; then
  echo "[ERROR] generated videos directory not found: $VIDEOS_DIR"
  exit 1
fi

mkdir -p "$OUTPUT_DIR"

if [[ -z "$RUN_TAG" ]]; then
  RUN_TAG="$(basename "$VIDEOS_DIR")"
fi
if [[ -z "$RUN_TAG" || "$RUN_TAG" == "." || "$RUN_TAG" == "/" ]]; then
  RUN_TAG="run"
fi
echo "[INFO] run_tag   : $RUN_TAG"

if [[ -n "${GEMINI_API_KEY:-}" && -z "${GOOGLE_API_KEY:-}" ]]; then
  export GOOGLE_API_KEY="$GEMINI_API_KEY"
fi
if [[ -n "${GOOGLE_API_KEY:-}" && -z "${GEMINI_API_KEY:-}" ]]; then
  export GEMINI_API_KEY="$GOOGLE_API_KEY"
fi

FACIAL_PROMPTS_DIR="$ROOT_DIR/eval/facial_consistency/prompts_expected_faces"
if [[ ! -d "$FACIAL_PROMPTS_DIR" ]]; then
  echo "[ERROR] facial prompts directory not found: $FACIAL_PROMPTS_DIR"
  exit 1
fi

declare -a FAILED_STEPS=()
declare -a SKIPPED_STEPS=()
declare -A REQUESTED_MODULES=()

normalize_module_id() {
  local module_id="$1"
  module_id="${module_id// /}"
  module_id="${module_id,,}"
  case "$module_id" in
    av_sync)
      echo "syncformer"
      ;;
    gemini_phy2)
      echo "gemini_phy"
      ;;
    videophy2)
      echo "videophy"
      ;;
    *)
      echo "$module_id"
      ;;
  esac
}

if [[ -n "$ONLY_MODULES" ]]; then
  IFS=',' read -r -a _requested_modules <<< "$ONLY_MODULES"
  for module_name in "${_requested_modules[@]}"; do
    module_name="$(normalize_module_id "$module_name")"
    if [[ -n "$module_name" ]]; then
      REQUESTED_MODULES["$module_name"]=1
    fi
  done

  for module_name in "${!REQUESTED_MODULES[@]}"; do
    case "$module_name" in
      q_align|audiobox|syncformer|ocr|syncnet|videophy|gemini_phy|facial|plot_matching|music|speech|aggregate)
        ;;
      *)
        echo "[ERROR] Unknown module in --only_modules: $module_name"
        usage
        exit 1
        ;;
    esac
  done
fi

module_requested() {
  local module_id
  module_id="$(normalize_module_id "$1")"
  if [[ ${#REQUESTED_MODULES[@]} -eq 0 ]]; then
    return 0
  fi
  [[ -n "${REQUESTED_MODULES[$module_id]:-}" ]]
}

should_auto_skip_step() {
  local completion_path="$1"
  if [[ "$AUTO_SKIP" != "1" ]]; then
    return 1
  fi
  if [[ -z "$completion_path" ]]; then
    return 1
  fi
  [[ -s "$completion_path" ]]
}

run_step() {
  local step_name="$1"
  local env_name="$2"
  local subdir="$3"
  shift 3

  local workdir="$ROOT_DIR/$subdir"
  if [[ ! -d "$workdir" ]]; then
    echo "[FAIL] ${step_name}: missing directory ${workdir}"
    FAILED_STEPS+=("$step_name")
    return 1
  fi

  echo
  echo "========== ${step_name} =========="
  echo "[INFO] workdir: ${workdir}"
  echo "[INFO] conda env: ${env_name}"

  (
    cd "$workdir" || exit 1
    conda run -n "$env_name" "$@"
  )
  local code=$?

  if [[ $code -ne 0 ]]; then
    echo "[FAIL] ${step_name} (exit code: ${code})"
    FAILED_STEPS+=("$step_name")
  else
    echo "[DONE] ${step_name}"
  fi
}

run_step_optional_gemini() {
  local step_name="$1"
  shift
  if [[ -z "${GEMINI_API_KEY:-}" && -z "${GOOGLE_API_KEY:-}" ]]; then
    echo "[SKIP] ${step_name}: GEMINI_API_KEY / GOOGLE_API_KEY is not set"
    SKIPPED_STEPS+=("$step_name")
    return 0
  fi
  run_step "$step_name" "$@"
}

run_selected_step() {
  local module_id="$1"
  local step_name="$2"
  shift 2
  if ! module_requested "$module_id"; then
    echo "[SKIP] ${step_name}: not selected by --only_modules"
    SKIPPED_STEPS+=("$step_name")
    return 0
  fi
  run_step "$step_name" "$@"
}

run_selected_step_optional_gemini() {
  local module_id="$1"
  local step_name="$2"
  shift 2
  if ! module_requested "$module_id"; then
    echo "[SKIP] ${step_name}: not selected by --only_modules"
    SKIPPED_STEPS+=("$step_name")
    return 0
  fi
  run_step_optional_gemini "$step_name" "$@"
}

run_selected_step_with_output() {
  local module_id="$1"
  local completion_path="$2"
  local step_name="$3"
  shift 3
  if ! module_requested "$module_id"; then
    echo "[SKIP] ${step_name}: not selected by --only_modules"
    SKIPPED_STEPS+=("$step_name")
    return 0
  fi
  if should_auto_skip_step "$completion_path"; then
    echo "[SKIP] ${step_name}: output already exists at ${completion_path}"
    SKIPPED_STEPS+=("$step_name")
    return 0
  fi
  run_step "$step_name" "$@"
}

run_selected_step_optional_gemini_with_output() {
  local module_id="$1"
  local completion_path="$2"
  local step_name="$3"
  shift 3
  if ! module_requested "$module_id"; then
    echo "[SKIP] ${step_name}: not selected by --only_modules"
    SKIPPED_STEPS+=("$step_name")
    return 0
  fi
  if should_auto_skip_step "$completion_path"; then
    echo "[SKIP] ${step_name}: output already exists at ${completion_path}"
    SKIPPED_STEPS+=("$step_name")
    return 0
  fi
  run_step_optional_gemini "$step_name" "$@"
}

mkdir -p "$OUTPUT_DIR/q_align"
run_selected_step_with_output "q_align" "$OUTPUT_DIR/q_align/${RUN_TAG}.csv" "Q-Align (Visual Quality)" "q_align" "eval/Q-Align" \
  python batch_eval.py \
  --root "$VIDEOS_DIR" \
  --save_summary_csv "$OUTPUT_DIR/q_align/${RUN_TAG}.csv"

mkdir -p "$OUTPUT_DIR/audiobox_aesthetic"
run_selected_step_with_output "audiobox" "$OUTPUT_DIR/audiobox_aesthetic/${RUN_TAG}.csv" "Audiobox-Aesthetic (Audio Quality)" "audiobox" "eval/audiobox-aesthetics" \
  python batch_eval.py \
  --root "$VIDEOS_DIR" \
  --save_summary_csv "$OUTPUT_DIR/audiobox_aesthetic/${RUN_TAG}.csv"

mkdir -p "$OUTPUT_DIR/av_sync"
run_selected_step_with_output "syncformer" "$OUTPUT_DIR/av_sync/${RUN_TAG}.csv" "Synchformer (AV Sync)" "syncformer" "eval/Syncformer" \
  python batch_eval.py \
  --root "$VIDEOS_DIR" \
  --save_summary_csv "$OUTPUT_DIR/av_sync/${RUN_TAG}.csv" \
  --exp_name "$SYNCFORMER_EXP_NAME"

mkdir -p "$OUTPUT_DIR/ocr/$RUN_TAG"
run_selected_step_with_output "ocr" "$OUTPUT_DIR/ocr/$RUN_TAG/summary.json" "OCR (Scene Text Rendering)" "ocr" "eval/Ocr" \
  python batch_eval.py \
  --root "$VIDEOS_DIR" \
  --prompts_dir "$PROMPTS_DIR" \
  --out_dir "$OUTPUT_DIR/ocr/$RUN_TAG" \
  --save_csv "$OUTPUT_DIR/ocr/$RUN_TAG/results_text_quality.csv" \
  --save_summary_csv "$OUTPUT_DIR/ocr/$RUN_TAG/summary.csv" \
  --save_summary_json "$OUTPUT_DIR/ocr/$RUN_TAG/summary.json" \
  --gemini_workers "$WORKERS"

mkdir -p "$OUTPUT_DIR/syncnet/$RUN_TAG"
run_selected_step_with_output "syncnet" "$OUTPUT_DIR/syncnet/$RUN_TAG/result.csv" "SyncNet (Lip Sync)" "syncnet" "eval/syncnet_python" \
  python batch_eval.py \
  --video_root "$VIDEOS_DIR" \
  --save_csv "$OUTPUT_DIR/syncnet/$RUN_TAG/result.csv" \
  --data_dir "$OUTPUT_DIR/syncnet/$RUN_TAG/work_batch" \
  --workers "$SYNCNET_WORKERS" \
  --conf_th 1.0 \
  --inference_py inference.py

mkdir -p "$OUTPUT_DIR/videophy2"
run_selected_step_with_output "videophy" "$OUTPUT_DIR/videophy2/${RUN_TAG}.csv" "VideoPhy2 (Low-level Physical Plausibility)" "videophy" "eval/videophy/VIDEOPHY2" \
  python batch_eval.py \
  --root "$VIDEOS_DIR" \
  --save_csv "$OUTPUT_DIR/videophy2/$RUN_TAG/results.csv" \
  --save_summary_csv "$OUTPUT_DIR/videophy2/${RUN_TAG}.csv" \
  --checkpoint "$VIDEOPHY_CHECKPOINT" \
  --task pc

mkdir -p "$OUTPUT_DIR/gemini_phy2/$RUN_TAG"
run_selected_step_optional_gemini_with_output "gemini_phy" "$OUTPUT_DIR/gemini_phy2/$RUN_TAG/summary.csv" "Gemini Phy (High-level Physical Plausibility)" "mllm" "eval/gemini_phy" \
  python batch_eval.py \
  --root_videos "$VIDEOS_DIR" \
  --prompts_dir "$PROMPTS_DIR" \
  --model gemini-3-flash-preview \
  --expectations_cache "$OUTPUT_DIR/gemini_phy2/expectations_cache.json" \
  --out_dir "$OUTPUT_DIR/gemini_phy2/$RUN_TAG" \
  --save_csv "$OUTPUT_DIR/gemini_phy2/$RUN_TAG/results.csv" \
  --save_summary_csv "$OUTPUT_DIR/gemini_phy2/$RUN_TAG/summary.csv" \
  --workers "$WORKERS"

mkdir -p "$OUTPUT_DIR/facial/$RUN_TAG"
run_selected_step_with_output "facial" "$OUTPUT_DIR/facial/$RUN_TAG/eval_results.json" "Facial Quality (Facial Consistency)" "face" "eval/facial_consistency" \
  python batch_eval.py \
  --prompts_dir "$FACIAL_PROMPTS_DIR" \
  --root_videos "$VIDEOS_DIR" \
  --out_json "$OUTPUT_DIR/facial/$RUN_TAG/eval_results.json" \
  --ctx_id 0

mkdir -p "$OUTPUT_DIR/plot_matching/$RUN_TAG"
run_selected_step_optional_gemini_with_output "plot_matching" "$OUTPUT_DIR/plot_matching/$RUN_TAG/eval_results.json" "Plot Matching (Holistic Semantic Alignment)" "mllm" "eval/plot_matching" \
  python batch_eval.py \
  --prompts_dir "$PROMPTS_DIR" \
  --root_videos "$VIDEOS_DIR" \
  --cache_json "$OUTPUT_DIR/plot_matching/$RUN_TAG/eval_cache.json" \
  --out_json "$OUTPUT_DIR/plot_matching/$RUN_TAG/eval_results.json"

mkdir -p "$OUTPUT_DIR/music/$RUN_TAG"
run_selected_step_optional_gemini_with_output "music" "$OUTPUT_DIR/music/$RUN_TAG/summary.json" "Music Check (Pitch Accuracy)" "music" "eval/music_check" \
  python batch_eval.py \
  --videos-root "$VIDEOS_DIR" \
  --prompts-root "$PROMPTS_DIR" \
  --outputs-root "$OUTPUT_DIR/music/$RUN_TAG/per_video_result" \
  --summary-out "$OUTPUT_DIR/music/$RUN_TAG/summary.json" \
  --only-category musical_instrument_tutorial \
  --workers "$WORKERS" \
  --constraints-cache-dir "$OUTPUT_DIR/music/$RUN_TAG/music_prompt_constraints"

mkdir -p "$OUTPUT_DIR/speech/$RUN_TAG"
run_selected_step_optional_gemini_with_output "speech" "$OUTPUT_DIR/speech/$RUN_TAG/summary.json" "Speech (Speech Intelligibility & Coherence)" "whisper" "eval/speech" \
  python batch_eval.py \
  --videos_root "$VIDEOS_DIR" \
  --prompts_dir "$PROMPTS_DIR" \
  --out_dir "$OUTPUT_DIR/speech/$RUN_TAG" \
  --gemini_workers "$WORKERS" \
  --whisper_model large-v3

if module_requested "aggregate"; then
  if should_auto_skip_step "$OUTPUT_DIR/overall_score/${RUN_TAG}.json"; then
    echo "[SKIP] Aggregate Score: output already exists at $OUTPUT_DIR/overall_score/${RUN_TAG}.json"
    SKIPPED_STEPS+=("Aggregate Score")
  else
  echo
  echo "========== Aggregate Score =========="
  mkdir -p "$OUTPUT_DIR/overall_score"
  python "$ROOT_DIR/aggregate_score.py" \
    --output-dir "$OUTPUT_DIR" \
    --run-tag "$RUN_TAG" \
    --save-json "$OUTPUT_DIR/overall_score/${RUN_TAG}.json" \
    --save-csv "$OUTPUT_DIR/overall_score/${RUN_TAG}.csv"
  if [[ $? -ne 0 ]]; then
    echo "[FAIL] Aggregate Score"
    FAILED_STEPS+=("Aggregate Score")
  else
    echo "[DONE] Aggregate Score"
  fi
  fi
else
  echo "[SKIP] Aggregate Score: not selected by --only_modules"
  SKIPPED_STEPS+=("Aggregate Score")
fi

echo
echo "========== Evaluation Summary =========="
echo "[INFO] prompts_dir: $PROMPTS_DIR"
echo "[INFO] videos_dir : $VIDEOS_DIR"
echo "[INFO] output_dir : $OUTPUT_DIR"
echo "[INFO] run_tag    : $RUN_TAG"
echo "[INFO] syncnet_workers: $SYNCNET_WORKERS"
if [[ ${#REQUESTED_MODULES[@]} -gt 0 ]]; then
  echo "[INFO] only_modules: ${ONLY_MODULES}"
fi
if [[ "$AUTO_SKIP" == "1" ]]; then
  echo "[INFO] auto_skip   : enabled"
fi

if [[ ${#SKIPPED_STEPS[@]} -gt 0 ]]; then
  echo "[INFO] skipped steps: ${SKIPPED_STEPS[*]}"
fi

if [[ ${#FAILED_STEPS[@]} -gt 0 ]]; then
  echo "[ERROR] failed steps: ${FAILED_STEPS[*]}"
  exit 1
fi

echo "[DONE] all requested evaluations finished successfully."
