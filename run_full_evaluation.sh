#!/usr/bin/env bash

set -u
set -o pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

PROMPTS_DIR="prompts"
VIDEOS_DIR="generated_videos/veo3.1_fast"
OUTPUT_DIR="avgenbench"

WORKERS="32"
SYNCFORMER_EXP_NAME="24-01-04T16-39-21"
VIDEOPHY_CHECKPOINT="videophy_2_auto"
RUN_TAG=""

usage() {
  cat <<'EOF'
Usage:
  bash run_full_evaluation.sh [--prompts-dir DIR] [--videos-dir DIR] [--output-dir DIR] [--run-tag TAG]

Defaults:
  --prompts-dir  prompts
  --videos-dir   generated_videos/veo3.1_fast
  --output-dir   avgenbench

Optional:
  --workers              32
  --syncformer-exp-name  24-01-04T16-39-21
  --videophy-checkpoint  videophy_2_auto
  --run-tag              basename(videos-dir)

Notes:
  - Run this script from anywhere; paths are resolved to absolute paths.
  - Commands are executed in each module subdirectory.
  - Outputs follow avgenbench-style structure, e.g. <output-dir>/q_align/<run-tag>.csv and <output-dir>/speech/<run-tag>/summary.json.
  - If neither GEMINI_API_KEY nor GOOGLE_API_KEY is set, Gemini-dependent modules are skipped.
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

mkdir -p "$OUTPUT_DIR/q_align"
run_step "Q-Align (Visual Quality)" "q_align" "eval/Q-Align" \
  python batch_eval.py \
  --root "$VIDEOS_DIR" \
  --save_summary_csv "$OUTPUT_DIR/q_align/${RUN_TAG}.csv"

mkdir -p "$OUTPUT_DIR/audiobox_aesthetic"
run_step "Audiobox-Aesthetic (Audio Quality)" "audiobox" "eval/audiobox-aesthetics" \
  python batch_eval.py \
  --root "$VIDEOS_DIR" \
  --save_summary_csv "$OUTPUT_DIR/audiobox_aesthetic/${RUN_TAG}.csv"

mkdir -p "$OUTPUT_DIR/av_sync"
run_step "Synchformer (AV Sync)" "syncformer" "eval/Syncformer" \
  python batch_eval.py \
  --root "$VIDEOS_DIR" \
  --save_summary_csv "$OUTPUT_DIR/av_sync/${RUN_TAG}.csv" \
  --exp_name "$SYNCFORMER_EXP_NAME"

mkdir -p "$OUTPUT_DIR/ocr/$RUN_TAG"
run_step "OCR (Scene Text Rendering)" "ocr" "eval/Ocr" \
  python batch_eval.py \
  --root "$VIDEOS_DIR" \
  --prompts_dir "$PROMPTS_DIR" \
  --out_dir "$OUTPUT_DIR/ocr/$RUN_TAG" \
  --save_csv "$OUTPUT_DIR/ocr/$RUN_TAG/results_text_quality.csv" \
  --gemini_workers "$WORKERS"

mkdir -p "$OUTPUT_DIR/syncnet/$RUN_TAG"
run_step "SyncNet (Lip Sync)" "syncnet" "eval/syncnet_python" \
  python batch_eval.py \
  --video_root "$VIDEOS_DIR" \
  --save_csv "$OUTPUT_DIR/syncnet/$RUN_TAG/result.csv" \
  --data_dir "$OUTPUT_DIR/syncnet/$RUN_TAG/work_batch" \
  --conf_th 1.0 \
  --inference_py inference.py

mkdir -p "$OUTPUT_DIR/videophy2"
run_step "VideoPhy2 (Low-level Physical Plausibility)" "videophy" "eval/videophy/VIDEOPHY2" \
  python batch_eval.py \
  --root "$VIDEOS_DIR" \
  --save_summary_csv "$OUTPUT_DIR/videophy2/${RUN_TAG}.csv" \
  --checkpoint "$VIDEOPHY_CHECKPOINT" \
  --task pc

mkdir -p "$OUTPUT_DIR/gemini_phy2/$RUN_TAG"
run_step_optional_gemini "Gemini Phy (High-level Physical Plausibility)" "mllm" "eval/gemini_phy" \
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
run_step "Facial Quality (Facial Consistency)" "face" "eval/facial_consistency" \
  python batch_eval.py \
  --prompts_dir "$FACIAL_PROMPTS_DIR" \
  --root_videos "$VIDEOS_DIR" \
  --out_json "$OUTPUT_DIR/facial/$RUN_TAG/eval_results.json" \
  --ctx_id 0

mkdir -p "$OUTPUT_DIR/plot_matching/$RUN_TAG"
run_step_optional_gemini "Plot Matching (Holistic Semantic Alignment)" "mllm" "eval/plot_matching" \
  python batch_eval.py \
  --prompts_dir "$PROMPTS_DIR" \
  --root_videos "$VIDEOS_DIR" \
  --cache_json "$OUTPUT_DIR/plot_matching/$RUN_TAG/eval_cache.json" \
  --out_json "$OUTPUT_DIR/plot_matching/$RUN_TAG/eval_results.json"

mkdir -p "$OUTPUT_DIR/music/$RUN_TAG"
run_step_optional_gemini "Music Check (Pitch Accuracy)" "music" "eval/music_check" \
  python batch_eval.py \
  --videos-root "$VIDEOS_DIR" \
  --prompts-root "$PROMPTS_DIR" \
  --outputs-root "$OUTPUT_DIR/music/$RUN_TAG/per_video_result" \
  --summary-out "$OUTPUT_DIR/music/$RUN_TAG/summary.json" \
  --only-category musical_instrument_tutorial \
  --workers "$WORKERS" \
  --constraints-cache-dir "$OUTPUT_DIR/music/$RUN_TAG/music_prompt_constraints"

mkdir -p "$OUTPUT_DIR/speech/$RUN_TAG"
run_step_optional_gemini "Speech (Speech Intelligibility & Coherence)" "whisper" "eval/speech" \
  python batch_eval.py \
  --videos_root "$VIDEOS_DIR" \
  --prompts_dir "$PROMPTS_DIR" \
  --out_dir "$OUTPUT_DIR/speech/$RUN_TAG" \
  --gemini_workers "$WORKERS" \
  --whisper_model large-v3

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

echo
echo "========== Evaluation Summary =========="
echo "[INFO] prompts_dir: $PROMPTS_DIR"
echo "[INFO] videos_dir : $VIDEOS_DIR"
echo "[INFO] output_dir : $OUTPUT_DIR"
echo "[INFO] run_tag    : $RUN_TAG"

if [[ ${#SKIPPED_STEPS[@]} -gt 0 ]]; then
  echo "[INFO] skipped steps: ${SKIPPED_STEPS[*]}"
fi

if [[ ${#FAILED_STEPS[@]} -gt 0 ]]; then
  echo "[ERROR] failed steps: ${FAILED_STEPS[*]}"
  exit 1
fi

echo "[DONE] all requested evaluations finished successfully."
