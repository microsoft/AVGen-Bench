#!/usr/bin/env bash

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

ALL_ENVS=(
  q_align
  audiobox
  syncformer
  ocr
  syncnet
  videophy
  mllm
  face
  music
  whisper
)

if [[ $# -gt 0 ]]; then
  ENVS=("$@")
else
  ENVS=("${ALL_ENVS[@]}")
fi

if ! command -v conda >/dev/null 2>&1; then
  echo "[ERROR] conda not found in PATH" >&2
  exit 1
fi

env_exists() {
  conda env list | awk '{print $1}' | grep -Fxq "$1"
}

run_python_check() {
  local env_name="$1"
  local workdir="$2"
  local code="$3"
  (
    cd "$ROOT_DIR/$workdir"
    conda run -n "$env_name" python -c "$code"
  )
}

check_ffmpeg() {
  local env_name="$1"
  if conda run -n "$env_name" bash -lc 'command -v ffmpeg >/dev/null 2>&1'; then
    echo "[OK] $env_name: ffmpeg found"
  else
    echo "[WARN] $env_name: ffmpeg not found in the environment PATH"
  fi
}

for env_name in "${ENVS[@]}"; do
  if ! env_exists "$env_name"; then
    echo "[FAIL] missing conda env: $env_name"
    exit 1
  fi

  echo
  echo "========== $env_name =========="

  case "$env_name" in
    q_align)
      run_python_check "$env_name" "eval/Q-Align" 'import torch, torchvision, transformers, decord, q_align; print("ok", torch.__version__)'
      ;;
    audiobox)
      run_python_check "$env_name" "eval/audiobox-aesthetics" 'import torch, torchaudio, audiobox_aesthetics; print("ok", torch.__version__)'
      check_ffmpeg "$env_name"
      ;;
    syncformer)
      run_python_check "$env_name" "eval/Syncformer" 'import torch, torchvision, av, omegaconf, timm; print("ok", torch.__version__)'
      check_ffmpeg "$env_name"
      ;;
    ocr)
      run_python_check "$env_name" "eval/Ocr" 'import cv2, paddle, paddleocr, google.generativeai; print("ok", paddle.__version__)'
      ;;
    syncnet)
      run_python_check "$env_name" "eval/syncnet_python" 'import torch, torchvision, cv2, scipy, python_speech_features; print("ok", torch.__version__)'
      check_ffmpeg "$env_name"
      ;;
    videophy)
      run_python_check "$env_name" "eval/videophy/VIDEOPHY2" 'import torch, transformers, decord, cv2, peft; print("ok", torch.__version__)'
      ;;
    mllm)
      run_python_check "$env_name" "eval/gemini_phy" 'import google.generativeai, google.api_core, requests; print("ok")'
      ;;
    face)
      run_python_check "$env_name" "eval/facial_consistency" 'import cv2, insightface, onnxruntime; print("ok")'
      ;;
    music)
      run_python_check "$env_name" "eval/music_check" 'import basic_pitch, librosa, moviepy, mido, pretty_midi, tflite_runtime.interpreter; import music_check_gemini; print("ok", basic_pitch.ICASSP_2022_MODEL_PATH)'
      check_ffmpeg "$env_name"
      ;;
    whisper)
      run_python_check "$env_name" "eval/speech" 'import faster_whisper, whisper, torch, google.generativeai; print("ok", torch.__version__)'
      check_ffmpeg "$env_name"
      ;;
    *)
      echo "[FAIL] unknown env id: $env_name" >&2
      exit 1
      ;;
  esac
done

echo
echo "[DONE] environment smoke checks passed"
