#!/usr/bin/env bash

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
OUT_DIR="${1:-$ROOT_DIR/environments/locks/linux-64}"

ENVS=(
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

mkdir -p "$OUT_DIR"
TMP_DIR="$(mktemp -d)"
trap 'rm -rf "$TMP_DIR"' EXIT

if ! command -v conda >/dev/null 2>&1; then
  echo "[ERROR] conda not found in PATH" >&2
  exit 1
fi

CONDA_VERSION="$(conda --version | tr '\t' ' ')"
PLATFORM="$(conda info --json | python -c 'import json,sys; print(json.load(sys.stdin).get("platform", "unknown"))')"
TIMESTAMP_UTC="$(date -u +%Y-%m-%dT%H:%M:%SZ)"

MANIFEST="$OUT_DIR/manifest.tsv"
printf "exported_at_utc\tenv\tconda_version\tplatform\tpython\ttorch\tcuda_available\tffmpeg\n" > "$MANIFEST"

env_exists() {
  conda env list | awk '{print $1}' | grep -Fxq "$1"
}

metadata_field() {
  local env_name="$1"
  local expr="$2"
  conda run -n "$env_name" python -c "$expr" 2>/dev/null || true
}

for env_name in "${ENVS[@]}"; do
  if ! env_exists "$env_name"; then
    echo "[WARN] missing env: $env_name"
    continue
  fi

  echo "[INFO] exporting $env_name -> $OUT_DIR"

  conda list -n "$env_name" --explicit > "$OUT_DIR/$env_name.conda-explicit.txt"
  conda env export -n "$env_name" --no-builds | sed '/^prefix:/d' > "$OUT_DIR/$env_name.conda-no-builds.yml"
  conda list -n "$env_name" --json > "$TMP_DIR/$env_name.conda-list.json"
  conda run -n "$env_name" python -m pip freeze --exclude-editable > "$TMP_DIR/$env_name.pip-freeze.raw"
  python - "$TMP_DIR/$env_name.conda-list.json" "$TMP_DIR/$env_name.pip-freeze.raw" > "$OUT_DIR/$env_name.pip-freeze.txt" <<'PY'
import json
import re
import sys

conda_json, freeze_path = sys.argv[1], sys.argv[2]

def normalize(name):
    return re.sub(r"[-_.]+", "-", name).lower()

with open(conda_json, "r", encoding="utf-8") as f:
    pypi_names = {
        normalize(pkg["name"])
        for pkg in json.load(f)
        if pkg.get("channel") == "pypi"
    }

with open(freeze_path, "r", encoding="utf-8") as f:
    for raw_line in f:
        line = raw_line.strip()
        if not line or line.startswith("-e ") or " @ file://" in line:
            continue
        match = re.match(r"([A-Za-z0-9_.-]+)\s*(?:==|~=|!=|<=|>=|<|>|@)", line)
        if match and normalize(match.group(1)) in pypi_names:
            print(line)
PY

  python_version="$(metadata_field "$env_name" 'import sys; print(sys.version.split()[0])')"
  torch_version="$(metadata_field "$env_name" 'import torch; print(torch.__version__)')"
  if [[ -z "$torch_version" ]]; then
    torch_version="not-installed"
  fi

  cuda_available="$(metadata_field "$env_name" 'import torch; print(torch.cuda.is_available())')"
  if [[ -z "$cuda_available" ]]; then
    cuda_available="unknown"
  fi

  if conda run -n "$env_name" bash -lc 'command -v ffmpeg >/dev/null 2>&1' >/dev/null 2>&1; then
    ffmpeg_status="yes"
  else
    ffmpeg_status="no"
  fi

  printf "%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n" \
    "$TIMESTAMP_UTC" \
    "$env_name" \
    "$CONDA_VERSION" \
    "$PLATFORM" \
    "${python_version:-unknown}" \
    "${torch_version:-not-installed}" \
    "${cuda_available:-unknown}" \
    "$ffmpeg_status" >> "$MANIFEST"
done

echo "[DONE] wrote reproducibility exports to $OUT_DIR"
