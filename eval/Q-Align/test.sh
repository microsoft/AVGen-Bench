#!/usr/bin/env bash
set -Eeuo pipefail

ROOT="/path/to/video_generation/veo3.1_fast"

export -f _fix_one || true

_fix_one() {
  local f="$1"
  printf 'FIX: %s\n' "$f" >&2

  local dir base tmp
  dir="$(dirname -- "$f")"
  base="$(basename -- "$f")"
  tmp="${dir}/.${base}.tmp_fixed.$$.$RANDOM.mp4"

  # Re-encode into a temporary file.
  if ! ffmpeg -y -hide_banner -loglevel error \
      -fflags +genpts -i "$f" \
      -map 0:v:0 -map 0:a? \
      -c:v libx264 -pix_fmt yuv420p -profile:v high -level 4.1 \
      -preset veryfast -crf 18 \
      -c:a aac -b:a 128k \
      -movflags +faststart \
      "$tmp"
  then
    printf 'FAILED: %s\n' "$f" >&2
    rm -f -- "$tmp"
    return 0
  fi

  mv -f -- "$tmp" "$f"
}

export -f _fix_one

# Process files one by one while preserving original paths (including spaces/parentheses).
find "$ROOT" -type f -iname "*.mp4" -exec bash -c '_fix_one "$1"' bash {} \;
