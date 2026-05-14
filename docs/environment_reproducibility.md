# Environment reproducibility policy

This repository runs each AVGen-Bench evaluation module in a dedicated Conda
environment. Treat environment files as release artifacts, not as one-off
machine dumps.

## Current issue

The files under `environments/*.yml` are full solver exports. They are useful as
a record of a working machine, but they are fragile as public installation
inputs because they include:

- build strings such as `h6fa692b_0`;
- transitive system libraries selected for one Linux machine;
- tightly pinned CUDA runtime packages;
- mixed `defaults`, `conda-forge`, `pytorch`, `nvidia`, and pip packages;
- occasionally local `prefix:` fields in exported files;
- pip CUDA wheels that must match the target machine's driver and Python ABI.

These details make `conda env create -f environments/foo.yml` much harder to
solve on a different host.

## Recommended artifact split

Maintain two environment layers:

1. Portable specs for users.

   Keep `environments/*.yml` small and hand-maintained. They should include only
   direct runtime requirements, Python major/minor version, a CUDA family when
   needed, and module-specific pip packages. Do not include `prefix:`, build
   strings, or long transitive dependency lists.

2. Exact Linux snapshots for result reproduction.

   Generate exact snapshots from the known-good local environments into
   `environments/locks/linux-64/`. These are not meant to replace portable
   specs. They are for reproducing the paper/evaluation host as closely as
   possible on another compatible Linux x86_64 machine.

Use the export helper:

```bash
bash scripts/export_repro_envs.sh
```

It writes, per environment:

- `<env>.conda-explicit.txt`: exact Conda package URLs for Linux x86_64;
- `<env>.pip-freeze.txt`: exact pip packages installed in the environment;
- `<env>.conda-no-builds.yml`: a debug snapshot without build strings or
  local `prefix:`;
- `manifest.tsv`: export metadata, including Python, torch, CUDA visibility,
  and ffmpeg visibility.

## Creating environments

For normal users, prefer the portable specs:

```bash
conda env create -f environments/visual_quality.yml -n q_align
conda env create -f environments/audio_quality.yml -n audiobox
conda env create -f environments/avsync.yml -n syncformer
```

For exact Linux reproduction, first create the Conda packages from an explicit
lock and then install the pip freeze:

```bash
conda create -n q_align --file environments/locks/linux-64/q_align.conda-explicit.txt
conda run -n q_align python -m pip install -r environments/locks/linux-64/q_align.pip-freeze.txt
```

This exact path is platform-specific. Use it only on compatible Linux x86_64
hosts with a CUDA driver new enough for the locked runtime.

## Validation

Run a fast import-level check before launching the full benchmark:

```bash
bash scripts/check_eval_envs.sh
```

You can check a subset:

```bash
bash scripts/check_eval_envs.sh q_align mllm whisper
```

This catches the most common reproducibility failures early: missing packages,
wrong Python version, CUDA stack mismatch, missing `ffmpeg`, or import-time ABI
errors in packages such as OpenCV, PaddleOCR, InsightFace, Faster-Whisper,
Basic Pitch, and PyTorch.

## Environment mapping

| Eval module | Conda env | Portable spec |
|---|---|---|
| `eval/Q-Align` | `q_align` | `environments/visual_quality.yml` |
| `eval/audiobox-aesthetics` | `audiobox` | `environments/audio_quality.yml` |
| `eval/Syncformer` | `syncformer` | `environments/avsync.yml` |
| `eval/Ocr` | `ocr` | `environments/text_rendering_quality.yml` |
| `eval/syncnet_python` | `syncnet` | `environments/lipsync.yml` |
| `eval/videophy/VIDEOPHY2` | `videophy` | `environments/low_level_physics.yml` |
| `eval/gemini_phy` | `mllm` | `environments/mllm.yml` |
| `eval/plot_matching` | `mllm` | `environments/mllm.yml` |
| `eval/facial_consistency` | `face` | `environments/facial_consistency.yml` |
| `eval/music_check` | `music` | `environments/pitch_accuracy.yml` |
| `eval/speech` | `whisper` | `environments/speech_quality.yml` |
