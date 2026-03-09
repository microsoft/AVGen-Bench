#!/bin/bash
python batch_eval.py \
  --video_root /path/to/video_generation/kling26 \
  --save_csv /path/to/avgenbench/syncnet/kling26/result.csv \
  --conf_th 1.0 \
  --inference_py inference.py

python batch_eval.py \
  --video_root /path/to/video_generation/ovi \
  --save_csv /path/to/avgenbench/syncnet/ovi/result.csv \
  --conf_th 1.0 \
  --inference_py inference.py

python batch_eval.py \
  --video_root /path/to/video_generation/ltx2 \
  --save_csv /path/to/avgenbench/syncnet/ltx2/result.csv \
  --conf_th 1.0 \
  --inference_py inference.py

python batch_eval.py \
  --video_root /path/to/video_generation/wan26_generated \
  --save_csv /path/to/avgenbench/syncnet/wan26/result.csv \
  --conf_th 1.0 \
  --inference_py inference.py

python batch_eval.py \
  --video_root /path/to/video_generation/sora2_generated \
  --save_csv /path/to/avgenbench/syncnet/sora2/result.csv \
  --conf_th 1.0 \
  --inference_py inference.py