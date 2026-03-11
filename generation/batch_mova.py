#!/usr/bin/env python3

import argparse
import subprocess
import sys
from pathlib import Path


def main() -> None:
    default_repo = Path(__file__).resolve().parents[1] / "third_party" / "MOVA"
    ap = argparse.ArgumentParser(description="Compatibility wrapper for provider=mova (TI2AV).")
    ap.add_argument("--prompts_dir", type=str, default="./prompts")
    ap.add_argument("--out_dir", type=str, default="./generated_videos/mova")
    ap.add_argument("--concurrency", type=int, default=1)
    ap.add_argument("--image_dir", type=str, required=True, help="Root dir of first-frame images.")

    ap.add_argument("--mova_repo_dir", type=str, default=str(default_repo))
    ap.add_argument("--mova_ckpt_path", type=str, required=True)
    ap.add_argument("--mova_height", type=int, default=720)
    ap.add_argument("--mova_width", type=int, default=1280)
    ap.add_argument("--mova_num_frames", type=int, default=193)
    ap.add_argument("--mova_fps", type=float, default=24.0)
    ap.add_argument("--mova_seed", type=int, default=42)
    ap.add_argument("--mova_num_inference_steps", type=int, default=50)
    ap.add_argument("--mova_cfg_scale", type=float, default=5.0)
    ap.add_argument("--mova_sigma_shift", type=float, default=5.0)
    ap.add_argument("--mova_cp_size", type=int, default=1)
    ap.add_argument("--mova_attn_type", type=str, default="fa")
    ap.add_argument("--mova_offload", type=str, default="none")
    ap.add_argument("--mova_offload_to_disk_path", type=str, default=None)
    ap.add_argument("--mova_remove_video_dit", action="store_true", default=False)
    ap.add_argument("--mova_timeout_s", type=int, default=7200)
    ap.add_argument("--mova_torchrun_bin", type=str, default=None)
    ap.add_argument("--mova_python_bin", type=str, default=None)

    ap.add_argument("--max_attempts", type=int, default=2)
    ap.add_argument("--rerun_existing", action="store_true")
    ap.add_argument("--gpu_ids", type=str, default="")
    args = ap.parse_args()

    root = Path(__file__).resolve().parents[1]
    cmd = [
        sys.executable,
        str(root / "batch_generate.py"),
        "--provider",
        "mova",
        "--task_type",
        "video_generation",
        "--prompts_dir",
        args.prompts_dir,
        "--out_dir",
        args.out_dir,
        "--concurrency",
        str(args.concurrency),
        "--image_dir",
        args.image_dir,
        "--mova_repo_dir",
        args.mova_repo_dir,
        "--mova_ckpt_path",
        args.mova_ckpt_path,
        "--mova_height",
        str(args.mova_height),
        "--mova_width",
        str(args.mova_width),
        "--mova_num_frames",
        str(args.mova_num_frames),
        "--mova_fps",
        str(args.mova_fps),
        "--mova_seed",
        str(args.mova_seed),
        "--mova_num_inference_steps",
        str(args.mova_num_inference_steps),
        "--mova_cfg_scale",
        str(args.mova_cfg_scale),
        "--mova_sigma_shift",
        str(args.mova_sigma_shift),
        "--mova_cp_size",
        str(args.mova_cp_size),
        "--mova_attn_type",
        args.mova_attn_type,
        "--mova_offload",
        args.mova_offload,
        "--mova_timeout_s",
        str(args.mova_timeout_s),
        "--max_attempts",
        str(args.max_attempts),
    ]
    if args.mova_offload_to_disk_path:
        cmd.extend(["--mova_offload_to_disk_path", args.mova_offload_to_disk_path])
    if args.mova_remove_video_dit:
        cmd.append("--mova_remove_video_dit")
    if args.mova_torchrun_bin:
        cmd.extend(["--mova_torchrun_bin", args.mova_torchrun_bin])
    if args.mova_python_bin:
        cmd.extend(["--mova_python_bin", args.mova_python_bin])
    if args.gpu_ids:
        cmd.extend(["--gpu_ids", args.gpu_ids])
    if args.rerun_existing:
        cmd.append("--rerun_existing")

    raise SystemExit(subprocess.call(cmd, cwd=root))


if __name__ == "__main__":
    main()

