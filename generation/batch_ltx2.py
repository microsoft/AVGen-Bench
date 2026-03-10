#!/usr/bin/env python3

import argparse
import subprocess
import sys
from pathlib import Path


def main() -> None:
    default_repo = Path(__file__).resolve().parents[1] / "third_party" / "LTX-2"
    ap = argparse.ArgumentParser(description="Compatibility wrapper for provider=ltx2.")
    ap.add_argument("--prompts_dir", type=str, default="./prompts")
    ap.add_argument("--out_dir", type=str, default="./generated_videos/ltx2")
    ap.add_argument("--concurrency", type=int, default=1)
    ap.add_argument("--ltx2_repo_dir", type=str, default=str(default_repo))
    ap.add_argument("--ltx2_pipeline", type=str, default="distilled")
    ap.add_argument("--ltx2_distilled_checkpoint_path", type=str, default=None)
    ap.add_argument("--ltx2_spatial_upsampler_path", type=str, default=None)
    ap.add_argument("--ltx2_gemma_root", type=str, default=None)
    ap.add_argument("--ltx2_size", type=str, default="1280x704")
    ap.add_argument("--ltx2_seed", type=int, default=100)
    ap.add_argument("--ltx2_num_frames", type=int, default=241)
    ap.add_argument("--ltx2_frame_rate", type=float, default=24.0)
    ap.add_argument("--ltx2_quantization", type=str, default=None)
    ap.add_argument("--ltx2_enhance_prompt", action="store_true")
    ap.add_argument("--ltx2_timeout_s", type=int, default=7200)
    ap.add_argument("--ltx2_python_bin", type=str, default=None)
    ap.add_argument("--max_attempts", type=int, default=2)
    ap.add_argument("--rerun_existing", action="store_true")
    args = ap.parse_args()

    root = Path(__file__).resolve().parents[1]
    cmd = [
        sys.executable,
        str(root / "batch_generate.py"),
        "--provider",
        "ltx2",
        "--prompts_dir",
        args.prompts_dir,
        "--out_dir",
        args.out_dir,
        "--concurrency",
        str(args.concurrency),
        "--ltx2_repo_dir",
        args.ltx2_repo_dir,
        "--ltx2_pipeline",
        args.ltx2_pipeline,
        "--ltx2_size",
        args.ltx2_size,
        "--ltx2_seed",
        str(args.ltx2_seed),
        "--ltx2_num_frames",
        str(args.ltx2_num_frames),
        "--ltx2_frame_rate",
        str(args.ltx2_frame_rate),
        "--ltx2_timeout_s",
        str(args.ltx2_timeout_s),
        "--max_attempts",
        str(args.max_attempts),
    ]
    if args.ltx2_distilled_checkpoint_path:
        cmd.extend(["--ltx2_distilled_checkpoint_path", args.ltx2_distilled_checkpoint_path])
    if args.ltx2_spatial_upsampler_path:
        cmd.extend(["--ltx2_spatial_upsampler_path", args.ltx2_spatial_upsampler_path])
    if args.ltx2_gemma_root:
        cmd.extend(["--ltx2_gemma_root", args.ltx2_gemma_root])
    if args.ltx2_quantization:
        cmd.extend(["--ltx2_quantization", args.ltx2_quantization])
    if args.ltx2_python_bin:
        cmd.extend(["--ltx2_python_bin", args.ltx2_python_bin])
    if args.ltx2_enhance_prompt:
        cmd.append("--ltx2_enhance_prompt")
    if args.rerun_existing:
        cmd.append("--rerun_existing")

    raise SystemExit(subprocess.call(cmd, cwd=root))


if __name__ == "__main__":
    main()
