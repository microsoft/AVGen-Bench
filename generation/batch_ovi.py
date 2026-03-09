#!/usr/bin/env python3

import argparse
import subprocess
import sys
from pathlib import Path


def main() -> None:
    default_ovi_repo = Path(__file__).resolve().parents[1] / "third_party" / "Ovi"
    ap = argparse.ArgumentParser(description="Compatibility wrapper for provider=ovi.")
    ap.add_argument("--prompts_dir", type=str, default="./prompts")
    ap.add_argument("--out_dir", type=str, default="./generated_videos/ovi")
    ap.add_argument("--concurrency", type=int, default=1)
    ap.add_argument("--ovi_repo_dir", type=str, default=str(default_ovi_repo))
    ap.add_argument("--ovi_ckpt_dir", type=str, default=None)
    ap.add_argument("--ovi_model_name", type=str, default="960x960_10s")
    ap.add_argument("--ovi_mode", type=str, default="t2v")
    ap.add_argument("--ovi_size", type=str, default="1280x720")
    ap.add_argument("--ovi_sample_steps", type=int, default=50)
    ap.add_argument("--ovi_solver_name", type=str, default="unipc")
    ap.add_argument("--ovi_shift", type=float, default=5.0)
    ap.add_argument("--ovi_seed", type=int, default=100)
    ap.add_argument("--ovi_audio_guidance_scale", type=float, default=3.0)
    ap.add_argument("--ovi_video_guidance_scale", type=float, default=4.0)
    ap.add_argument("--ovi_slg_layer", type=int, default=11)
    ap.add_argument("--ovi_sp_size", type=int, default=1)
    ap.add_argument("--ovi_cpu_offload", action="store_true")
    ap.add_argument("--ovi_fp8", action="store_true")
    ap.add_argument("--ovi_video_negative_prompt", type=str, default="jitter, bad hands, blur, distortion")
    ap.add_argument("--ovi_audio_negative_prompt", type=str, default="robotic, muffled, echo, distorted")
    ap.add_argument("--ovi_timeout_s", type=int, default=7200)
    ap.add_argument("--ovi_python_bin", type=str, default=None)
    ap.add_argument("--ovi_torchrun_nproc", type=int, default=1)
    ap.add_argument("--max_attempts", type=int, default=2)
    ap.add_argument("--rerun_existing", action="store_true")
    args = ap.parse_args()

    root = Path(__file__).resolve().parents[1]
    cmd = [
        sys.executable,
        str(root / "batch_generate.py"),
        "--provider",
        "ovi",
        "--prompts_dir",
        args.prompts_dir,
        "--out_dir",
        args.out_dir,
        "--concurrency",
        str(args.concurrency),
        "--ovi_repo_dir",
        args.ovi_repo_dir,
        "--ovi_model_name",
        args.ovi_model_name,
        "--ovi_mode",
        args.ovi_mode,
        "--ovi_size",
        args.ovi_size,
        "--ovi_sample_steps",
        str(args.ovi_sample_steps),
        "--ovi_solver_name",
        args.ovi_solver_name,
        "--ovi_shift",
        str(args.ovi_shift),
        "--ovi_seed",
        str(args.ovi_seed),
        "--ovi_audio_guidance_scale",
        str(args.ovi_audio_guidance_scale),
        "--ovi_video_guidance_scale",
        str(args.ovi_video_guidance_scale),
        "--ovi_slg_layer",
        str(args.ovi_slg_layer),
        "--ovi_sp_size",
        str(args.ovi_sp_size),
        "--ovi_video_negative_prompt",
        args.ovi_video_negative_prompt,
        "--ovi_audio_negative_prompt",
        args.ovi_audio_negative_prompt,
        "--ovi_timeout_s",
        str(args.ovi_timeout_s),
        "--ovi_torchrun_nproc",
        str(args.ovi_torchrun_nproc),
        "--max_attempts",
        str(args.max_attempts),
    ]
    if args.ovi_ckpt_dir:
        cmd.extend(["--ovi_ckpt_dir", args.ovi_ckpt_dir])
    if args.ovi_python_bin:
        cmd.extend(["--ovi_python_bin", args.ovi_python_bin])
    if args.ovi_cpu_offload:
        cmd.append("--ovi_cpu_offload")
    if args.ovi_fp8:
        cmd.append("--ovi_fp8")
    if args.rerun_existing:
        cmd.append("--rerun_existing")

    raise SystemExit(subprocess.call(cmd, cwd=root))


if __name__ == "__main__":
    main()
