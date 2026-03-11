import argparse
import glob
import json
import os
import random
import re
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from generation.clients import BaseGenerationClient

try:
    from tqdm import tqdm
except Exception:
    tqdm = None


def safe_filename(name: str, max_len: int = 180) -> str:
    name = str(name).strip()
    name = re.sub(r"[\/\\\:\*\?\"\<\>\|\n\r\t]", "_", name)
    name = re.sub(r"\s+", " ", name).strip()
    if not name:
        name = "untitled"
    if len(name) > max_len:
        name = name[:max_len].rstrip()
    return name


def load_json_list(path: str) -> List[Dict[str, Any]]:
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, list):
        raise ValueError("%s must be a JSON list, got %s" % (path, type(data)))
    return data


def parse_gpu_ids(value: str) -> List[str]:
    if not value:
        return []
    parts = [p.strip() for p in value.split(",") if p.strip()]
    ids: List[int] = []
    for part in parts:
        if "-" in part:
            start_str, end_str = part.split("-", 1)
            start = int(start_str.strip())
            end = int(end_str.strip())
            if end < start:
                raise ValueError("Invalid GPU range '%s' (end < start)." % part)
            ids.extend(range(start, end + 1))
        else:
            ids.append(int(part))
    return [str(i) for i in ids]


@dataclass
class PromptTask:
    category: str
    content: str
    prompt: str
    extra_kwargs: Dict[str, Any] = field(default_factory=dict)


class ThreadLocalClientPool:
    def __init__(self, provider: str):
        self.provider = provider
        self._local = threading.local()

    def get(self) -> BaseGenerationClient:
        client = getattr(self._local, "client", None)
        if client is None:
            client = create_client(self.provider)
            self._local.client = client
        return client


def create_client(provider: str) -> BaseGenerationClient:
    provider = (provider or "").strip().lower()
    if provider in ("sora2", "sora-2"):
        from generation.clients.sora2 import Sora2Client

        return Sora2Client()
    if provider in ("kling26", "kling-v2-6", "kling"):
        from generation.clients.kling26 import Kling26Client

        return Kling26Client()
    if provider in ("wan26", "wan2.6", "wan"):
        from generation.clients.wan26 import Wan26Client

        return Wan26Client()
    if provider in ("seedance", "seedance15pro", "doubao-seedance"):
        from generation.clients.seedance import SeedanceClient

        return SeedanceClient()
    if provider in ("ltx2", "ltx-2"):
        from generation.clients.ltx2 import Ltx2Client

        return Ltx2Client()
    if provider in ("ovi",):
        from generation.clients.ovi import OviClient

        return OviClient()
    if provider in ("mova",):
        from generation.clients.mova import MovaClient

        return MovaClient()
    raise ValueError(
        "Unknown provider '%s'. Supported providers: sora2, kling26, wan26, seedance, ltx2, ovi, mova."
        % provider
    )


def build_tasks(prompts_dir: Path) -> List[PromptTask]:
    json_files = sorted(glob.glob(str(prompts_dir / "*.json")))
    if not json_files:
        raise FileNotFoundError("No JSON files found in %s" % prompts_dir.resolve())

    tasks: List[PromptTask] = []
    for jf in json_files:
        category = Path(jf).stem
        items = load_json_list(jf)
        for item in items:
            content = item.get("content")
            prompt = item.get("prompt")
            if not content or not prompt:
                continue
            extra_kwargs = {
                k: v for k, v in item.items() if k not in ("content", "prompt") and v is not None
            }
            tasks.append(
                PromptTask(
                    category=category,
                    content=content,
                    prompt=prompt,
                    extra_kwargs=extra_kwargs,
                )
            )
    return tasks


def output_path_for_task(
    out_root: Path,
    task: PromptTask,
    task_type: str,
    image_ext: str,
) -> Path:
    out_dir = out_root / task.category
    out_dir.mkdir(parents=True, exist_ok=True)

    if task_type == "video_generation":
        ext = ".mp4"
    else:
        ext = image_ext if image_ext.startswith(".") else "." + image_ext
    return out_dir / ("%s%s" % (safe_filename(task.content), ext))


def is_existing_output_valid(path: Path, min_bytes: int = 1024) -> bool:
    return path.exists() and path.stat().st_size > min_bytes


def run_one_task(
    client_pool: ThreadLocalClientPool,
    task: PromptTask,
    out_root: Path,
    task_type: str,
    provider_kwargs: Dict[str, Any],
    rerun_existing: bool,
    image_ext: str,
    max_attempts: int = 4,
) -> Tuple[str, str]:
    """
    Returns (status, message). status in {"ok","skipped","failed"}.
    """
    out_path = output_path_for_task(out_root, task, task_type, image_ext)

    if (not rerun_existing) and is_existing_output_valid(out_path):
        return "skipped", "[%s] exists: %s" % (task.category, out_path.name)

    last_err: Optional[Exception] = None
    for attempt in range(1, max_attempts + 1):
        try:
            client = client_pool.get()
            call_kwargs = dict(provider_kwargs)
            call_kwargs.update(task.extra_kwargs)
            if task_type == "video_generation":
                artifact = client.video_generation(prompt=task.prompt, **call_kwargs)
            elif task_type == "image_generation":
                call_kwargs.setdefault("image_ext", image_ext)
                artifact = client.image_generation(prompt=task.prompt, **call_kwargs)
            else:
                raise ValueError("Unsupported task_type: %s" % task_type)

            final_out_path = out_path.with_suffix(artifact.extension)
            tmp_path = final_out_path.with_suffix(final_out_path.suffix + ".part")
            with open(tmp_path, "wb") as f:
                f.write(artifact.data)
            os.replace(tmp_path, final_out_path)
            return "ok", "[%s] saved: %s" % (task.category, final_out_path.name)

        except (NotImplementedError, ValueError) as e:
            return (
                "failed",
                "[%s] FAILED (non-retriable): %s | err=%s" % (task.category, task.content, e),
            )
        except Exception as e:
            last_err = e
            sleep_s = min(60, (2 ** (attempt - 1)) * 2) + random.uniform(0, 1.0)
            print(
                "Retry %s/%s [%s] %s: %s (sleep %.1fs)"
                % (attempt, max_attempts, task.category, task.content, e, sleep_s)
            )
            time.sleep(sleep_s)

    return "failed", "[%s] FAILED: %s | last_err=%s" % (task.category, task.content, last_err)


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(
        description=(
            "Batch generation framework for AVGen-Bench prompts. "
            "Default provider=sora2 with extensible client interface."
        )
    )
    ap.add_argument("--provider", type=str, default="sora2", help="Generation provider key.")
    ap.add_argument(
        "--task_type",
        type=str,
        default="video_generation",
        choices=["video_generation", "image_generation"],
        help="Choose between video or image generation interface.",
    )
    ap.add_argument("--prompts_dir", type=str, default="./prompts")
    ap.add_argument("--out_dir", type=str, default="./generated_videos/sora2")
    ap.add_argument("--concurrency", type=int, default=2)
    ap.add_argument("--seconds", type=int, default=12, help="Used by video generation.")
    ap.add_argument("--size", type=str, default="1280x720")
    ap.add_argument("--image_ext", type=str, default=".png")
    ap.add_argument("--max_attempts", type=int, default=4)
    ap.add_argument(
        "--rerun_existing",
        action="store_true",
        help="If set, regenerate even when output already exists.",
    )
    ap.add_argument(
        "--gpu_ids",
        type=str,
        default="",
        help="Comma-separated GPU ids for local inference (e.g., '0,1,2,3' or '0-3').",
    )
    ap.add_argument("--kling_model_name", type=str, default="kling-v2-6")
    ap.add_argument("--kling_duration", type=str, default="10")
    ap.add_argument("--kling_aspect_ratio", type=str, default="16:9")
    ap.add_argument("--kling_mode", type=str, default="pro")
    ap.add_argument("--kling_sound", type=str, default="on")
    ap.add_argument("--kling_poll_interval", type=float, default=6.0)
    ap.add_argument("--kling_timeout_s", type=int, default=1800)

    ap.add_argument("--wan_duration", type=int, default=10)
    ap.add_argument("--wan_size", type=str, default="1280*720")
    ap.add_argument("--wan_shot_type", type=str, default="multi")
    ap.add_argument("--wan_poll_interval", type=float, default=8.0)
    ap.add_argument("--wan_timeout_s", type=int, default=1800)
    ap.add_argument(
        "--wan_no_prompt_extend",
        action="store_true",
        help="Disable prompt_extend for wan26.",
    )

    ap.add_argument("--seedance_model_name", type=str, default="doubao-seedance-1-5-pro-251215")
    ap.add_argument("--seedance_resolution", type=str, default="720p")
    ap.add_argument("--seedance_ratio", type=str, default="16:9")
    ap.add_argument("--seedance_duration", type=int, default=10)
    ap.add_argument("--seedance_watermark", action="store_true", default=False)
    ap.add_argument("--seedance_image_url", type=str, default=None)
    ap.add_argument("--seedance_poll_interval", type=float, default=3.0)
    ap.add_argument("--seedance_timeout_s", type=int, default=1800)

    ap.add_argument(
        "--ltx2_repo_dir",
        type=str,
        default=str(Path(__file__).resolve().parents[1] / "third_party" / "LTX-2"),
    )
    ap.add_argument("--ltx2_pipeline", type=str, default="distilled")
    ap.add_argument("--ltx2_distilled_checkpoint_path", type=str, default=None)
    ap.add_argument("--ltx2_spatial_upsampler_path", type=str, default=None)
    ap.add_argument("--ltx2_gemma_root", type=str, default=None)
    ap.add_argument("--ltx2_size", type=str, default="1280x704")
    ap.add_argument("--ltx2_seed", type=int, default=100)
    ap.add_argument("--ltx2_num_frames", type=int, default=241)
    ap.add_argument("--ltx2_frame_rate", type=float, default=24.0)
    ap.add_argument("--ltx2_quantization", type=str, default=None)
    ap.add_argument("--ltx2_enhance_prompt", action="store_true", default=False)
    ap.add_argument("--ltx2_timeout_s", type=int, default=7200)
    ap.add_argument("--ltx2_python_bin", type=str, default=None)

    ap.add_argument("--ovi_repo_dir", type=str, default=str(Path(__file__).resolve().parents[1] / "third_party" / "Ovi"))
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
    ap.add_argument("--ovi_cpu_offload", action="store_true", default=False)
    ap.add_argument("--ovi_fp8", action="store_true", default=False)
    ap.add_argument(
        "--ovi_video_negative_prompt",
        type=str,
        default="jitter, bad hands, blur, distortion",
    )
    ap.add_argument(
        "--ovi_audio_negative_prompt",
        type=str,
        default="robotic, muffled, echo, distorted",
    )
    ap.add_argument("--ovi_timeout_s", type=int, default=7200)
    ap.add_argument("--ovi_python_bin", type=str, default=None)
    ap.add_argument("--ovi_torchrun_nproc", type=int, default=1)

    ap.add_argument(
        "--image_dir",
        type=str,
        default=None,
        help="Root directory for first-frame images (used by TI2AV providers like mova).",
    )

    ap.add_argument(
        "--mova_repo_dir",
        type=str,
        default=str(Path(__file__).resolve().parents[1] / "third_party" / "MOVA"),
    )
    ap.add_argument("--mova_ckpt_path", type=str, default=None)
    ap.add_argument("--mova_negative_prompt", type=str, default=None)
    ap.add_argument("--mova_num_frames", type=int, default=193)
    ap.add_argument("--mova_fps", type=float, default=24.0)
    ap.add_argument("--mova_height", type=int, default=720)
    ap.add_argument("--mova_width", type=int, default=1280)
    ap.add_argument("--mova_seed", type=int, default=42)
    ap.add_argument("--mova_num_inference_steps", type=int, default=50)
    ap.add_argument("--mova_cfg_scale", type=float, default=5.0)
    ap.add_argument("--mova_sigma_shift", type=float, default=5.0)
    ap.add_argument("--mova_cp_size", type=int, default=1)
    ap.add_argument("--mova_attn_type", type=str, default="fa")
    ap.add_argument("--mova_offload", type=str, default="none", choices=("none", "cpu", "group"))
    ap.add_argument("--mova_offload_to_disk_path", type=str, default=None)
    ap.add_argument("--mova_remove_video_dit", action="store_true", default=False)
    ap.add_argument("--mova_timeout_s", type=int, default=7200)
    ap.add_argument("--mova_torchrun_bin", type=str, default=None)
    ap.add_argument("--mova_python_bin", type=str, default=None)
    return ap.parse_args()


def build_provider_kwargs(args: argparse.Namespace) -> Dict[str, Any]:
    provider = (args.provider or "").strip().lower()
    if provider in ("sora2", "sora-2"):
        return {"seconds": args.seconds, "size": args.size}
    if provider in ("kling26", "kling-v2-6", "kling"):
        return {
            "model_name": args.kling_model_name,
            "duration": args.kling_duration,
            "aspect_ratio": args.kling_aspect_ratio,
            "mode": args.kling_mode,
            "sound": args.kling_sound,
            "poll_interval": args.kling_poll_interval,
            "timeout_s": args.kling_timeout_s,
        }
    if provider in ("wan26", "wan2.6", "wan"):
        shot_type = args.wan_shot_type.strip() if args.wan_shot_type else None
        if shot_type == "":
            shot_type = None
        return {
            "duration": args.wan_duration,
            "size": args.wan_size,
            "prompt_extend": (not args.wan_no_prompt_extend),
            "shot_type": shot_type,
            "poll_interval": args.wan_poll_interval,
            "timeout_s": args.wan_timeout_s,
        }
    if provider in ("seedance", "seedance15pro", "doubao-seedance"):
        return {
            "model_name": args.seedance_model_name,
            "resolution": args.seedance_resolution,
            "ratio": args.seedance_ratio,
            "duration": args.seedance_duration,
            "watermark": args.seedance_watermark,
            "image_url": args.seedance_image_url,
            "poll_interval": args.seedance_poll_interval,
            "timeout_s": args.seedance_timeout_s,
        }
    if provider in ("ltx2", "ltx-2"):
        os.environ["LTX2_REPO_DIR"] = args.ltx2_repo_dir
        return {
            "pipeline": args.ltx2_pipeline,
            "distilled_checkpoint_path": args.ltx2_distilled_checkpoint_path,
            "spatial_upsampler_path": args.ltx2_spatial_upsampler_path,
            "gemma_root": args.ltx2_gemma_root,
            "size": args.ltx2_size,
            "seed": args.ltx2_seed,
            "num_frames": args.ltx2_num_frames,
            "frame_rate": args.ltx2_frame_rate,
            "quantization": args.ltx2_quantization,
            "enhance_prompt": args.ltx2_enhance_prompt,
            "timeout_s": args.ltx2_timeout_s,
            "python_bin": args.ltx2_python_bin,
        }
    if provider in ("ovi",):
        os.environ["OVI_REPO_DIR"] = args.ovi_repo_dir
        return {
            "ckpt_dir": args.ovi_ckpt_dir,
            "model_name": args.ovi_model_name,
            "mode": args.ovi_mode,
            "size": args.ovi_size,
            "sample_steps": args.ovi_sample_steps,
            "solver_name": args.ovi_solver_name,
            "shift": args.ovi_shift,
            "seed": args.ovi_seed,
            "audio_guidance_scale": args.ovi_audio_guidance_scale,
            "video_guidance_scale": args.ovi_video_guidance_scale,
            "slg_layer": args.ovi_slg_layer,
            "sp_size": args.ovi_sp_size,
            "cpu_offload": args.ovi_cpu_offload,
            "fp8": args.ovi_fp8,
            "video_negative_prompt": args.ovi_video_negative_prompt,
            "audio_negative_prompt": args.ovi_audio_negative_prompt,
            "timeout_s": args.ovi_timeout_s,
            "python_bin": args.ovi_python_bin,
            "torchrun_nproc": args.ovi_torchrun_nproc,
        }
    if provider in ("mova",):
        os.environ["MOVA_REPO_DIR"] = args.mova_repo_dir
        return {
            "ckpt_path": args.mova_ckpt_path,
            "negative_prompt": args.mova_negative_prompt,
            "num_frames": args.mova_num_frames,
            "fps": args.mova_fps,
            "height": args.mova_height,
            "width": args.mova_width,
            "seed": args.mova_seed,
            "num_inference_steps": args.mova_num_inference_steps,
            "cfg_scale": args.mova_cfg_scale,
            "sigma_shift": args.mova_sigma_shift,
            "cp_size": args.mova_cp_size,
            "attn_type": args.mova_attn_type,
            "offload": args.mova_offload,
            "offload_to_disk_path": args.mova_offload_to_disk_path,
            "remove_video_dit": args.mova_remove_video_dit,
            "timeout_s": args.mova_timeout_s,
            "torchrun_bin": args.mova_torchrun_bin,
            "python_bin": args.mova_python_bin,
        }
    return {}


def run(args: argparse.Namespace) -> int:
    prompts_dir = Path(args.prompts_dir)
    provider = (args.provider or "").strip().lower()
    default_out_dir = "./generated_videos/sora2"
    if args.out_dir == default_out_dir:
        normalized = (
            "kling26"
            if provider in ("kling26", "kling-v2-6", "kling")
            else "wan26"
            if provider in ("wan26", "wan2.6", "wan")
            else "seedance"
            if provider in ("seedance", "seedance15pro", "doubao-seedance")
            else "ltx2"
            if provider in ("ltx2", "ltx-2")
            else "ovi"
            if provider in ("ovi",)
            else "mova"
            if provider in ("mova",)
            else "sora2"
        )
        out_root = Path("./generated_videos") / normalized
    else:
        out_root = Path(args.out_dir)
    out_root.mkdir(parents=True, exist_ok=True)

    tasks = build_tasks(prompts_dir)
    total = len(tasks)
    if total == 0:
        raise RuntimeError("No valid prompt items loaded from %s" % prompts_dir.resolve())

    gpu_ids = parse_gpu_ids(args.gpu_ids)
    if gpu_ids:
        if args.concurrency > len(gpu_ids):
            print(
                "Warning: concurrency=%s > gpu_ids=%s. Tasks will share GPUs."
                % (args.concurrency, ",".join(gpu_ids))
            )
        # For MOVA with cp_size>1, a single task spans multiple GPUs. Do not auto-assign a single gpu_id.
        if provider in ("mova",) and int(getattr(args, "mova_cp_size", 1) or 1) > 1:
            print("Warning: provider=mova with mova_cp_size>1 ignores --gpu_ids auto-assignment.")
        else:
            for idx, task in enumerate(tasks):
                if "gpu_id" in task.extra_kwargs or "cuda_visible_devices" in task.extra_kwargs:
                    continue
                task.extra_kwargs["gpu_id"] = gpu_ids[idx % len(gpu_ids)]

    if provider in ("mova",):
        if not args.image_dir:
            # Allow explicit per-task ref_path in JSON, but require a global image_dir otherwise.
            missing = [t for t in tasks if "ref_path" not in t.extra_kwargs]
            if missing:
                raise ValueError(
                    "provider=mova requires --image_dir (first-frame images root) unless each prompt item includes 'ref_path'."
                )
        else:
            image_root = Path(args.image_dir).expanduser().resolve()
            if not image_root.exists():
                raise FileNotFoundError(f"--image_dir not found: {image_root}")

            def _resolve_image(task: PromptTask) -> str:
                # Match image generators that save as: <image_root>/<category>/<safe_filename(content)>.<ext>
                base = safe_filename(task.content)
                candidates = []
                for ext in (".png", ".jpg", ".jpeg", ".webp", ".bmp"):
                    candidates.append(image_root / task.category / f"{base}{ext}")
                    candidates.append(image_root / f"{base}{ext}")
                for p in candidates:
                    if p.exists() and p.stat().st_size > 0:
                        return str(p)
                # Fallback: any file with matching stem under category.
                cat_dir = image_root / task.category
                if cat_dir.exists():
                    for p in sorted(cat_dir.glob(base + ".*")):
                        if p.is_file() and p.stat().st_size > 0:
                            return str(p)
                raise FileNotFoundError(f"First-frame image not found for '{task.category}/{task.content}' under {image_root}")

            for task in tasks:
                task.extra_kwargs.setdefault("ref_path", _resolve_image(task))

    print(
        "Found %s prompts. provider=%s task_type=%s concurrency=%s rerun_existing=%s"
        % (total, args.provider, args.task_type, args.concurrency, args.rerun_existing)
    )

    provider_kwargs = build_provider_kwargs(args)
    client_pool = ThreadLocalClientPool(provider=args.provider)

    ok = 0
    skipped = 0
    failed = 0

    pbar = None
    if tqdm is not None:
        pbar = tqdm(total=total, desc="Generating", dynamic_ncols=True)
    else:
        print("Tip: pip install tqdm for a progress bar.")

    with ThreadPoolExecutor(max_workers=args.concurrency) as ex:
        futures = [
            ex.submit(
                run_one_task,
                client_pool,
                task,
                out_root,
                args.task_type,
                provider_kwargs,
                args.rerun_existing,
                args.image_ext,
                args.max_attempts,
            )
            for task in tasks
        ]

        for fut in as_completed(futures):
            status, msg = fut.result()
            if status == "ok":
                ok += 1
            elif status == "skipped":
                skipped += 1
            else:
                failed += 1

            if pbar is not None:
                pbar.update(1)
                pbar.set_postfix({"ok": ok, "skipped": skipped, "failed": failed})
                tqdm.write(msg)
            else:
                done = ok + skipped + failed
                print("[%s/%s] %s" % (done, total, msg))

    if pbar is not None:
        pbar.close()

    print("Done. ok=%s skipped=%s failed=%s" % (ok, skipped, failed))
    return 0 if failed == 0 else 1


def main() -> None:
    args = parse_args()
    raise SystemExit(run(args))


if __name__ == "__main__":
    main()
