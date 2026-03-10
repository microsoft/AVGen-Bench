#!/usr/bin/env python3

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

import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from generation.clients.emu35 import Emu35Client

try:
    from tqdm import tqdm
except Exception:
    tqdm = None


PROMPT_PREFIX = "generate the first frame of the following video:"


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


def output_path_for_task(out_root: Path, task: PromptTask, image_ext: str) -> Path:
    out_dir = out_root / task.category
    out_dir.mkdir(parents=True, exist_ok=True)
    ext = image_ext if image_ext.startswith(".") else "." + image_ext
    return out_dir / ("%s%s" % (safe_filename(task.content), ext))


def is_existing_output_valid(path: Path, min_bytes: int = 1024) -> bool:
    return path.exists() and path.stat().st_size > min_bytes


def build_prompt(user_prompt: str) -> str:
    return f"{PROMPT_PREFIX}{user_prompt}"


class ThreadLocalClientPool:
    def __init__(self, repo_dir: Optional[str]) -> None:
        self.repo_dir = repo_dir
        self._local = threading.local()

    def get(self) -> Emu35Client:
        client = getattr(self._local, "client", None)
        if client is None:
            client = Emu35Client(repo_dir=self.repo_dir)
            self._local.client = client
        return client


def run_one_task(
    client_pool: ThreadLocalClientPool,
    task: PromptTask,
    out_root: Path,
    image_ext: str,
    provider_kwargs: Dict[str, Any],
    rerun_existing: bool,
    max_attempts: int = 2,
) -> Tuple[str, str]:
    out_path = output_path_for_task(out_root, task, image_ext)

    if (not rerun_existing) and is_existing_output_valid(out_path):
        return "skipped", "[%s] exists: %s" % (task.category, out_path.name)

    last_err: Optional[Exception] = None
    for attempt in range(1, max_attempts + 1):
        try:
            client = client_pool.get()
            call_kwargs = dict(provider_kwargs)
            call_kwargs.update(task.extra_kwargs)
            call_kwargs.setdefault("image_ext", image_ext)
            prompt = build_prompt(task.prompt)
            artifact = client.image_generation(prompt=prompt, **call_kwargs)

            final_out_path = out_path.with_suffix(artifact.extension)
            tmp_path = final_out_path.with_suffix(final_out_path.suffix + ".part")
            with open(tmp_path, "wb") as f:
                f.write(artifact.data)
            os.replace(tmp_path, final_out_path)
            return "ok", "[%s] saved: %s" % (task.category, final_out_path.name)

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
    ap = argparse.ArgumentParser(description="Batch image generation with Emu3.5.")
    ap.add_argument("--prompts_dir", type=str, default="./prompts")
    ap.add_argument("--out_dir", type=str, default="./generated_images/emu35")
    ap.add_argument("--concurrency", type=int, default=1)
    ap.add_argument("--image_ext", type=str, default=".png")
    ap.add_argument("--max_attempts", type=int, default=2)
    ap.add_argument("--rerun_existing", action="store_true")
    ap.add_argument("--gpu_ids", type=str, default="")

    ap.add_argument("--emu35_repo_dir", type=str, default=None)
    ap.add_argument("--model_path", type=str, default="BAAI/Emu3.5-Image")
    ap.add_argument("--vq_path", type=str, default="BAAI/Emu3.5-VisionTokenizer")
    ap.add_argument("--tokenizer_path", type=str, default="./src/tokenizer_emu3_ibq")
    ap.add_argument("--vq_type", type=str, default="ibq")
    ap.add_argument("--task_type", type=str, default="t2i")
    ap.add_argument("--use_image", action="store_true", default=False)
    ap.add_argument("--aspect_ratio", type=str, default="default")
    ap.add_argument("--size", type=str, default="1280x720")
    ap.add_argument("--hf_device", type=str, default="auto")
    ap.add_argument("--vq_device", type=str, default="cuda:0")
    ap.add_argument("--classifier_free_guidance", type=float, default=5.0)
    ap.add_argument("--max_new_tokens", type=int, default=5120)
    ap.add_argument("--image_area", type=int, default=1048576)
    ap.add_argument("--image_cfg_scale", type=float, default=1.0)
    ap.add_argument("--seed", type=int, default=6666)
    ap.add_argument("--timeout_s", type=int, default=7200)
    ap.add_argument("--python_bin", type=str, default=None)
    ap.add_argument("--cuda_visible_devices", type=str, default=None)
    ap.add_argument(
        "--use_vllm",
        action="store_true",
        default=False,
        help="Use Emu3.5 vLLM inference script (inference_vllm.py).",
    )
    return ap.parse_args()


def build_provider_kwargs(args: argparse.Namespace) -> Dict[str, Any]:
    return {
        "model_path": args.model_path,
        "vq_path": args.vq_path,
        "tokenizer_path": args.tokenizer_path,
        "vq_type": args.vq_type,
        "task_type": args.task_type,
        "use_image": args.use_image,
        "aspect_ratio": args.aspect_ratio,
        "size": args.size,
        "hf_device": args.hf_device,
        "vq_device": args.vq_device,
        "classifier_free_guidance": args.classifier_free_guidance,
        "max_new_tokens": args.max_new_tokens,
        "image_area": args.image_area,
        "image_cfg_scale": args.image_cfg_scale,
        "seed": args.seed,
        "timeout_s": args.timeout_s,
        "python_bin": args.python_bin,
        "cuda_visible_devices": args.cuda_visible_devices,
        "use_vllm": args.use_vllm,
    }


def main() -> None:
    args = parse_args()

    prompts_dir = Path(args.prompts_dir)
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
        for idx, task in enumerate(tasks):
            if "gpu_id" in task.extra_kwargs or "cuda_visible_devices" in task.extra_kwargs:
                continue
            task.extra_kwargs["gpu_id"] = gpu_ids[idx % len(gpu_ids)]

    print(
        "Found %s prompts. provider=emu35 concurrency=%s rerun_existing=%s"
        % (total, args.concurrency, args.rerun_existing)
    )

    provider_kwargs = build_provider_kwargs(args)
    client_pool = ThreadLocalClientPool(repo_dir=args.emu35_repo_dir)

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
                args.image_ext,
                provider_kwargs,
                args.rerun_existing,
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
    raise SystemExit(0 if failed == 0 else 1)


if __name__ == "__main__":
    main()
