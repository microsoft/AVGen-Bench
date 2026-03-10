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

from generation.clients.nanobanana2 import Nanobanana2Client

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
    def __init__(self, model: str, api_key: Optional[str]) -> None:
        self.model = model
        self.api_key = api_key
        self._local = threading.local()

    def get(self) -> Nanobanana2Client:
        client = getattr(self._local, "client", None)
        if client is None:
            client = Nanobanana2Client(model=self.model, api_key=self.api_key)
            self._local.client = client
        return client


def run_one_task(
    client_pool: ThreadLocalClientPool,
    task: PromptTask,
    out_root: Path,
    image_ext: str,
    default_model: str,
    rerun_existing: bool,
    max_attempts: int = 4,
) -> Tuple[str, str]:
    out_path = output_path_for_task(out_root, task, image_ext)

    if (not rerun_existing) and is_existing_output_valid(out_path):
        return "skipped", "[%s] exists: %s" % (task.category, out_path.name)

    last_err: Optional[Exception] = None
    for attempt in range(1, max_attempts + 1):
        try:
            client = client_pool.get()
            call_kwargs = dict(task.extra_kwargs)
            call_kwargs.setdefault("image_ext", image_ext)
            call_kwargs.setdefault("model", default_model)
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
            sleep_s = min(30, (2 ** (attempt - 1)) * 2) + random.uniform(0, 1.0)
            print(
                "Retry %s/%s [%s] %s: %s (sleep %.1fs)"
                % (attempt, max_attempts, task.category, task.content, e, sleep_s)
            )
            time.sleep(sleep_s)

    return "failed", "[%s] FAILED: %s | last_err=%s" % (task.category, task.content, last_err)


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Batch image generation with nanobanana2.")
    ap.add_argument("--prompts_dir", type=str, default="./prompts")
    ap.add_argument("--out_dir", type=str, default="./generated_images/nanobanana2")
    ap.add_argument("--concurrency", type=int, default=2)
    ap.add_argument("--image_ext", type=str, default=".png")
    ap.add_argument("--model", type=str, default="gemini-3.1-flash-image-preview")
    ap.add_argument("--api_key", type=str, default=None)
    ap.add_argument("--max_attempts", type=int, default=4)
    ap.add_argument("--rerun_existing", action="store_true")
    return ap.parse_args()


def main() -> None:
    args = parse_args()

    prompts_dir = Path(args.prompts_dir)
    out_root = Path(args.out_dir)
    out_root.mkdir(parents=True, exist_ok=True)

    tasks = build_tasks(prompts_dir)
    total = len(tasks)
    if total == 0:
        raise RuntimeError("No valid prompt items loaded from %s" % prompts_dir.resolve())

    print(
        "Found %s prompts. provider=nanobanana2 concurrency=%s rerun_existing=%s"
        % (total, args.concurrency, args.rerun_existing)
    )

    client_pool = ThreadLocalClientPool(model=args.model, api_key=args.api_key)

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
                args.model,
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
