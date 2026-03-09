import os
import random
import time
from typing import Optional

import requests

from .base import BaseGenerationClient, GenerationArtifact


def download_file(url: str, timeout: int = 120, max_retries: int = 8) -> bytes:
    last_err = None
    for i in range(max_retries):
        try:
            resp = requests.get(url, timeout=timeout)
            if resp.ok:
                return resp.content
            last_err = RuntimeError(f"HTTP {resp.status_code}: {resp.text[:300]}")
        except (
            requests.exceptions.SSLError,
            requests.exceptions.ConnectionError,
            requests.exceptions.ReadTimeout,
        ) as e:
            last_err = e
        time.sleep(min(20, 0.8 * (2**i)) + random.uniform(0, 0.5))
    raise RuntimeError(f"Seedance download failed: {url}, last_err={last_err}")


class SeedanceClient(BaseGenerationClient):
    def __init__(self) -> None:
        try:
            from volcenginesdkarkruntime import Ark
        except ImportError as e:
            raise ImportError(
                "Missing dependency 'volcenginesdkarkruntime'. Install it to use provider=seedance."
            ) from e

        api_key = os.environ.get("ARK_API_KEY")
        if not api_key:
            raise ValueError("ARK_API_KEY environment variable is required.")
        base_url = os.environ.get("ARK_BASE_URL", "https://ark.cn-beijing.volces.com/api/v3")
        self.client = Ark(base_url=base_url, api_key=api_key)

    def video_generation(
        self,
        prompt: str,
        model_name: str = "doubao-seedance-1-5-pro-251215",
        resolution: str = "720p",
        ratio: str = "16:9",
        duration: int = 10,
        watermark: bool = False,
        image_url: Optional[str] = None,
        poll_interval: float = 3.0,
        timeout_s: int = 1800,
        **kwargs,
    ) -> GenerationArtifact:
        del kwargs
        content = [{"type": "text", "text": prompt}]
        if image_url:
            content.append({"type": "image_url", "image_url": {"url": image_url}})

        create_result = self.client.content_generation.tasks.create(
            model=model_name,
            content=content,
            resolution=resolution,
            ratio=ratio,
            duration=int(duration),
            watermark=bool(watermark),
        )
        task_id = create_result.id

        deadline = time.time() + timeout_s
        while True:
            if time.time() > deadline:
                raise TimeoutError(f"Seedance timeout after {timeout_s}s task_id={task_id}")

            get_result = self.client.content_generation.tasks.get(task_id=task_id)
            status = getattr(get_result, "status", None)

            if status == "succeeded":
                content_obj = getattr(get_result, "content", None)
                video_url = getattr(content_obj, "video_url", None) if content_obj else None
                if not video_url:
                    raise RuntimeError(f"Seedance no video_url in result task_id={task_id}")
                return GenerationArtifact(data=download_file(video_url), extension=".mp4")

            if status == "failed":
                err = getattr(get_result, "error", None)
                raise RuntimeError(f"Seedance task failed task_id={task_id}, error={err}")

            time.sleep(poll_interval)

