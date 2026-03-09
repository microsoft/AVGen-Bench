import os
import random
import time
from typing import Any, Dict, Optional, Tuple

import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

from .base import BaseGenerationClient, GenerationArtifact


def build_session() -> requests.Session:
    retry = Retry(
        total=8,
        connect=8,
        read=8,
        backoff_factor=0.8,
        status_forcelist=[429, 500, 502, 503, 504],
        allowed_methods=["GET", "POST"],
        raise_on_status=False,
    )
    session = requests.Session()
    session.mount("https://", HTTPAdapter(max_retries=retry))
    session.mount("http://", HTTPAdapter(max_retries=retry))
    return session


def get_json_with_retry(
    session: requests.Session,
    url: str,
    headers: Dict[str, str],
    timeout: Tuple[int, int] = (10, 60),
    max_ssl_retries: int = 6,
) -> Dict[str, Any]:
    last_err = None
    for i in range(max_ssl_retries):
        try:
            resp = session.get(url, headers=headers, timeout=timeout)
            resp.raise_for_status()
            return resp.json()
        except (
            requests.exceptions.SSLError,
            requests.exceptions.ConnectionError,
            requests.exceptions.ReadTimeout,
        ) as e:
            last_err = e
            time.sleep(min(10, 0.8 * (2**i)))
    raise RuntimeError(f"GET failed after retries: {url}\nLast error: {last_err}")


def post_json_with_retry(
    session: requests.Session,
    url: str,
    headers: Dict[str, str],
    body: Dict[str, Any],
    timeout: Tuple[int, int] = (10, 120),
    max_ssl_retries: int = 6,
) -> requests.Response:
    last_err = None
    for i in range(max_ssl_retries):
        try:
            return session.post(url, headers=headers, json=body, timeout=timeout)
        except (
            requests.exceptions.SSLError,
            requests.exceptions.ConnectionError,
            requests.exceptions.ReadTimeout,
        ) as e:
            last_err = e
            time.sleep(min(10, 0.8 * (2**i)))
    raise RuntimeError(f"POST failed after retries: {url}\nLast error: {last_err}")


def download_bytes_with_retry(
    session: requests.Session,
    url: str,
    headers: Dict[str, str],
    timeout: Tuple[int, int] = (10, 180),
    max_retries: int = 8,
) -> bytes:
    last_err = None
    for i in range(max_retries):
        try:
            resp = session.get(url, headers=headers, timeout=timeout, stream=True)
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
    raise RuntimeError(f"DOWNLOAD failed after retries: {url}\nLast error: {last_err}")


class Wan26Client(BaseGenerationClient):
    def __init__(self) -> None:
        self.session = build_session()
        self.base_url = os.getenv("DASHSCOPE_BASE_URL", "https://dashscope.aliyuncs.com").rstrip(
            "/"
        )
        self.api_key = os.getenv("DASHSCOPE_API_KEY")
        if not self.api_key:
            raise ValueError("DASHSCOPE_API_KEY environment variable is required.")

        self.create_url = f"{self.base_url}/api/v1/services/aigc/video-generation/video-synthesis"
        self.task_base = f"{self.base_url}/api/v1/tasks"

        self.headers_create = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
            "X-DashScope-Async": "enable",
        }
        self.headers_get = {"Authorization": f"Bearer {self.api_key}"}

    def _create_task(
        self,
        prompt: str,
        duration: int,
        size: str,
        prompt_extend: bool,
        shot_type: Optional[str],
        negative_prompt: Optional[str],
    ) -> str:
        body: Dict[str, Any] = {
            "model": "wan2.6-t2v",
            "input": {"prompt": prompt},
            "parameters": {
                "duration": duration,
                "size": size,
                "prompt_extend": prompt_extend,
            },
        }
        if shot_type:
            body["parameters"]["shot_type"] = shot_type
        if negative_prompt:
            body["input"]["negative_prompt"] = negative_prompt

        resp = post_json_with_retry(
            self.session, self.create_url, self.headers_create, body, timeout=(10, 180)
        )
        if not resp.ok:
            try:
                detail = resp.json()
            except Exception:
                detail = resp.text
            raise RuntimeError(f"Wan submit failed: {detail}")

        data = resp.json()
        task_id = (data.get("output") or {}).get("task_id") or data.get("task_id")
        if not task_id:
            raise RuntimeError(f"Wan submit missing task_id: {str(data)[:500]}")
        return task_id

    def _poll_until_done(self, task_id: str, poll_interval: float, timeout_s: int) -> Dict[str, Any]:
        url = f"{self.task_base}/{task_id}"
        start = time.time()
        while True:
            if time.time() - start > timeout_s:
                raise TimeoutError(f"Wan task {task_id} polling timed out after {timeout_s}s")
            job = get_json_with_retry(self.session, url, self.headers_get, timeout=(10, 60))
            output = job.get("output") or {}
            status = output.get("task_status") or job.get("task_status")
            if status in ("SUCCEEDED", "FAILED"):
                return job
            time.sleep(poll_interval)

    def _extract_video_url(self, final_job: Dict[str, Any]) -> str:
        output = final_job.get("output") or {}
        if isinstance(output.get("video_url"), str) and output["video_url"]:
            return output["video_url"]
        if isinstance(output.get("video_urls"), list) and output["video_urls"]:
            if isinstance(output["video_urls"][0], str):
                return output["video_urls"][0]
        if isinstance(output.get("results"), list) and output["results"]:
            first = output["results"][0]
            if isinstance(first, dict) and first.get("url"):
                return first["url"]
        raise RuntimeError(f"Wan cannot find video url in response: {str(final_job)[:800]}")

    def _download_video(self, video_url: str) -> bytes:
        return download_bytes_with_retry(
            self.session,
            video_url,
            headers={},
            timeout=(10, 300),
            max_retries=8,
        )

    def video_generation(
        self,
        prompt: str,
        duration: int = 10,
        size: str = "1280*720",
        prompt_extend: bool = True,
        shot_type: Optional[str] = "multi",
        negative_prompt: Optional[str] = None,
        poll_interval: float = 8.0,
        timeout_s: int = 1800,
        **kwargs,
    ) -> GenerationArtifact:
        del kwargs
        normalized_size = str(size).replace("x", "*")
        task_id = self._create_task(
            prompt=prompt,
            duration=int(duration),
            size=normalized_size,
            prompt_extend=bool(prompt_extend),
            shot_type=shot_type,
            negative_prompt=negative_prompt,
        )
        final_job = self._poll_until_done(task_id=task_id, poll_interval=poll_interval, timeout_s=timeout_s)
        output = final_job.get("output") or {}
        status = output.get("task_status") or final_job.get("task_status")
        if status != "SUCCEEDED":
            raise RuntimeError(f"Wan task failed: {str(final_job)[:800]}")

        video_url = self._extract_video_url(final_job)
        video_bytes = self._download_video(video_url)
        return GenerationArtifact(data=video_bytes, extension=".mp4")

