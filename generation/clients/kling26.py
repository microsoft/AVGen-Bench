import os
import random
import time
from typing import Any, Dict, Optional, Tuple

import jwt
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


class Kling26Client(BaseGenerationClient):
    def __init__(self) -> None:
        self.session = build_session()
        self.base_url = os.getenv("KLING_BASE_URL", "https://api-beijing.klingai.com").rstrip("/")
        self.access_key = os.getenv("KLING_ACCESS_KEY")
        self.secret_key = os.getenv("KLING_SECRET_KEY")
        if not self.access_key or not self.secret_key:
            raise ValueError(
                "KLING_ACCESS_KEY and KLING_SECRET_KEY environment variables are required."
            )

        self.create_url = f"{self.base_url}/v1/videos/text2video"
        self.get_url_tmpl = f"{self.base_url}/v1/videos/text2video/{{id}}"

    def _make_api_token(self, ttl_s: int = 1800) -> str:
        headers = {"alg": "HS256", "typ": "JWT"}
        now = int(time.time())
        payload = {"iss": self.access_key, "exp": now + ttl_s, "nbf": now - 5}
        return jwt.encode(payload, self.secret_key, headers=headers)

    def _auth_headers(self) -> Dict[str, str]:
        return {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self._make_api_token()}",
        }

    def _create_task(
        self,
        prompt: str,
        model_name: str,
        duration: str,
        aspect_ratio: str,
        mode: str,
        sound: str,
    ) -> str:
        body = {
            "model_name": model_name,
            "prompt": prompt,
            "duration": duration,
            "aspect_ratio": aspect_ratio,
            "mode": mode,
            "sound": sound,
        }
        resp = self.session.post(
            self.create_url,
            headers=self._auth_headers(),
            json=body,
            timeout=(25, 60),
        )
        resp.raise_for_status()
        data = resp.json()
        if data.get("code") != 0:
            raise RuntimeError(f"Kling create failed: {data}")
        task_id = (data.get("data") or {}).get("task_id")
        if not task_id:
            raise RuntimeError(f"Kling create missing task_id: {str(data)[:500]}")
        return task_id

    def _poll(self, task_id: str, poll_interval: float, timeout_s: int) -> Dict[str, Any]:
        url = self.get_url_tmpl.format(id=task_id)
        start = time.time()
        last_err: Optional[str] = None

        while True:
            if time.time() - start > timeout_s:
                raise TimeoutError(f"Kling polling timeout task_id={task_id}, last_err={last_err}")

            try:
                resp = self.session.get(url, headers=self._auth_headers(), timeout=(25, 60))
                resp.raise_for_status()
                data = resp.json()
                if data.get("code") != 0:
                    raise RuntimeError(f"Kling query failed: {data}")
                st = ((data.get("data") or {}).get("task_status") or "").lower()
                if st in ("succeed", "failed"):
                    return data
                time.sleep(poll_interval)
            except (
                requests.exceptions.ReadTimeout,
                requests.exceptions.ConnectTimeout,
                requests.exceptions.SSLError,
                requests.exceptions.ConnectionError,
            ) as e:
                last_err = repr(e)
                time.sleep(min(30, poll_interval + random.uniform(0, 2)))

    def _extract_video_url(self, final_job: Dict[str, Any]) -> str:
        videos = (((final_job.get("data") or {}).get("task_result") or {}).get("videos") or [])
        if videos and videos[0].get("url"):
            return videos[0]["url"]
        raise RuntimeError(f"Kling missing video url: {str(final_job)[:800]}")

    def _download_video(self, video_url: str, max_retries: int = 8) -> bytes:
        last_err: Optional[Exception] = None
        for i in range(max_retries):
            try:
                resp = self.session.get(video_url, timeout=(25, 300))
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
        raise RuntimeError(f"Kling download failed after retries: {video_url}, last_err={last_err}")

    def video_generation(
        self,
        prompt: str,
        model_name: str = "kling-v2-6",
        duration: str = "10",
        aspect_ratio: str = "16:9",
        mode: str = "pro",
        sound: str = "on",
        poll_interval: float = 6.0,
        timeout_s: int = 1800,
        **kwargs,
    ) -> GenerationArtifact:
        del kwargs
        task_id = self._create_task(
            prompt=prompt,
            model_name=model_name,
            duration=str(duration),
            aspect_ratio=aspect_ratio,
            mode=mode,
            sound=sound,
        )
        final_job = self._poll(task_id=task_id, poll_interval=poll_interval, timeout_s=timeout_s)
        status = ((final_job.get("data") or {}).get("task_status") or "").lower()
        if status != "succeed":
            raise RuntimeError(f"Kling task failed: {str(final_job)[:800]}")

        video_url = self._extract_video_url(final_job)
        video_bytes = self._download_video(video_url)
        return GenerationArtifact(data=video_bytes, extension=".mp4")

