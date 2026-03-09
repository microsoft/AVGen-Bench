import os
import random
import time
from typing import Any, Dict, Tuple

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
    raise RuntimeError("GET failed after retries: %s\nLast error: %s" % (url, last_err))


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
    raise RuntimeError("POST failed after retries: %s\nLast error: %s" % (url, last_err))


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
            resp = session.get(url, headers=headers, timeout=timeout)
            if resp.ok:
                return resp.content
            last_err = RuntimeError("HTTP %s: %s" % (resp.status_code, resp.text[:300]))
        except (
            requests.exceptions.SSLError,
            requests.exceptions.ConnectionError,
            requests.exceptions.ReadTimeout,
        ) as e:
            last_err = e
        time.sleep(min(20, 0.8 * (2**i)) + random.uniform(0, 0.5))
    raise RuntimeError("DOWNLOAD failed after retries: %s\nLast error: %s" % (url, last_err))


class Sora2Client(BaseGenerationClient):
    def __init__(self) -> None:
        self.session = build_session()

        self.endpoint = os.getenv(
            "ENDPOINT_URL", "https://sc-qo-mgt1o1ix-eastus2.openai.azure.com/"
        ).rstrip("/")
        self.deployment = os.getenv("DEPLOYMENT_NAME", "sora-2")
        self.subscription_key = os.getenv("AZURE_OPENAI_API_KEY")
        if not self.subscription_key:
            raise ValueError("AZURE_OPENAI_API_KEY environment variable not set.")

        self.api_version = "preview"
        self.params = "?api-version=%s" % self.api_version
        self.base_path = "openai/v1/videos"
        self.videos_url = "%s/%s%s" % (self.endpoint, self.base_path, self.params)

        self.headers = {
            "api-key": self.subscription_key,
            "Content-Type": "application/json",
        }

    def _submit_video_job(self, prompt: str, seconds: int = 8, size: str = "1280x720") -> str:
        body = {
            "prompt": prompt,
            "seconds": str(seconds),
            "size": size,
            "model": self.deployment,
        }
        job_resp = post_json_with_retry(self.session, self.videos_url, self.headers, body)
        if not job_resp.ok:
            try:
                detail = job_resp.json()
            except Exception:
                detail = job_resp.text
            raise RuntimeError("Video generation submit failed: %s" % detail)
        job_id = job_resp.json().get("id")
        if not job_id:
            raise RuntimeError("Submit ok but missing job id: %s" % job_resp.text[:300])
        return job_id

    def _poll_until_done(
        self, job_id: str, poll_interval: float = 5.0, timeout_s: int = 900
    ) -> Dict[str, Any]:
        status_url = "%s/openai/v1/videos/%s%s" % (self.endpoint, job_id, self.params)
        start = time.time()
        while True:
            if time.time() - start > timeout_s:
                raise TimeoutError("Job %s polling timed out after %ss" % (job_id, timeout_s))
            job = get_json_with_retry(self.session, status_url, self.headers, timeout=(10, 60))
            status = job.get("status")
            if status in ("completed", "failed"):
                return job
            time.sleep(poll_interval)

    def _download_video(self, generation_id: str) -> bytes:
        video_url = "%s/openai/v1/videos/%s/content%s" % (
            self.endpoint,
            generation_id,
            self.params,
        )
        return download_bytes_with_retry(
            self.session,
            video_url,
            self.headers,
            timeout=(10, 180),
            max_retries=8,
        )

    def video_generation(
        self,
        prompt: str,
        seconds: int = 12,
        size: str = "1280x720",
        poll_interval: float = 10.0,
        timeout_s: int = 900,
        **kwargs
    ) -> GenerationArtifact:
        del kwargs
        job_id = self._submit_video_job(prompt=prompt, seconds=seconds, size=size)
        final_job = self._poll_until_done(
            job_id=job_id, poll_interval=poll_interval, timeout_s=timeout_s
        )
        if final_job.get("status") != "completed":
            raise RuntimeError("job failed: %s" % str(final_job)[:500])

        generation_id = final_job.get("id")
        if not generation_id:
            raise RuntimeError("completed but missing generation id: %s" % final_job)

        video_bytes = self._download_video(generation_id)
        return GenerationArtifact(data=video_bytes, extension=".mp4")

