import io
import os
import random
import time
from typing import Optional

from .base import BaseGenerationClient, GenerationArtifact


def _normalize_image_ext(image_ext: Optional[str]) -> str:
    if not image_ext:
        return ".png"
    ext = image_ext.strip().lower()
    if not ext:
        return ".png"
    if not ext.startswith("."):
        ext = "." + ext
    return ext


def _format_for_ext(ext: str) -> str:
    if ext in (".jpg", ".jpeg"):
        return "JPEG"
    if ext == ".webp":
        return "WEBP"
    return "PNG"


def _ext_from_mime(mime_type: Optional[str]) -> Optional[str]:
    if not mime_type:
        return None
    mt = mime_type.lower()
    if mt in ("image/png", "image/x-png"):
        return ".png"
    if mt in ("image/jpeg", "image/jpg"):
        return ".jpg"
    if mt == "image/webp":
        return ".webp"
    return None


class Nanobanana2Client(BaseGenerationClient):
    def __init__(self, model: str = "gemini-3.1-flash-image-preview", api_key: Optional[str] = None) -> None:
        try:
            from google import genai
        except Exception as e:
            raise ImportError("google-genai is required. Install it with: pip install google-genai") from e

        self._genai = genai
        self.model = model
        self.api_key = api_key or os.getenv("GOOGLE_API_KEY") or os.getenv("GEMINI_API_KEY")
        if not self.api_key:
            raise ValueError("GOOGLE_API_KEY (or GEMINI_API_KEY) environment variable not set.")
        self.client = genai.Client(api_key=self.api_key)

    def image_generation(
        self,
        prompt: str,
        model: Optional[str] = None,
        image_ext: str = ".png",
        max_retries: int = 3,
        retry_backoff_s: float = 2.0,
        **kwargs,
    ) -> GenerationArtifact:
        del kwargs
        model_name = model or self.model
        last_err: Optional[Exception] = None
        for attempt in range(1, max_retries + 2):
            try:
                response = self.client.models.generate_content(
                    model=model_name,
                    contents=[prompt],
                )
                break
            except Exception as e:
                last_err = e
                err_text = str(e).lower()
                transient = any(
                    token in err_text
                    for token in ("ssl", "eof", "timeout", "connection", "temporarily")
                )
                if (not transient) or attempt > max_retries:
                    raise
                sleep_s = min(20.0, retry_backoff_s * (2 ** (attempt - 1)))
                sleep_s += random.uniform(0, 0.5)
                time.sleep(sleep_s)
        else:
            raise RuntimeError(f"Nanobanana2 request failed after retries: {last_err}")

        parts = getattr(response, "parts", None) or []
        for part in parts:
            inline = getattr(part, "inline_data", None)
            if inline is None:
                continue
            data = getattr(inline, "data", None)
            mime_type = getattr(inline, "mime_type", None)
            ext = _normalize_image_ext(image_ext)
            src_ext = _ext_from_mime(mime_type)
            if data:
                if src_ext and src_ext == ext:
                    return GenerationArtifact(data=data, extension=ext)
                try:
                    from PIL import Image as PILImage
                except Exception as e:
                    if src_ext:
                        return GenerationArtifact(data=data, extension=src_ext)
                    raise ImportError(
                        "Pillow is required to convert image formats. Install it with: pip install pillow"
                    ) from e
                image = PILImage.open(io.BytesIO(data))
                buf = io.BytesIO()
                image.save(buf, format=_format_for_ext(ext))
                return GenerationArtifact(data=buf.getvalue(), extension=ext)

            try:
                image = part.as_image()
            except Exception:
                continue
            buf = io.BytesIO()
            try:
                image.save(buf, format=_format_for_ext(ext))
            except TypeError:
                image.save(buf)
            return GenerationArtifact(data=buf.getvalue(), extension=ext)

        text_parts = []
        for part in parts:
            text = getattr(part, "text", None)
            if text:
                text_parts.append(text)
        detail = " | ".join(text_parts) if text_parts else "no image parts returned"
        raise RuntimeError(f"Nanobanana2 image generation returned no image. detail={detail}")

    def video_generation(self, prompt: str, **kwargs) -> GenerationArtifact:
        del prompt, kwargs
        raise NotImplementedError("Nanobanana2Client only supports image_generation().")
