#!/usr/bin/env python3

import base64
import json
import mimetypes
import os
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Union


DEFAULT_API_BASE = "https://generativelanguage.googleapis.com"
API_KEY_ENV_VARS = (
    "GEMINI_API_KEY",
    "GOOGLE_API_KEY",
)


class GeminiAPIError(RuntimeError):
    pass


PartLike = Union[str, Dict[str, Any]]


def resolve_api_key(api_key: Optional[str] = None, required: bool = True) -> str:
    if api_key and api_key.strip():
        return api_key.strip()

    for env_name in API_KEY_ENV_VARS:
        value = os.getenv(env_name, "").strip()
        if value:
            return value

    if required:
        raise RuntimeError(
            "Missing Gemini API key. Set one of: " + ", ".join(API_KEY_ENV_VARS)
        )
    return ""


def guess_mime_type(file_path: Union[str, Path]) -> str:
    mime_type, _ = mimetypes.guess_type(str(file_path))
    return mime_type or "application/octet-stream"


def inline_file_part(file_path: Union[str, Path], mime_type: Optional[str] = None) -> Dict[str, Any]:
    path = Path(file_path)
    with path.open("rb") as f:
        data = base64.b64encode(f.read()).decode("utf-8")
    return {
        "inline_data": {
            "mime_type": mime_type or guess_mime_type(path),
            "data": data,
        }
    }


def _normalize_part_dict(part: Dict[str, Any]) -> Dict[str, Any]:
    if "inline_data" in part and "inlineData" not in part:
        inline_data = dict(part["inline_data"])
        if "mimeType" in inline_data and "mime_type" not in inline_data:
            inline_data["mime_type"] = inline_data.pop("mimeType")
        return {"inline_data": inline_data}

    if "inlineData" in part:
        inline_data = dict(part["inlineData"])
        if "mimeType" in inline_data and "mime_type" not in inline_data:
            inline_data["mime_type"] = inline_data.pop("mimeType")
        return {"inline_data": inline_data}

    if "file_data" in part and "fileData" not in part:
        file_data = dict(part["file_data"])
        if "mimeType" in file_data and "mime_type" not in file_data:
            file_data["mime_type"] = file_data.pop("mimeType")
        if "fileUri" in file_data and "file_uri" not in file_data:
            file_data["file_uri"] = file_data.pop("fileUri")
        return {"file_data": file_data}

    if "fileData" in part:
        file_data = dict(part["fileData"])
        if "mimeType" in file_data and "mime_type" not in file_data:
            file_data["mime_type"] = file_data.pop("mimeType")
        if "fileUri" in file_data and "file_uri" not in file_data:
            file_data["file_uri"] = file_data.pop("fileUri")
        return {"file_data": file_data}

    return part


def _normalize_parts(parts: Iterable[PartLike]) -> List[Dict[str, Any]]:
    normalized: List[Dict[str, Any]] = []
    for part in parts:
        if isinstance(part, str):
            normalized.append({"text": part})
        elif isinstance(part, dict):
            normalized.append(_normalize_part_dict(part))
        else:
            raise TypeError(f"Unsupported Gemini part type: {type(part)!r}")
    return normalized


def _extract_response_text(response_json: Dict[str, Any]) -> str:
    texts: List[str] = []
    for candidate in response_json.get("candidates", []):
        content = candidate.get("content", {}) or {}
        for part in content.get("parts", []) or []:
            text = part.get("text")
            if text:
                texts.append(text)
    return "\n".join(texts).strip()


def _model_resource(model_name: str) -> str:
    model_name = model_name.strip()
    if model_name.startswith("models/"):
        return model_name
    return f"models/{model_name}"


def generate_content_response(
    model_name: str,
    user_parts: Iterable[PartLike],
    *,
    api_key: Optional[str] = None,
    timeout_s: int = 300,
    system_instruction: Optional[str] = None,
    generation_config: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    api_key = resolve_api_key(api_key=api_key, required=True)
    model_resource = urllib.parse.quote(_model_resource(model_name), safe="/")
    endpoint = (
        f"{DEFAULT_API_BASE}/v1beta/{model_resource}:generateContent"
    )

    payload: Dict[str, Any] = {
        "contents": [
            {
                "role": "user",
                "parts": _normalize_parts(user_parts),
            }
        ]
    }
    if system_instruction:
        payload["system_instruction"] = {
            "parts": [{"text": system_instruction}],
        }
    if generation_config:
        payload["generationConfig"] = generation_config

    request = urllib.request.Request(
        endpoint,
        data=json.dumps(payload).encode("utf-8"),
        headers={
            "Content-Type": "application/json",
            "Accept": "application/json",
            "x-goog-api-key": api_key,
        },
        method="POST",
    )

    try:
        with urllib.request.urlopen(request, timeout=timeout_s) as response:
            raw = response.read().decode("utf-8")
    except urllib.error.HTTPError as exc:
        body = exc.read().decode("utf-8", errors="replace")
        raise GeminiAPIError(f"HTTP {exc.code}: {body}") from exc
    except urllib.error.URLError as exc:
        raise GeminiAPIError(f"Request failed: {exc}") from exc

    return json.loads(raw)


def generate_content_text(
    model_name: str,
    user_parts: Iterable[PartLike],
    *,
    api_key: Optional[str] = None,
    timeout_s: int = 300,
    system_instruction: Optional[str] = None,
    generation_config: Optional[Dict[str, Any]] = None,
) -> str:
    response_json = generate_content_response(
        model_name=model_name,
        user_parts=user_parts,
        api_key=api_key,
        timeout_s=timeout_s,
        system_instruction=system_instruction,
        generation_config=generation_config,
    )
    text = _extract_response_text(response_json)
    if not text:
        raise GeminiAPIError(
            f"No text found in response: {json.dumps(response_json, ensure_ascii=False)}"
        )
    return text
