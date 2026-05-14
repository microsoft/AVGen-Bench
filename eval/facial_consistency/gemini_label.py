import os
import json
import time
import glob
import random
import sys
from pathlib import Path
from typing import Dict, Any, List, Tuple, Optional
from concurrent.futures import ThreadPoolExecutor, as_completed

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from gemini_client import generate_content_text, resolve_api_key

MODEL_NAME = os.getenv("GEMINI_MODEL", "gemini-3-flash-preview")
PROMPT_VARIANT = os.getenv("FACIAL_PROMPT_VARIANT", "original").strip().lower()
REQUEST_TIMEOUT_S = int(os.getenv("FACIAL_GEMINI_TIMEOUT_S", "300"))


def _get_gemini_api_key() -> str:
    return resolve_api_key()

FACESPEC_INSTRUCTION = r"""
You are an evaluation analyst for text-to-video prompts.
Infer the EXPECTED NUMBER OF PRIMARY ON-SCREEN HUMAN FACIAL IDENTITIES from the prompt text.

Definitions:
- "Primary characters" are the main narrative actors explicitly described with actions, dialogue, close-ups, or repeated focus.
- Count DISTINCT human facial identities for primary characters only.
- Ignore background bystanders/crowds unless a specific person is singled out as a main actor.
- If the prompt implies many people (crowds, groups, soldiers), set crowd_possible=true but do NOT count them as primary unless clearly highlighted.
- If faces could appear as graphics/posters/screens/logos/masks, set non_human_or_graphic_face_possible=true.
- If there are NO humans or faces, min_ids=max_ids=0.

Output STRICT JSON ONLY (no markdown, no code fences), matching this schema:

{
  "primary_faces": {
    "min_ids": integer,
    "max_ids": integer,
    "confidence": number,
    "rationale": string,
    "likely_on_screen": boolean,
    "crowd_possible": boolean,
    "non_human_or_graphic_face_possible": boolean
  },
  "primary_characters": [
    {
      "name": string,
      "type": "human" | "nonhuman" | "unknown",
      "count": integer,
      "face_visibility": "likely" | "possible" | "unlikely",
      "notes": string
    }
  ],
  "constraints": {
    "ignore_background_bystanders": true,
    "ignore_posters_screens_faces": true
  }
}

Rules:
- min_ids <= max_ids
- confidence in [0,1]
- Keep rationale concise.
""".strip()

FACESPEC_INSTRUCTION_V1 = r"""
You are an evaluation analyst for text-to-video prompts.

## Task
Infer the EXPECTED NUMBER OF PRIMARY ON-SCREEN HUMAN FACIAL IDENTITIES from the prompt text.

## Definitions
- "Primary characters" are the main narrative actors explicitly described with actions, dialogue, close-ups, or repeated focus.
- Count DISTINCT human facial identities for primary characters only.
- Ignore background bystanders/crowds unless a specific person is singled out as a main actor.
- If the prompt implies many people (crowds, groups, soldiers), set crowd_possible=true but do NOT count them as primary unless clearly highlighted.
- If faces could appear as graphics/posters/screens/logos/masks, set non_human_or_graphic_face_possible=true.
- If there are NO humans or faces, min_ids=max_ids=0.

## Output STRICT JSON ONLY
No markdown, no code fences.

Use this schema:
{
  "primary_faces": {
    "min_ids": integer,
    "max_ids": integer,
    "confidence": number,
    "rationale": string,
    "likely_on_screen": boolean,
    "crowd_possible": boolean,
    "non_human_or_graphic_face_possible": boolean
  },
  "primary_characters": [
    {
      "name": string,
      "type": "human" | "nonhuman" | "unknown",
      "count": integer,
      "face_visibility": "likely" | "possible" | "unlikely",
      "notes": string
    }
  ],
  "constraints": {
    "ignore_background_bystanders": true,
    "ignore_posters_screens_faces": true
  }
}

## Rules
- min_ids <= max_ids
- confidence in [0,1]
- Keep rationale concise.
""".strip()

FACESPEC_INSTRUCTION_V2 = r"""
You are an evaluation analyst for text-to-video prompts. Infer the EXPECTED NUMBER OF PRIMARY ON-SCREEN HUMAN FACIAL IDENTITIES from the prompt text.

Definitions: "Primary characters" are the main narrative actors explicitly described with actions, dialogue, close-ups, or repeated focus. Count DISTINCT human facial identities for primary characters only. Ignore background bystanders/crowds unless a specific person is singled out as a main actor. If the prompt implies many people such as crowds, groups, or soldiers, set crowd_possible=true but do NOT count them as primary unless clearly highlighted. If faces could appear as graphics, posters, screens, logos, or masks, set non_human_or_graphic_face_possible=true. If there are NO humans or faces, min_ids=max_ids=0.

Return STRICT JSON ONLY, with no markdown and no code fences, matching this schema:
{
  "primary_faces": {
    "min_ids": integer,
    "max_ids": integer,
    "confidence": number,
    "rationale": string,
    "likely_on_screen": boolean,
    "crowd_possible": boolean,
    "non_human_or_graphic_face_possible": boolean
  },
  "primary_characters": [
    {
      "name": string,
      "type": "human" | "nonhuman" | "unknown",
      "count": integer,
      "face_visibility": "likely" | "possible" | "unlikely",
      "notes": string
    }
  ],
  "constraints": {
    "ignore_background_bystanders": true,
    "ignore_posters_screens_faces": true
  }
}

Rules: min_ids <= max_ids. confidence in [0,1]. Keep rationale concise.
""".strip()

FACESPEC_PROMPTS = {
    "original": FACESPEC_INSTRUCTION,
    "v1": FACESPEC_INSTRUCTION_V1,
    "v2": FACESPEC_INSTRUCTION_V2,
}

if PROMPT_VARIANT not in FACESPEC_PROMPTS:
    raise ValueError(f"Unsupported FACIAL_PROMPT_VARIANT: {PROMPT_VARIANT}")


def _strip_code_fences(text: str) -> str:
    t = (text or "").strip()
    if t.startswith("```"):
        t = t.split("\n", 1)[-1]
        if t.endswith("```"):
            t = t[:-3]
    return t.strip()


def ensure_int_range(spec: Dict[str, Any]) -> Dict[str, Any]:
    pf = spec.get("primary_faces", {}) or {}
    pf["min_ids"] = int(pf.get("min_ids", 0))
    pf["max_ids"] = int(pf.get("max_ids", pf["min_ids"]))
    if pf["max_ids"] < pf["min_ids"]:
        pf["max_ids"] = pf["min_ids"]
    c = float(pf.get("confidence", 0.5))
    pf["confidence"] = max(0.0, min(1.0, c))
    spec["primary_faces"] = pf

    # enforce constraints
    spec["constraints"] = {
        "ignore_background_bystanders": True,
        "ignore_posters_screens_faces": True
    }
    if "primary_characters" not in spec or spec["primary_characters"] is None:
        spec["primary_characters"] = []
    return spec


def gemini_analyze_prompt(prompt_text: str, timeout_s: int = REQUEST_TIMEOUT_S) -> Dict[str, Any]:
    user = f"PROMPT:\n{prompt_text}\n"
    out = generate_content_text(
        model_name=MODEL_NAME,
        user_parts=[FACESPEC_PROMPTS[PROMPT_VARIANT], user],
        api_key=_get_gemini_api_key(),
        timeout_s=timeout_s,
    )
    out = _strip_code_fences(out)
    return ensure_int_range(json.loads(out))


def load_json(path: str) -> Any:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def save_json(path: str, obj: Any):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)


def build_done_map(existing: Optional[Any]) -> Dict[Tuple[str, str], Dict[str, Any]]:
    done = {}
    if isinstance(existing, list):
        for it in existing:
            key = (it.get("content", ""), it.get("prompt", ""))
            done[key] = it
    return done


def make_error_record(content: str, prompt: str, err: str) -> Dict[str, Any]:
    return {
        "content": content,
        "prompt": prompt,
        "error": err,
        "primary_faces": {
            "min_ids": 0,
            "max_ids": 0,
            "confidence": 0.0,
            "rationale": "Gemini call failed; placeholder",
            "likely_on_screen": False,
            "crowd_possible": False,
            "non_human_or_graphic_face_possible": False
        },
        "primary_characters": [],
        "constraints": {
            "ignore_background_bystanders": True,
            "ignore_posters_screens_faces": True
        }
    }


def analyze_one(
    idx: int,
    content: str,
    prompt: str,
    max_retries: int,
) -> Tuple[int, Dict[str, Any]]:
    """
    Worker task. Each call performs its own Gemini request.
    """
    last_err = None
    for attempt in range(max_retries):
        try:
            spec = gemini_analyze_prompt(prompt)
            merged = {"content": content, "prompt": prompt, **spec}
            return idx, merged
        except Exception as e:
            last_err = str(e)
            # exponential backoff + jitter (helps with 429)
            sleep_s = (2 ** attempt) * 1.0 + random.uniform(0, 0.5)
            time.sleep(sleep_s)
    return idx, make_error_record(content, prompt, last_err or "unknown error")


def batch_analyze_prompts_parallel(
    prompts_dir: str,
    out_dir: str,
    max_workers: int = 6,
    max_retries: int = 4,
):
    _get_gemini_api_key()

    files = sorted(glob.glob(os.path.join(prompts_dir, "*.json")))
    if not files:
        raise RuntimeError(f"No json files found in: {prompts_dir}")

    for fp in files:
        name = os.path.splitext(os.path.basename(fp))[0]
        out_path = os.path.join(out_dir, f"{name}.expected_faces.json")

        data: List[Dict[str, Any]] = load_json(fp)

        existing = None
        if os.path.exists(out_path):
            try:
                existing = load_json(out_path)
            except Exception:
                existing = None
        done_map = build_done_map(existing)

        # Prepare result slots in original order
        results: List[Optional[Dict[str, Any]]] = [None] * len(data)

        # Fill from existing (resume)
        pending = []
        for i, item in enumerate(data):
            content = item.get("content", "")
            prompt = item.get("prompt", "")
            key = (content, prompt)
            if key in done_map:
                results[i] = done_map[key]
            else:
                pending.append((i, content, prompt))

        print(f"[{name}] total={len(data)} done={len(data)-len(pending)} pending={len(pending)} workers={max_workers}")

        if pending:
            with ThreadPoolExecutor(max_workers=max_workers) as ex:
                futures = [
                    ex.submit(analyze_one, i, content, prompt, max_retries)
                    for (i, content, prompt) in pending
                ]
                for fut in as_completed(futures):
                    i, rec = fut.result()
                    results[i] = rec

        # Safety: replace any None
        for i in range(len(results)):
            if results[i] is None:
                item = data[i]
                results[i] = make_error_record(item.get("content",""), item.get("prompt",""), "missing result")

        save_json(out_path, results)
        print(f"[OK] {fp} -> {out_path}")


if __name__ == "__main__":
    # export GEMINI_API_KEY="..."
    # python batch_prompt_facespec_parallel.py
    batch_analyze_prompts_parallel(
        prompts_dir="/path/to/video_generation/prompts",
        out_dir="prompts_expected_faces",
        max_workers=int(os.getenv("GEMINI_WORKERS", "32")),
        max_retries=4,
    )
