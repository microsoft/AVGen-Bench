# batch_eval.py
# Batch Gemini judge for Plot Matching (T2AV quality: plot/story alignment)
#
# Input layout (per your structure):
#   prompts_dir/
#     ads.json
#     news.json
#     ...
#   root_videos/
#     ads/<safe_filename(content)>.mp4
#     news/<safe_filename(content)>.mp4
#     ...
#
# The `content` field in prompt JSON is the mp4 title (already sanitized the same way as safe_filename).
#
# Output:
# - one JSON file with:
#   - per-category mean plot_alignment_score
#   - overall mean plot_alignment_score
#   - per-video result records (status ok/skip/missing/failed)
# - optionally caches raw Gemini results per video to allow resume.
#
# Install:
#   pip install google-generativeai
#
# Env:
#   export GEMINI_API_KEY="..."
#   export GEMINI_MODEL="gemini-3-pro-preview"   # optional
#   export GEMINI_TIMEOUT_S="900"                # optional
#
# Run:
#   python batch_eval.py --prompts_dir prompts --root_videos sora2_generated --out_json plot_eval_results.json
#
# Notes:
# - This script uploads each video to Gemini File API. Expect cost/latency.
# - Concurrency is supported; keep workers small to avoid 429.
# - Resume: if cache exists for a video key, it will not re-judge unless --force.
from collections import defaultdict
import os
import re
import time
import json
import random
import argparse
import hashlib
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
from concurrent.futures import ThreadPoolExecutor, as_completed

import google.generativeai as genai


# ----------------------------
# safe_filename (given)
# ----------------------------
def safe_filename(name: str, max_len: int = 180) -> str:
    name = str(name).strip()
    name = re.sub(r"[/\:*?\"<>|\n\r\t]", "_", name)
    name = re.sub(r"\s+", " ", name).strip()
    if not name:
        name = "untitled"
    if len(name) > max_len:
        name = name[:max_len].rstrip()
    return name


# ----------------------------
# Gemini judge prompt (same as your single script)
# ----------------------------
PLOT_JUDGE_SYSTEM = """
You are a strict, evidence-based judge for a text-to-audio-video benchmark.

Goal:
Evaluate how well the provided MP4 matches the given PROMPT in terms of plot/story alignment,
including:
- story beats / event sequence (plot development)
- shot changes / continuity / pacing (cuts)
- objects and attributes (counts, colors, identities)
- actions and causal relations
- audio alignment (dialogue/sfx/music/ambience) and whether audio matches what is shown

Rules:
- Use ONLY observable evidence from the video and audio.
- If something is not clearly observable (low resolution, fast motion, occlusion, darkness), mark UNCERTAIN.
- Do not invent details not supported by evidence.
- If the prompt is underspecified, do not penalize missing details; focus on core requirements.
- Output STRICT JSON only. No markdown. No code fences. No extra text.

Scoring:
Return plot_alignment_score in [0,100].
Use subscores in [0,100]:
- narrative_alignment: plot beats and ordering
- shot_alignment: shot changes/cuts consistent with prompt (if specified) and continuity
- visual_attribute_alignment: object presence, counts, colors, key attributes
- audio_alignment: audio events that should occur vs actually heard; audio-visual consistency

Overall plot_alignment_score = round(0.35*narrative_alignment + 0.15*shot_alignment
                                    + 0.30*visual_attribute_alignment + 0.20*audio_alignment)

Deduction guidance:
- Severe mismatch of a core plot beat: -25
- Moderate mismatch: -12
- Minor mismatch/subtle: -5
- Missing core requirement: -10
- Uncertain: -0 (but reduce confidence)
""".strip()

PLOT_JUDGE_USER_TEMPLATE = """
PROMPT (what the video should depict):
{prompt_text}

Output STRICT JSON:
{{
  "plot_alignment_score": 0,
  "subscores": {{
    "narrative_alignment": 0,
    "shot_alignment": 0,
    "visual_attribute_alignment": 0,
    "audio_alignment": 0
  }},
  "confidence": 0.0,

  "prompt_constraints": {{
    "required_beats": [
      {{
        "id": "B1",
        "beat": "A specific plot event that must happen, in order if applicable",
        "strength": "strong|medium|weak"
      }}
    ],
    "required_visual_attributes": [
      {{
        "id": "V1",
        "object": "object/entity name",
        "attributes": {{
          "count": "integer or null",
          "color": "string or null",
          "other": ["..."]
        }},
        "strength": "strong|medium|weak"
      }}
    ],
    "required_audio_events": [
      {{
        "id": "A1",
        "audio_event": "e.g., metronome ticking / explosion / dialogue / calm ambience",
        "strength": "strong|medium|weak"
      }}
    ],
    "editing_requirements": [
      {{
        "id": "E1",
        "requirement": "e.g., continuous shot / quick cuts / time-lapse / slow zoom",
        "strength": "strong|medium|weak"
      }}
    ]
  }},

  "observed_event_log": [
    {{
      "where": "beginning|middle|end|event_k",
      "visual": "what is seen (concise, factual)",
      "audio": "what is heard (concise, factual; say off-screen if applicable)"
    }}
  ],

  "checks": [
    {{
      "constraint_id": "B1|V1|A1|E1",
      "category": "beat|visual_attribute|audio|editing",
      "status": "match|mismatch|missing|uncertain",
      "severity": 1,
      "evidence": "observable evidence; do NOT speculate"
    }}
  ],

  "violations": [
    {{
      "type": "beat_order_error|missing_beat|wrong_object|wrong_count|wrong_color|identity_drift|shot_continuity|audio_mismatch|offscreen_audio_mismatch|other",
      "severity": 1,
      "evidence": "observable evidence"
    }}
  ],

  "summary": "short assessment of alignment and key gaps"
}}

Constraints:
- Keep required_beats / required_visual_attributes / required_audio_events / editing_requirements each to 3-10 items max.
- checks must cover ALL extracted constraints (B*, V*, A*, E*).
- severity is 1-5 (5 most severe).
- confidence is 0-1.
""".strip()


# ----------------------------
# helpers: parsing & clamping
# ----------------------------
def _extract_json(text: str) -> str:
    text = (text or "").strip()
    if text.startswith("```"):
        text = re.sub(r"^```[a-zA-Z]*\n?", "", text)
        text = re.sub(r"\n?```$", "", text)
        text = text.strip()
    if text.startswith("{") and text.endswith("}"):
        return text
    m = re.search(r"\{.*\}", text, flags=re.DOTALL)
    if not m:
        raise ValueError(f"Could not find JSON in response:\n{text[:1500]}")
    return m.group(0)

def _clamp_int_0_100(x: Any) -> int:
    try:
        v = int(x)
    except Exception:
        v = 0
    return max(0, min(100, v))

def _clamp_float_0_1(x: Any) -> float:
    try:
        v = float(x)
    except Exception:
        v = 0.0
    return max(0.0, min(1.0, v))

def sha1(s: str) -> str:
    return hashlib.sha1(s.encode("utf-8")).hexdigest()


# ----------------------------
# Gemini call
# ----------------------------
def upload_video(video_path: str):
    video_file = genai.upload_file(path=video_path)
    while video_file.state.name == "PROCESSING":
        time.sleep(2)
        video_file = genai.get_file(video_file.name)
    if video_file.state.name == "FAILED":
        raise RuntimeError("Video processing failed on Gemini File API.")
    return video_file

def judge_plot_matching(video_path: str, prompt_text: str, model_name: str, timeout_s: int) -> Dict[str, Any]:
    video_file = upload_video(video_path)

    model = genai.GenerativeModel(
        model_name=model_name,
        system_instruction=PLOT_JUDGE_SYSTEM,
    )

    try:
        resp = model.generate_content(
            [video_file, PLOT_JUDGE_USER_TEMPLATE.format(prompt_text=prompt_text)],
            request_options={"timeout": timeout_s},
        )
        data = json.loads(_extract_json(resp.text))

        # Clamp scores/confidence
        data["plot_alignment_score"] = _clamp_int_0_100(data.get("plot_alignment_score", 0))
        subs = data.get("subscores", {}) or {}
        data["subscores"] = {
            "narrative_alignment": _clamp_int_0_100(subs.get("narrative_alignment", 0)),
            "shot_alignment": _clamp_int_0_100(subs.get("shot_alignment", 0)),
            "visual_attribute_alignment": _clamp_int_0_100(subs.get("visual_attribute_alignment", 0)),
            "audio_alignment": _clamp_int_0_100(subs.get("audio_alignment", 0)),
        }
        data["confidence"] = _clamp_float_0_1(data.get("confidence", 0.0))

        data["_input"] = {
            "video_path": video_path,
            "prompt_text": prompt_text,
            "model": model_name,
        }
        return data
    finally:
        try:
            genai.delete_file(video_file.name)
        except Exception:
            pass


# ----------------------------
# Batch logic
# ----------------------------
def load_prompt_items(path: str) -> List[Dict[str, Any]]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

def list_prompt_files(prompts_dir: str) -> List[str]:
    # use the original prompts/*.json
    files = sorted([str(p) for p in Path(prompts_dir).glob("*.json")])
    return files

def category_from_prompt_file(path: str) -> str:
    return Path(path).stem  # ads.json -> ads

def video_path_from_item(root_videos: str, category: str, content: str) -> str:
    title = safe_filename(content)
    return os.path.join(root_videos, category, f"{title}.mp4")

def load_cache(cache_path: str) -> Dict[str, Any]:
    if not cache_path or (not os.path.exists(cache_path)):
        return {}
    try:
        with open(cache_path, "r", encoding="utf-8") as f:
            return json.load(f) or {}
    except Exception:
        return {}

def save_cache(cache_path: str, cache: Dict[str, Any]):
    os.makedirs(os.path.dirname(cache_path) or ".", exist_ok=True)
    tmp = cache_path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(cache, f, ensure_ascii=False, indent=2)
    os.replace(tmp, cache_path)

def mean(xs: List[float]) -> Optional[float]:
    return float(sum(xs) / len(xs)) if xs else None

def worker_task(task: Dict[str, Any]) -> Tuple[str, Dict[str, Any]]:
    """
    Returns (cache_key, record)
    """
    cache_key = task["cache_key"]
    video_path = task["video_path"]
    prompt_text = task["prompt_text"]
    model_name = task["model_name"]
    timeout_s = task["timeout_s"]
    max_retries = task["max_retries"]

    last_err = None
    for attempt in range(max_retries):
        try:
            result = judge_plot_matching(video_path, prompt_text, model_name, timeout_s)
            record = {
                "status": "ok",
                "result": result,
            }
            return cache_key, record
        except Exception as e:
            last_err = str(e)
            # exponential backoff + jitter (helps with 429)
            sleep_s = (2 ** attempt) * 2.0 + random.uniform(0, 1.0)
            time.sleep(sleep_s)

    return cache_key, {"status": "failed", "error": last_err or "unknown error"}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--prompts_dir", type=str, required=True, help="Directory containing original prompt JSONs (ads.json, news.json, ...)")
    ap.add_argument("--root_videos", type=str, required=True, help="Root dir containing generated videos per category folder")
    ap.add_argument("--out_json", type=str, default="plot_eval_results.json", help="Write aggregated results here")
    ap.add_argument("--cache_json", type=str, default="plot_eval_cache.json", help="Cache raw per-video judge results for resume")
    ap.add_argument("--force", action="store_true", help="Ignore cache and re-judge everything")

    ap.add_argument("--model", type=str, default=os.getenv("GEMINI_MODEL", "gemini-3-flash-preview"))
    ap.add_argument("--timeout_s", type=int, default=int(os.getenv("GEMINI_TIMEOUT_S", "900")))
    ap.add_argument("--api_key", type=str, default=os.getenv("GEMINI_API_KEY", ""))

    ap.add_argument("--workers", type=int, default=32, help="Parallel Gemini requests ")
    ap.add_argument("--max_retries", type=int, default=4)

    # optional: limit number of items per category (debug)
    ap.add_argument("--limit", type=int, default=0)

    args = ap.parse_args()

    if not args.api_key:
        raise SystemExit("Missing GEMINI_API_KEY. Set env GEMINI_API_KEY or pass --api_key")

    genai.configure(api_key=args.api_key)

    cache = load_cache(args.cache_json) if (args.cache_json and not args.force) else {}

    prompt_files = list_prompt_files(args.prompts_dir)
    if not prompt_files:
        raise SystemExit(f"No prompt json files found in {args.prompts_dir}")

    # Build tasks
    tasks = []
    meta = {}  # cache_key -> metadata for aggregation/output

    for pf in prompt_files:
        category = category_from_prompt_file(pf)
        items = load_prompt_items(pf)
        if args.limit and args.limit > 0:
            items = items[:args.limit]

        for it in items:
            content = it.get("content", "")
            prompt_text = (it.get("prompt", "") or "").strip()
            if not prompt_text:
                continue

            video_path = video_path_from_item(args.root_videos, category, content)
            cache_key = sha1(f"{category}|{video_path}|{prompt_text}")

            meta[cache_key] = {
                "category": category,
                "content": content,
                "prompt": prompt_text,
                "video_path": video_path,
            }

            if (not args.force) and (cache_key in cache) and (cache[cache_key].get("status") == "ok"):
                continue

            if not os.path.exists(video_path):
                print(f"Missing video: '{video_path}'")
                # store missing in cache for completeness (no Gemini call)
                cache[cache_key] = {"status": "missing_video"}
                continue

            tasks.append({
                "cache_key": cache_key,
                "video_path": video_path,
                "prompt_text": prompt_text,
                "model_name": args.model,
                "timeout_s": args.timeout_s,
                "max_retries": args.max_retries,
            })

    # Run parallel judgments
    if tasks:
        print(f"Pending Gemini judgments: {len(tasks)}  workers={args.workers}")
        with ThreadPoolExecutor(max_workers=args.workers) as ex:
            futures = [ex.submit(worker_task, t) for t in tasks]
            for fut in as_completed(futures):
                k, rec = fut.result()
                cache[k] = rec
                # flush cache periodically
                if args.cache_json:
                    save_cache(args.cache_json, cache)
    else:
        print("No pending judgments (all cached or missing).")

    # Aggregate results
    per_category_scores = defaultdict(list)
    overall_scores = []
    results_out = []

    counts = {
        "ok": 0,
        "missing_video": 0,
        "failed": 0,
        "other": 0,
    }

    for cache_key, m in meta.items():
        rec = cache.get(cache_key, {"status": "other"})
        status = rec.get("status", "other")

        out_rec = {
            **m,
            "status": status,
        }

        if status == "ok":
            result = rec.get("result", {}) or {}
            score = _clamp_int_0_100(result.get("plot_alignment_score", 0))
            out_rec["plot_alignment_score"] = score
            out_rec["subscores"] = result.get("subscores", {})
            out_rec["confidence"] = result.get("confidence", 0.0)
            out_rec["summary"] = result.get("summary", "")
            # keep full judge output for debugging
            out_rec["judge"] = result

            per_category_scores[m["category"]].append(float(score))
            overall_scores.append(float(score))
            counts["ok"] += 1
        elif status == "missing_video":
            counts["missing_video"] += 1
        elif status == "failed":
            out_rec["error"] = rec.get("error", "")
            counts["failed"] += 1
        else:
            counts["other"] += 1

        results_out.append(out_rec)

    summaries = []
    for cat in sorted(per_category_scores.keys()):
        xs = per_category_scores[cat]
        summaries.append({
            "category": cat,
            "count_scored": int(len(xs)),
            "mean_plot_alignment_score": mean(xs),
        })

    overall = {
        "count_scored_total": int(len(overall_scores)),
        "mean_plot_alignment_score_total": mean(overall_scores),
    }

    out = {
        "model": args.model,
        "timeout_s": args.timeout_s,
        "workers": args.workers,
        "counts": counts,
        "summaries_by_category": summaries,
        "overall": overall,
        "results": results_out,
    }

    os.makedirs(os.path.dirname(args.out_json) or ".", exist_ok=True)
    with open(args.out_json, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)

    print("=== Category Means (plot matching) ===")
    for s in summaries:
        print(f"{s['category']:>20s}  scored={s['count_scored']:4d}  mean={s['mean_plot_alignment_score']}")
    print("=== Overall ===")
    print(f"scored_total={overall['count_scored_total']}  mean_total={overall['mean_plot_alignment_score_total']}")
    print(f"Wrote: {args.out_json}")
    if args.cache_json:
        print(f"Cache: {args.cache_json}")


if __name__ == "__main__":
    main()
