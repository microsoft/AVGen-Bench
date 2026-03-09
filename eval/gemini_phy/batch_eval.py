#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import re
import time
import json
import csv
import hashlib
import argparse
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, DefaultDict
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed

import google.generativeai as genai

try:
    from filelock import FileLock
except Exception:
    FileLock = None  # will fallback to no-lock mode (not recommended)


TEXT_TO_EXPECTATIONS_SYSTEM = """
You are a benchmark judge. Extract ONLY observable, testable expectations from a given generation prompt.
Focus on semantic/world-knowledge physical correctness: what a viewer should be able to see/hear if the prompt is followed.

Rules:
- Produce expectations that are directly observable in the video/audio (color change, glow, bubbles, delayed vs sudden change, etc.).
- Avoid non-observable chemistry/physics equations unless tied to an observable outcome.
- If the prompt is underspecified, keep expectations minimal and mark them as "weak".
- Output STRICT JSON only. No markdown. No code fences. No extra text.
""".strip()

EXPECTATIONS_USER_INSTRUCTIONS = """
Given the prompt, output STRICT JSON with:
{
  "prompt": "...",
  "expectations": [
    {
      "id": "E1",
      "type": "color_change|luminescence|gas|precipitation|motion|contact|lighting|audio|temporal|other",
      "strength": "strong|medium|weak",
      "expectation": "one observable constraint",
      "negative_examples": ["what would clearly violate it (observable)"],
      "notes": "optional"
    }
  ]
}

Constraints:
- 3 to 10 expectations max.
- Keep each expectation short and testable.
""".strip()

SEMANTIC_JUDGE_SYSTEM = """
You are a strict, evidence-based video judge for an AV generation benchmark.

Inputs:
- A text prompt (what the video should depict)
- A list of observable expectations derived from that prompt
- The video

Task:
1) Create a short event log (3-8 items) describing what is seen/heard (no speculation).
2) For each expectation, determine status: match|mismatch|missing|uncertain.
3) Score semantic_physics_score in [0,100] based ONLY on expectations alignment.
4) Output STRICT JSON only.

Rules:
- Use ONLY observable audio/visual evidence. Do NOT assume hidden causes.
- If not determinable due to quality/darkness/blur/cuts, choose UNCERTAIN and do not penalize.
- Do not penalize continuity breaks caused by obvious hard cuts unless the prompt explicitly requires a single continuous shot.
- Output STRICT JSON only. No markdown. No code fences. No extra text.

Deduct guidelines (apply per notable expectation; floor at 0):
- Severe mismatch: -30
- Moderate mismatch: -15
- Minor mismatch: -8
- Missing a strong expected phenomenon: -20 (medium: -12, weak: -6)
- Uncertain: -0
""".strip()

SEMANTIC_JUDGE_USER_TEMPLATE = """
Prompt:
{prompt_text}

Expectations JSON:
{expectations_json}

Output STRICT JSON with:
{{
  "semantic_physics_score": 0,
  "confidence": 0.0,
  "event_log": [
    {{"where": "beginning|middle|end|event_k", "what": "observable description"}}
  ],
  "checks": [
    {{
      "expectation_id": "E1",
      "status": "match|mismatch|missing|uncertain",
      "severity": 1,
      "evidence": "what is seen/heard that supports the status (no speculation)"
    }}
  ],
  "major_issues": ["top semantic issues affecting the score"]
}}

Constraints:
- checks must cover ALL expectations from the input.
- severity is 1-5 (5 = most severe).
- confidence is 0-1 (lower if many uncertain checks or low visibility).
""".strip()

VISUAL_SCAN_SYSTEM = """
You are a strict video physics auditor for an AV generation benchmark.

Input:
- The video only

Task:
Scan the video for PHYSICAL IMPLAUSIBILITIES that do NOT require the text prompt:
- Kinematics: teleporting, sudden velocity/acceleration jumps without cause, impossible trajectories
- Gravity: hovering, falling upward, wrong fall behavior
- Dynamics/Collisions: impact without reaction, missing recoil, implausible momentum transfer
- Contact/Support: interpenetration (objects pass through), floating without support, incorrect grasping
- Occlusion/Depth: front/back ordering flips, impossible occlusion boundaries
- Temporal identity: object shape/texture/identity changes within a continuous shot
- Fluids/Smoke/Fire: clearly impossible flow behavior
- Camera/Imaging physics (optional): shadows/reflections grossly inconsistent within a continuous shot

Rules:
- Use ONLY observable evidence. No speculation about hidden forces.
- Do NOT treat hard cuts as violations; mark them as "cut_detected" in notes if relevant.
- Output STRICT JSON only. No markdown. No code fences. No extra text.

Scoring:
Return visual_physics_score in [0,100] with deduction:
- Severe violation: -25
- Moderate: -12
- Minor: -5
- Uncertain: -0 (do not penalize)
""".strip()

VISUAL_SCAN_USER_TEMPLATE = """
Watch the video and output STRICT JSON with:
{
  "visual_physics_score": 0,
  "confidence": 0.0,
  "cut_detected": true,
  "audit_by_category": {
    "kinematics": [ { "severity": 1, "evidence": "...", "where": "beginning|middle|end|event_k" } ],
    "gravity":     [ ... ],
    "dynamics":    [ ... ],
    "contact":     [ ... ],
    "occlusion":   [ ... ],
    "identity":    [ ... ],
    "fluids_fire": [ ... ],
    "imaging":     [ ... ]
  },
  "visual_violations": [
    {
      "type": "kinematics|gravity|dynamics|contact|occlusion|identity_drift|fluid|imaging|other",
      "severity": 1,
      "evidence": "observable evidence",
      "where": "beginning|middle|end|event_k"
    }
  ],
  "major_issues": ["top visual issues affecting the score"]
}

Constraints:
- audit_by_category MUST include all listed keys (use empty arrays if none).
- visual_violations should be a deduplicated summary of audit_by_category.
- severity is 1-5 (5 = most severe).
- confidence is 0-1.
""".strip()

MODEL_NAME_DEFAULT = "gemini-3-flash-preview"
DEFAULT_TIMEOUT_S = int(os.getenv("GEMINI_TIMEOUT_S", "900"))


# ========================== Utilities ==========================
def safe_filename(name: str, max_len: int = 180) -> str:
    name = str(name).strip()
    name = re.sub(r"[\/\\\:\*\?\"\<\>\|\n\r\t]", "_", name)
    name = re.sub(r"\s+", " ", name).strip()
    if not name:
        name = "untitled"
    if len(name) > max_len:
        name = name[:max_len].rstrip()
    return name


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


def _stable_prompt_id(prompt_text: str) -> str:
    return hashlib.sha256(prompt_text.strip().encode("utf-8")).hexdigest()[:16]


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


def safe_mean(vals: List[float]) -> Optional[float]:
    return (sum(vals) / len(vals)) if vals else None


# ========================== Gemini core ==========================
def extract_expectations(prompt_text: str, model_name: str, timeout_s: int) -> Dict[str, Any]:
    model = genai.GenerativeModel(model_name=model_name, system_instruction=TEXT_TO_EXPECTATIONS_SYSTEM)
    resp = model.generate_content(
        EXPECTATIONS_USER_INSTRUCTIONS + "\n\nPROMPT:\n" + prompt_text,
        request_options={"timeout": timeout_s},
    )
    return json.loads(_extract_json(resp.text))


def judge_semantic_with_expectations(
    video_file: Any,
    prompt_text: str,
    expectations: Dict[str, Any],
    model_name: str,
    timeout_s: int,
) -> Dict[str, Any]:
    model = genai.GenerativeModel(model_name=model_name, system_instruction=SEMANTIC_JUDGE_SYSTEM)
    user_text = SEMANTIC_JUDGE_USER_TEMPLATE.format(
        prompt_text=prompt_text,
        expectations_json=json.dumps(expectations, ensure_ascii=False),
    )
    resp = model.generate_content([video_file, user_text], request_options={"timeout": timeout_s})
    data = json.loads(_extract_json(resp.text))
    data["semantic_physics_score"] = _clamp_int_0_100(data.get("semantic_physics_score", 0))
    data["confidence"] = _clamp_float_0_1(data.get("confidence", 0.0))
    return data


def scan_visual_physics(video_file: Any, model_name: str, timeout_s: int) -> Dict[str, Any]:
    model = genai.GenerativeModel(model_name=model_name, system_instruction=VISUAL_SCAN_SYSTEM)
    resp = model.generate_content([video_file, VISUAL_SCAN_USER_TEMPLATE], request_options={"timeout": timeout_s})
    data = json.loads(_extract_json(resp.text))
    data["visual_physics_score"] = _clamp_int_0_100(data.get("visual_physics_score", 0))
    data["confidence"] = _clamp_float_0_1(data.get("confidence", 0.0))
    data["cut_detected"] = bool(data.get("cut_detected", False))
    return data


def upload_video(video_path: str) -> Any:
    video_file = genai.upload_file(path=video_path)
    while video_file.state.name == "PROCESSING":
        time.sleep(2)
        video_file = genai.get_file(video_file.name)
    if video_file.state.name == "FAILED":
        raise RuntimeError("Video processing failed on Gemini File API.")
    return video_file


def _load_cache(cache_path: str) -> Dict[str, Any]:
    if not cache_path or not os.path.exists(cache_path):
        return {}
    with open(cache_path, "r", encoding="utf-8") as f:
        return json.load(f)


def _save_cache(cache_path: str, cache: Dict[str, Any]) -> None:
    os.makedirs(os.path.dirname(cache_path) or ".", exist_ok=True)
    with open(cache_path, "w", encoding="utf-8") as f:
        json.dump(cache, f, ensure_ascii=False, indent=2)


def get_expectations_with_cache(
    prompt_text: str,
    model_name: str,
    timeout_s: int,
    expectations_cache_path: Optional[str],
    lock_timeout_s: int = 120,
) -> Dict[str, Any]:
    """
    Multi-process safe Stage1 cache:
    - If filelock is available: lock -> read -> maybe write -> unlock
    - Otherwise: best-effort (may duplicate Stage1 under concurrency)
    """
    prompt_id = _stable_prompt_id(prompt_text)

    if not expectations_cache_path:
        return extract_expectations(prompt_text, model_name=model_name, timeout_s=timeout_s)

    lock_path = expectations_cache_path + ".lock"

    if FileLock is None:
        cache = _load_cache(expectations_cache_path)
        if prompt_id in cache:
            return cache[prompt_id]
        exp = extract_expectations(prompt_text, model_name=model_name, timeout_s=timeout_s)
        cache[prompt_id] = exp
        _save_cache(expectations_cache_path, cache)
        return exp

    lock = FileLock(lock_path)
    with lock.acquire(timeout=lock_timeout_s):
        cache = _load_cache(expectations_cache_path)
        if prompt_id in cache:
            return cache[prompt_id]

    # compute outside lock (avoid holding lock during API call)
    exp = extract_expectations(prompt_text, model_name=model_name, timeout_s=timeout_s)

    # write with lock
    with lock.acquire(timeout=lock_timeout_s):
        cache = _load_cache(expectations_cache_path)
        if prompt_id not in cache:
            cache[prompt_id] = exp
            _save_cache(expectations_cache_path, cache)
        return cache[prompt_id]


def run_two_stage_with_dual_stage2(
    video_path: str,
    prompt_text: str,
    model_name: str,
    expectations_cache_path: Optional[str],
    save_result_path: Optional[str],
    timeout_s: int,
    weights: Tuple[float, float],
) -> Dict[str, Any]:
    prompt_id = _stable_prompt_id(prompt_text)

    expectations = get_expectations_with_cache(
        prompt_text=prompt_text,
        model_name=model_name,
        timeout_s=timeout_s,
        expectations_cache_path=expectations_cache_path,
    )

    video_file = upload_video(video_path)
    try:
        semantic = judge_semantic_with_expectations(
            video_file=video_file,
            prompt_text=prompt_text,
            expectations=expectations,
            model_name=model_name,
            timeout_s=timeout_s,
        )
        visual = scan_visual_physics(
            video_file=video_file,
            model_name=model_name,
            timeout_s=timeout_s,
        )

        w_sem, w_vis = weights
        semantic_score = _clamp_int_0_100(semantic.get("semantic_physics_score", 0))
        visual_score = _clamp_int_0_100(visual.get("visual_physics_score", 0))
        overall = int(round(w_sem * semantic_score + w_vis * visual_score))
        overall = max(0, min(100, overall))

        result: Dict[str, Any] = {
            "overall_score": overall,
            "semantic_physics_score": semantic_score,
            "visual_physics_score": visual_score,
            "confidence": round((semantic.get("confidence", 0.0) + visual.get("confidence", 0.0)) / 2.0, 4),
            "expectations": expectations.get("expectations", expectations),
            "semantic": {
                "event_log": semantic.get("event_log", []),
                "checks": semantic.get("checks", []),
                "major_issues": semantic.get("major_issues", []),
                "confidence": semantic.get("confidence", 0.0),
            },
            "visual": {
                "cut_detected": visual.get("cut_detected", False),
                "audit_by_category": visual.get("audit_by_category", {}),
                "visual_violations": visual.get("visual_violations", []),
                "major_issues": visual.get("major_issues", []),
                "confidence": visual.get("confidence", 0.0),
            },
            "_input": {
                "prompt_id": prompt_id,
                "prompt_text": prompt_text,
                "video_path": video_path,
                "model": model_name,
                "weights": {"semantic": w_sem, "visual": w_vis},
            },
        }

        if save_result_path:
            os.makedirs(os.path.dirname(save_result_path) or ".", exist_ok=True)
            with open(save_result_path, "w", encoding="utf-8") as f:
                json.dump(result, f, ensure_ascii=False, indent=2)

        return result
    finally:
        try:
            genai.delete_file(video_file.name)
        except Exception:
            pass


# ========================== Prompt index ==========================
def load_prompt_index(prompts_dir: Path) -> Dict[str, Dict[str, str]]:
    index: Dict[str, Dict[str, str]] = {}
    for jf in sorted(prompts_dir.glob("*.json")):
        with open(jf, "r", encoding="utf-8") as f:
            arr = json.load(f)
        if not isinstance(arr, list):
            continue
        for item in arr:
            if not isinstance(item, dict):
                continue
            content = item.get("content", "")
            prompt = item.get("prompt", "")
            if not content or not prompt:
                continue
            key = safe_filename(content)
            if key not in index:
                index[key] = {"prompt": prompt, "content": content, "source_file": jf.name}
    return index


def collect_videos_by_subfolder(root_videos: Path, recursive: bool) -> List[Tuple[str, List[Path]]]:
    if recursive:
        mp4s = sorted(root_videos.rglob("*.mp4"))
        grp: DefaultDict[str, List[Path]] = defaultdict(list)
        for vp in mp4s:
            rel_parent = str(vp.parent.relative_to(root_videos))
            grp[rel_parent].append(vp)
        return sorted(grp.items(), key=lambda x: x[0])

    groups: List[Tuple[str, List[Path]]] = []
    for sf in sorted([p for p in root_videos.iterdir() if p.is_dir()]):
        vids = sorted(sf.rglob("*.mp4"))
        if vids:
            groups.append((str(sf.relative_to(root_videos)), vids))
    return groups


# ========================== Worker (must be top-level for multiprocessing) ==========================
def _worker_eval_one(task: Dict[str, Any]) -> Dict[str, Any]:
    """
    task keys:
      video_path, folder, stem, prompt_text, matched_content, source_prompt_file,
      model, timeout_s, weights, expectations_cache, save_json, api_key
    """
    api_key = task["api_key"]
    genai.configure(api_key=api_key)

    try:
        result = run_two_stage_with_dual_stage2(
            video_path=task["video_path"],
            prompt_text=task["prompt_text"],
            model_name=task["model"],
            expectations_cache_path=task["expectations_cache"],
            save_result_path=task["save_json"],
            timeout_s=task["timeout_s"],
            weights=tuple(task["weights"]),
        )
        return {
            "folder": task["folder"],
            "video_path": task["video_path"],
            "video_stem": task["stem"],
            "matched_content": task["matched_content"],
            "source_prompt_file": task["source_prompt_file"],
            "overall_score": int(result["overall_score"]),
            "semantic_physics_score": int(result["semantic_physics_score"]),
            "visual_physics_score": int(result["visual_physics_score"]),
            "confidence": float(result.get("confidence", 0.0)),
            "status": "ok",
            "error": "",
        }
    except Exception as e:
        return {
            "folder": task["folder"],
            "video_path": task["video_path"],
            "video_stem": task["stem"],
            "matched_content": task["matched_content"],
            "source_prompt_file": task["source_prompt_file"],
            "overall_score": "",
            "semantic_physics_score": "",
            "visual_physics_score": "",
            "confidence": "",
            "status": "failed",
            "error": repr(e),
        }


# ========================== Main ==========================
def main():
    parser = argparse.ArgumentParser(description="Batch eval Gemini physics (dual-stage) for mp4 under subfolders (parallel).")
    parser.add_argument("--root_videos", type=str, required=True)
    parser.add_argument("--prompts_dir", type=str, required=True)

    parser.add_argument("--model", type=str, default=MODEL_NAME_DEFAULT)
    parser.add_argument("--timeout_s", type=int, default=DEFAULT_TIMEOUT_S)
    parser.add_argument("--weights", type=float, nargs=2, default=(0.5, 0.5), metavar=("W_SEM", "W_VIS"))

    parser.add_argument("--expectations_cache", type=str, default="cache/expectations_cache.json")
    parser.add_argument("--out_dir", type=str, default="outputs_batch", help="Per-video JSON output directory; empty string disables saving")

    parser.add_argument("--save_csv", type=str, default="")
    parser.add_argument("--save_summary_csv", type=str, default="")
    parser.add_argument("--recursive_root", action="store_true")
    parser.add_argument("--fail_fast", action="store_true")

    parser.add_argument("--workers", type=int, default=1, help="Video-level parallel workers (recommended 2-8 depending on quota/network)")
    args = parser.parse_args()

    api_key = os.getenv("GEMINI_API_KEY", "").strip()
    if not api_key:
        raise RuntimeError("Missing GEMINI_API_KEY env var. Please: export GEMINI_API_KEY='...'")

    if args.workers > 1 and FileLock is None and args.expectations_cache:
        print("WARNING: filelock not installed; expectations cache is not multiprocess-safe and may duplicate Stage1 calls.")
        print("         pip install filelock  (recommended)")

    root_videos = Path(args.root_videos).expanduser().resolve()
    prompts_dir = Path(args.prompts_dir).expanduser().resolve()
    if not root_videos.exists():
        raise FileNotFoundError(f"root_videos not found: {root_videos}")
    if not prompts_dir.exists():
        raise FileNotFoundError(f"prompts_dir not found: {prompts_dir}")

    prompt_index = load_prompt_index(prompts_dir)
    if not prompt_index:
        raise RuntimeError(f"No valid prompts loaded from: {prompts_dir}")

    groups = collect_videos_by_subfolder(root_videos, recursive=args.recursive_root)
    if not groups:
        print(f"No mp4 found under: {root_videos}")
        return

    out_dir = Path(args.out_dir).expanduser().resolve() if args.out_dir else None
    if out_dir:
        out_dir.mkdir(parents=True, exist_ok=True)

    # build tasks
    tasks: List[Dict[str, Any]] = []
    per_video_rows: List[Dict[str, Any]] = []

    for folder_name, mp4s in groups:
        for vp in mp4s:
            stem = vp.stem
            if stem not in prompt_index:
                # record no-match immediately
                per_video_rows.append({
                    "folder": folder_name,
                    "video_path": str(vp),
                    "video_stem": stem,
                    "matched_content": "",
                    "source_prompt_file": "",
                    "overall_score": "",
                    "semantic_physics_score": "",
                    "visual_physics_score": "",
                    "confidence": "",
                    "status": "no_prompt_match",
                    "error": "no_prompt_match",
                })
                continue

            prompt_text = prompt_index[stem]["prompt"]
            matched_content = prompt_index[stem]["content"]
            source_prompt_file = prompt_index[stem]["source_file"]

            save_json = None
            if out_dir:
                rel = vp.relative_to(root_videos)
                save_json = str(out_dir / (str(rel).replace("/", "__") + ".json"))

            tasks.append({
                "api_key": api_key,
                "video_path": str(vp),
                "folder": folder_name,
                "stem": stem,
                "prompt_text": prompt_text,
                "matched_content": matched_content,
                "source_prompt_file": source_prompt_file,
                "model": args.model,
                "timeout_s": args.timeout_s,
                "weights": (float(args.weights[0]), float(args.weights[1])),
                "expectations_cache": args.expectations_cache,
                "save_json": save_json,
            })

    total = len(tasks)
    print(f"Total videos with prompt match: {total} | workers={args.workers}")

    # run parallel
    if args.workers <= 1:
        for i, t in enumerate(tasks, 1):
            row = _worker_eval_one(t)
            print(f"[{i}/{total}] {Path(row['video_path']).name} -> {row.get('overall_score')} status={row['status']}")
            if args.fail_fast and row["status"] != "ok":
                raise RuntimeError(f"Failed: {row['video_path']} | {row['error']}")
            per_video_rows.append(row)
    else:
        with ProcessPoolExecutor(max_workers=args.workers) as ex:
            futs = [ex.submit(_worker_eval_one, t) for t in tasks]
            done = 0
            for fut in as_completed(futs):
                done += 1
                row = fut.result()
                print(f"[{done}/{total}] {Path(row['video_path']).name} -> {row.get('overall_score')} status={row['status']}")
                if args.fail_fast and row["status"] != "ok":
                    raise RuntimeError(f"Failed: {row['video_path']} | {row['error']}")
                per_video_rows.append(row)

    # aggregate
    folder_scores: DefaultDict[str, List[int]] = defaultdict(list)
    all_scores: List[int] = []
    for r in per_video_rows:
        if r.get("status") == "ok":
            folder_scores[r["folder"]].append(int(r["overall_score"]))
            all_scores.append(int(r["overall_score"]))

    print("\n" + "=" * 80)
    for folder, scores in sorted(folder_scores.items(), key=lambda x: x[0]):
        m = safe_mean(scores)
        print(f"Folder mean: {folder} -> {m:.4f} (n={len(scores)})" if m is not None else f"Folder mean: {folder} -> N/A")

    overall_mean = safe_mean(all_scores)
    print("-" * 80)
    print(f"Overall mean: {overall_mean:.4f} (n={len(all_scores)})" if overall_mean is not None else "Overall mean: N/A")

    # save per-video CSV
    if args.save_csv:
        out_csv = Path(args.save_csv).expanduser().resolve()
        out_csv.parent.mkdir(parents=True, exist_ok=True)
        with out_csv.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(
                f,
                fieldnames=[
                    "folder", "video_path", "video_stem",
                    "matched_content", "source_prompt_file",
                    "overall_score", "semantic_physics_score", "visual_physics_score",
                    "confidence", "status", "error"
                ],
            )
            writer.writeheader()
            writer.writerows(per_video_rows)
        print(f"Saved per-video CSV: {out_csv}")

    # save summary CSV
    if args.save_summary_csv:
        out_sum = Path(args.save_summary_csv).expanduser().resolve()
        out_sum.parent.mkdir(parents=True, exist_ok=True)
        rows = []
        for folder, scores in sorted(folder_scores.items(), key=lambda x: x[0]):
            rows.append({"folder": folder, "n": len(scores), "mean_overall": safe_mean(scores) if scores else ""})
        rows.append({"folder": "__ALL__", "n": len(all_scores), "mean_overall": overall_mean if overall_mean is not None else ""})
        with out_sum.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=["folder", "n", "mean_overall"])
            writer.writeheader()
            writer.writerows(rows)
        print(f"Saved summary CSV: {out_sum}")


if __name__ == "__main__":
    main()
