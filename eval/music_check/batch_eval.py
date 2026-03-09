# batch_eval.py
import os
import re
import json
import argparse
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple

from concurrent.futures import ProcessPoolExecutor, as_completed

# ====== IMPORT YOUR JUDGE ======
from music_check_gemini import judge_video_music_raw_midi


# -------------------------
# Filename matching helpers
# -------------------------
def safe_filename(name: str, max_len: int = 180) -> str:
    name = str(name).strip()
    name = re.sub(r"[/\:*?\"<>|\n\r\t]", "_", name)
    name = re.sub(r"\s+", " ", name).strip()
    if not name:
        name = "untitled"
    if len(name) > max_len:
        name = name[:max_len].rstrip()
    return name


VIDEO_EXTS = [".mp4", ".mov", ".mkv", ".webm", ".avi", ".m4v"]


def find_video_by_content(video_dir: Path, content_value: str) -> Optional[Path]:
    """
    Match video file by base name == safe_filename(content).
    If multiple matches (different extension), pick the first by ext order.
    """
    base = safe_filename(content_value)

    # fast path: exact match by extension preference
    for ext in VIDEO_EXTS:
        p = video_dir / f"{base}{ext}"
        if p.exists():
            return p

    # fallback: case-insensitive scan
    if not video_dir.exists():
        return None

    target = base.lower()
    candidates = []
    for p in video_dir.iterdir():
        if p.is_file() and p.suffix.lower() in VIDEO_EXTS:
            if p.stem.lower() == target:
                candidates.append(p)

    if not candidates:
        return None

    def rank(p: Path) -> Tuple[int, str]:
        try:
            ext_rank = VIDEO_EXTS.index(p.suffix.lower())
        except ValueError:
            ext_rank = 999
        return (ext_rank, p.name.lower())

    candidates.sort(key=rank)
    return candidates[0]


# -------------------------
# Prompt loading
# -------------------------
def load_prompts(prompts_root: Path) -> List[Tuple[str, Dict[str, Any]]]:
    """
    Returns list of (category, item) where item has keys like content/prompt.
    """
    out: List[Tuple[str, Dict[str, Any]]] = []
    for jf in sorted(prompts_root.glob("*.json")):
        category = jf.stem
        data = json.loads(jf.read_text(encoding="utf-8"))
        if not isinstance(data, list):
            raise ValueError(f"{jf} must be a JSON list")
        for item in data:
            if isinstance(item, dict):
                out.append((category, item))
    return out


# -------------------------
# Worker
# -------------------------
def _eval_one(
    category: str,
    item: Dict[str, Any],
    videos_root: str,
    outputs_root: str,
    constraints_cache_dir: str,
    timeout_s: int,
    overwrite: bool,
) -> Dict[str, Any]:
    content = item.get("content", "")
    prompt_text = item.get("prompt", "")

    video_dir = Path(videos_root) / category
    video_path = find_video_by_content(video_dir, content)

    result: Dict[str, Any] = {
        "category": category,
        "content": content,
        "safe_content": safe_filename(content),
        "video_path": str(video_path) if video_path else None,
        "status": "ok",
        "error": None,

        # scoring fields
        "overall_score": None,
        "confidence": None,
        "skipped": None,
        "skip_reason": None,

        "out_path": None,
    }

    if video_path is None:
        result["status"] = "missing_video"
        result["error"] = f"Video not found under {video_dir} for content='{content}' (safe='{safe_filename(content)}')"
        return result

    out_dir = Path(outputs_root) / category
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{safe_filename(content)}.music_raw_midi_result.json"
    result["out_path"] = str(out_path)

    # If exists and not overwrite: load and return (keeps batch fast)
    if out_path.exists() and not overwrite:
        try:
            existing = json.loads(out_path.read_text(encoding="utf-8"))
            result["overall_score"] = existing.get("overall_score")
            result["confidence"] = existing.get("confidence")
            result["skipped"] = existing.get("skipped")
            result["skip_reason"] = existing.get("skip_reason")
            result["status"] = "skipped_exists"
            return result
        except Exception:
            # corrupted -> re-run
            pass

    verdict = judge_video_music_raw_midi(
        video_path=str(video_path),
        prompt_text=str(prompt_text),
        out_path=str(out_path),
        constraints_cache_dir=str(constraints_cache_dir),
        timeout_s=int(timeout_s),
    )

    result["overall_score"] = verdict.get("overall_score")
    result["confidence"] = verdict.get("confidence")
    result["skipped"] = verdict.get("skipped")
    result["skip_reason"] = verdict.get("skip_reason")
    return result


# -------------------------
# Main
# -------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--videos-root", type=str, default="ovi",
                    help="Root dir containing category subfolders with videos.")
    ap.add_argument("--prompts-root", type=str, default="prompts",
                    help="Dir containing category JSON prompt files.")
    ap.add_argument("--outputs-root", type=str, default="outputs_batch_music",
                    help="Where to write per-video outputs.")
    ap.add_argument("--constraints-cache-dir", type=str, default="cache/music_prompt_constraints",
                    help="Directory for per-prompt constraint cache files: <prompt_id>.json")

    ap.add_argument("--timeout", type=int, default=int(os.getenv("GEMINI_TIMEOUT_S", "180")))
    ap.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 4) // 2))
    ap.add_argument("--only-category", type=str, default=None,
                    help="If set, only run this category (e.g., musical_instrument_tutorial).")
    ap.add_argument("--overwrite", action="store_true", help="Re-run even if output exists.")
    ap.add_argument("--summary-out", type=str, default="outputs_batch_music/summary.json")
    args = ap.parse_args()

    prompts_root = Path(args.prompts_root)
    videos_root = Path(args.videos_root)
    outputs_root = Path(args.outputs_root)
    outputs_root.mkdir(parents=True, exist_ok=True)
    Path(args.constraints_cache_dir).mkdir(parents=True, exist_ok=True)

    all_items = load_prompts(prompts_root)
    if args.only_category:
        all_items = [x for x in all_items if x[0] == args.only_category]

    tasks = list(all_items)
    print(f"Loaded {len(tasks)} prompts. workers={args.workers}")

    results: List[Dict[str, Any]] = []

    with ProcessPoolExecutor(max_workers=args.workers) as ex:
        futs = [
            ex.submit(
                _eval_one,
                category=category,
                item=item,
                videos_root=str(videos_root),
                outputs_root=str(outputs_root),
                constraints_cache_dir=str(args.constraints_cache_dir),
                timeout_s=int(args.timeout),
                overwrite=bool(args.overwrite),
            )
            for (category, item) in tasks
        ]

        for fut in as_completed(futs):
            try:
                r = fut.result()
            except Exception as e:
                r = {"status": "exception", "error": repr(e)}
            results.append(r)

            # progress
            st = r.get("status")
            cat = r.get("category")
            name = r.get("safe_content")
            if st in ("ok", "skipped_exists"):
                print(f"[{st}] {cat} / {name} score={r.get('overall_score')} conf={r.get('confidence')} skipped={r.get('skipped')}")
            else:
                print(f"[{st}] {cat} / {name} err={r.get('error')}")

    # -------------------------
    # Aggregation (IMPORTANT):
    # Only aggregate samples where skipped == False (i.e., prompt had explicit pitch constraints and is midi-evaluable)
    # -------------------------
    ok_or_cached = [r for r in results if r.get("status") in ("ok", "skipped_exists")]

    midi_evaluable = [
        r for r in ok_or_cached
        if (r.get("skipped") is False) and isinstance(r.get("overall_score"), (int, float))
    ]

    def _conf(r: Dict[str, Any]) -> float:
        c = r.get("confidence")
        try:
            return max(0.0, min(1.0, float(c)))
        except Exception:
            return 0.0

    total_score = float(sum(float(r.get("overall_score", 0)) for r in midi_evaluable))
    mean_score = (total_score / len(midi_evaluable)) if midi_evaluable else 0.0

    total_w = sum(_conf(r) for r in midi_evaluable)
    weighted_mean = (
        sum(float(r.get("overall_score", 0)) * _conf(r) for r in midi_evaluable) / total_w
        if total_w > 0 else 0.0
    )

    buckets = {"0": 0, "1-20": 0, "21-40": 0, "41-60": 0, "61-80": 0, "81-100": 0}
    for r in midi_evaluable:
        s = float(r.get("overall_score", 0))
        if s <= 0: buckets["0"] += 1
        elif s <= 20: buckets["1-20"] += 1
        elif s <= 40: buckets["21-40"] += 1
        elif s <= 60: buckets["41-60"] += 1
        elif s <= 80: buckets["61-80"] += 1
        else: buckets["81-100"] += 1

    summary = {
        "num_total": len(results),
        "num_ok": sum(1 for r in results if r.get("status") == "ok"),
        "num_skipped_exists": sum(1 for r in results if r.get("status") == "skipped_exists"),
        "num_missing_video": sum(1 for r in results if r.get("status") == "missing_video"),
        "num_exception": sum(1 for r in results if r.get("status") == "exception"),

        # prompt suitability stats
        "num_prompt_skipped": sum(1 for r in ok_or_cached if r.get("skipped") is True),
        "num_midi_evaluable": len(midi_evaluable),

        # aggregated scores on midi-evaluable subset only
        "total_score": total_score,
        "mean_score": mean_score,
        "confidence_weighted_mean_score": weighted_mean,
        "score_buckets": buckets,

        "results": results,
    }

    summary_out = Path(args.summary_out)
    summary_out.parent.mkdir(parents=True, exist_ok=True)
    summary_out.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Saved summary to {summary_out}")


if __name__ == "__main__":
    main()
