#!/usr/bin/env python3
import argparse
import importlib
import json
import os
import random
import re
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


def safe_filename(name: str, max_len: int = 180) -> str:
    name = str(name).strip()
    name = re.sub(r"[\/\\\:\*\?\"\<\>\|\n\r\t]", "_", name)
    name = re.sub(r"\s+", " ", name).strip()
    if not name:
        name = "untitled"
    if len(name) > max_len:
        name = name[:max_len].rstrip()
    return name


def read_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def write_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)


def write_jsonl(path: Path, rows: Iterable[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def write_csv(path: Path, rows: List[Dict[str, Any]], fieldnames: List[str]) -> None:
    import csv

    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({k: row.get(k, "") for k in fieldnames})


def mean(xs: List[float]) -> Optional[float]:
    return (sum(xs) / len(xs)) if xs else None


def import_module_with_variant(module_name: str, variant: str, env_name: str):
    os.environ[env_name] = variant
    module = importlib.import_module(module_name)
    module = importlib.reload(module)
    if hasattr(module, "PROMPT_VARIANT"):
        module.PROMPT_VARIANT = variant
    return module


def find_repeats(session_model_dir: Path) -> List[Path]:
    return sorted([p for p in session_model_dir.iterdir() if p.is_dir() and p.name.startswith("repeat_")])


def load_master_manifest(repeat_dir: Path) -> List[Dict[str, Any]]:
    plot_path = repeat_dir / "eval_output" / "plot_matching" / repeat_dir.name / "eval_results.json"
    if plot_path.exists():
        data = read_json(plot_path)
        results = data.get("results", [])
        if results:
            return results

    speech_path = repeat_dir / "eval_output" / "speech" / repeat_dir.name / "transcripts.json"
    if speech_path.exists():
        return read_json(speech_path)

    raise FileNotFoundError(f"Cannot find manifest in {repeat_dir}")


def infer_videos_root(manifest: List[Dict[str, Any]]) -> Path:
    for item in manifest:
        video_path = item.get("video_path")
        category = item.get("category")
        if video_path and category:
            vp = Path(video_path)
            try:
                return vp.parents[1]
            except Exception:
                pass
    raise RuntimeError("Unable to infer videos_root from manifest")


def build_prompt_index(prompts_dir: Path) -> Dict[Tuple[str, str], Dict[str, Any]]:
    index: Dict[Tuple[str, str], Dict[str, Any]] = {}
    for jf in sorted(prompts_dir.glob("*.json")):
        arr = read_json(jf)
        if not isinstance(arr, list):
            continue
        for i, item in enumerate(arr):
            if not isinstance(item, dict):
                continue
            category = jf.stem
            content = item.get("content", "")
            prompt = item.get("prompt", "")
            if content and prompt:
                index[(category, content)] = {
                    "category": category,
                    "content": content,
                    "prompt": prompt,
                    "prompt_file": str(jf),
                    "index_in_file": i,
                }
    return index


def build_prompt_index_by_stem(prompts_dir: Path) -> Dict[str, Dict[str, Any]]:
    index: Dict[str, Dict[str, Any]] = {}
    for jf in sorted(prompts_dir.glob("*.json")):
        arr = read_json(jf)
        if not isinstance(arr, list):
            continue
        category = jf.stem
        for i, item in enumerate(arr):
            if not isinstance(item, dict):
                continue
            content = item.get("content", "")
            prompt = item.get("prompt", "")
            if not content or not prompt:
                continue
            stem = safe_filename(content)
            if stem not in index:
                index[stem] = {
                    "category": category,
                    "content": content,
                    "prompt": prompt,
                    "prompt_file": str(jf),
                    "index_in_file": i,
                }
    return index


def build_summary_by_category(rows: List[Dict[str, Any]], score_key: str, pass_key: Optional[str] = None) -> Dict[str, Any]:
    out: Dict[str, Any] = {}
    grouped: Dict[str, List[Dict[str, Any]]] = {}
    for row in rows:
        grouped.setdefault(str(row.get("category", "unknown")), []).append(row)
    for category, group in sorted(grouped.items()):
        scores = [float(r[score_key]) for r in group if r.get(score_key) is not None]
        rec: Dict[str, Any] = {"n": len(scores), "avg_score": mean(scores)}
        if pass_key is not None:
            passes = [1.0 if r.get(pass_key) else 0.0 for r in group if pass_key in r]
            rec["pass_rate"] = mean(passes)
        out[category] = rec
    return out


def _ocr_worker(module, payload_path: Path, prompt_meta: Dict[str, Any], videos_root: Path, out_json_dir: Path, variant: str) -> Dict[str, Any]:
    stem = payload_path.name.replace(".payload.json", "")
    payload = read_json(payload_path)
    category = prompt_meta["category"]
    content = prompt_meta["content"]
    result = module.gemini_judge(prompt_meta["prompt"], payload)
    result["_input"] = {
        "prompt_text": prompt_meta["prompt"],
        "video_path": str(videos_root / category / f"{safe_filename(content)}.mp4"),
        "variant": variant,
    }
    write_json(out_json_dir / f"{stem}.json", result)
    subs = result.get("subscores", {}) or {}
    return {
        "category": category,
        "content": content,
        "video_path": str(videos_root / category / f"{safe_filename(content)}.mp4"),
        "prompt_requires_visible_text": result.get("prompt_requires_visible_text"),
        "text_presence": result.get("text_presence"),
        "missing_required_text": result.get("missing_required_text"),
        "incidental_text_is_contextual": result.get("incidental_text_is_contextual"),
        "overall_text_quality_score": result.get("overall_text_quality_score"),
        "legibility_accuracy": subs.get("legibility_accuracy"),
        "temporal_stability": subs.get("temporal_stability"),
        "spatial_stability": subs.get("spatial_stability"),
        "completeness": subs.get("completeness"),
        "prompt_text_match": subs.get("prompt_text_match"),
        "confidence": result.get("confidence"),
        "summary": result.get("summary"),
    }


def run_ocr_variant(
    repeat_dir: Path,
    out_repeat_dir: Path,
    variant: str,
    prompts_index: Dict[Tuple[str, str], Dict[str, Any]],
    videos_root: Path,
    workers: int,
) -> None:
    module = import_module_with_variant("eval.Ocr.ocr_gemini", variant, "OCR_PROMPT_VARIANT")
    payload_dir = repeat_dir / "eval_output" / "ocr" / repeat_dir.name / "payload_json"
    out_dir = out_repeat_dir / "ocr" / repeat_dir.name
    out_json_dir = out_dir / "per_video_json"
    rows: List[Dict[str, Any]] = []
    prompts_by_stem = build_prompt_index_by_stem(Path(next(iter(prompts_index.values()))["prompt_file"]).parent) if prompts_index else {}
    tasks: List[Tuple[Path, Dict[str, Any]]] = []
    for payload_path in sorted(payload_dir.glob("*.payload.json")):
        stem = payload_path.name.replace(".payload.json", "")
        prompt_meta = prompts_by_stem.get(stem)
        if not prompt_meta:
            continue
        tasks.append((payload_path, prompt_meta))

    with ThreadPoolExecutor(max_workers=workers) as ex:
        futures = [
            ex.submit(_ocr_worker, module, payload_path, prompt_meta, videos_root, out_json_dir, variant)
            for payload_path, prompt_meta in tasks
        ]
        for fut in as_completed(futures):
            rows.append(fut.result())
    rows.sort(key=lambda x: (x["category"], x["content"]))

    write_csv(
        out_dir / "results_text_quality.csv",
        rows,
        [
            "category",
            "content",
            "video_path",
            "prompt_requires_visible_text",
            "text_presence",
            "missing_required_text",
            "incidental_text_is_contextual",
            "overall_text_quality_score",
            "legibility_accuracy",
            "temporal_stability",
            "spatial_stability",
            "completeness",
            "prompt_text_match",
            "confidence",
            "summary",
        ],
    )
    eligible = [
        r
        for r in rows
        if r["prompt_requires_visible_text"] is True or r["text_presence"] == "incidental"
    ]
    summary = {
        "variant": variant,
        "videos_root": str(videos_root),
        "count_scored": len(rows),
        "count_eligible_for_summary": len(eligible),
        "avg_score": mean([float(r["overall_text_quality_score"]) for r in eligible if r.get("overall_text_quality_score") is not None]),
        "text_presence_counts": {
            key: sum(1 for r in rows if r["text_presence"] == key)
            for key in sorted({str(r["text_presence"]) for r in rows})
        },
        "by_category": build_summary_by_category(eligible, "overall_text_quality_score"),
        "note": "Reused OCR payload_json from original run; only Gemini judgement was rerun.",
    }
    write_json(out_dir / "summary.json", summary)


def _speech_worker(module, item: Dict[str, Any]) -> Dict[str, Any]:
    judgement = module.evaluate_speech_with_gemini(
        generation_prompt=item["prompt"],
        transcript_text=item.get("transcript", ""),
    )
    return {
        "video_path": item["video_path"],
        "category": item["category"],
        "content": item["content"],
        "prompt": item["prompt"],
        "transcript": item.get("transcript", ""),
        "whisper_meta": item.get("whisper_meta", {}),
        "judgement": judgement,
    }


def run_speech_variant(repeat_dir: Path, out_repeat_dir: Path, variant: str, videos_root: Path, prompts_dir: Path, workers: int) -> None:
    module = import_module_with_variant("eval.speech.gemini_speech", variant, "SPEECH_PROMPT_VARIANT")
    src_dir = repeat_dir / "eval_output" / "speech" / repeat_dir.name
    transcripts = read_json(src_dir / "transcripts.json")
    scored: List[Dict[str, Any]] = []
    with ThreadPoolExecutor(max_workers=workers) as ex:
        futures = [ex.submit(_speech_worker, module, item) for item in transcripts]
        for fut in as_completed(futures):
            scored.append(fut.result())
    scored.sort(key=lambda x: (x["category"], x["content"]))

    out_dir = out_repeat_dir / "speech" / repeat_dir.name
    write_json(out_dir / "transcripts.json", transcripts)
    write_json(out_dir / "scored.json", scored)
    write_jsonl(out_dir / "results.jsonl", scored)
    summary = {
        "variant": variant,
        "videos_root": str(videos_root),
        "prompts_dir": str(prompts_dir),
        "whisper_model": module.WHISPER_MODEL_NAME,
        "whisper_language": "en",
        "gemini_model": module.GEMINI_MODEL_NAME,
        "count_total_videos_found": len(scored),
        "count_transcribed": len(scored),
        "count_scored": len(scored),
        "avg_score": mean([float(x["judgement"]["score"]) for x in scored]),
        "by_category": build_summary_by_category(
            [
                {
                    "category": x["category"],
                    "score": x["judgement"]["score"],
                    "pass": x["judgement"]["pass"],
                }
                for x in scored
            ],
            "score",
            "pass",
        ),
        "note": "Reused transcripts.json from original run; Whisper/faster-whisper was not rerun.",
    }
    write_json(out_dir / "summary.json", summary)


def _plot_worker(module, variant: str, item: Dict[str, Any]) -> Dict[str, Any]:
    try:
        judge = module.judge_plot_matching(item["video_path"], item["prompt"])
        return {
            "category": item["category"],
            "content": item["content"],
            "prompt": item["prompt"],
            "video_path": item["video_path"],
            "status": "ok",
            "plot_alignment_score": judge["plot_alignment_score"],
            "subscores": judge.get("subscores", {}),
            "confidence": judge.get("confidence", 0.0),
            "summary": judge.get("summary", ""),
            "judge": judge,
            "variant": variant,
        }
    except Exception as e:
        return {
            "category": item["category"],
            "content": item["content"],
            "prompt": item["prompt"],
            "video_path": item["video_path"],
            "status": "failed",
            "error": repr(e),
            "variant": variant,
        }


def run_plot_variant(repeat_dir: Path, out_repeat_dir: Path, variant: str, manifest: List[Dict[str, Any]], workers: int) -> None:
    module = import_module_with_variant("eval.plot_matching.plot_matching_gemini", variant, "PLOT_MATCHING_PROMPT_VARIANT")
    rows: List[Dict[str, Any]] = []
    with ThreadPoolExecutor(max_workers=workers) as ex:
        futures = [ex.submit(_plot_worker, module, variant, item) for item in manifest]
        for fut in as_completed(futures):
            rows.append(fut.result())
    rows.sort(key=lambda x: (x.get("category", ""), x.get("content", "")))
    ok_rows = [r for r in rows if r["status"] == "ok"]
    out_dir = out_repeat_dir / "plot_matching" / repeat_dir.name
    summary = {
        "model": module.MODEL_NAME,
        "timeout_s": module.TIMEOUT_S,
        "workers": workers,
        "counts": {
            "total": len(rows),
            "ok": len(ok_rows),
            "failed": len(rows) - len(ok_rows),
        },
        "summaries_by_category": build_summary_by_category(ok_rows, "plot_alignment_score"),
        "overall": {"n": len(ok_rows), "avg_score": mean([float(r["plot_alignment_score"]) for r in ok_rows])},
        "results": rows,
        "variant": variant,
        "note": "Plot matching has no reusable small-model cache; Gemini was rerun directly on videos.",
    }
    write_json(out_dir / "eval_results.json", summary)


def _phy_worker(module, variant: str, cache_path: Path, out_dir: Path, item: Dict[str, Any]) -> Dict[str, Any]:
    out_json = out_dir / f"{Path(item['video_path']).name}.json"
    try:
        result = module.run_two_stage_with_dual_stage2(
            video_path=item["video_path"],
            prompt_text=item["prompt"],
            model_name=module.MODEL_NAME_DEFAULT,
            expectations_cache_path=str(cache_path),
            save_result_path=str(out_json),
            timeout_s=module.DEFAULT_TIMEOUT_S,
            weights=(0.5, 0.5),
        )
        return {
            "category": item["category"],
            "content": item["content"],
            "video_path": item["video_path"],
            "status": "ok",
            "overall_score": result["overall_score"],
            "semantic_physics_score": result["semantic_physics_score"],
            "visual_physics_score": result["visual_physics_score"],
            "confidence": result.get("confidence", 0.0),
        }
    except Exception as e:
        return {
            "category": item["category"],
            "content": item["content"],
            "video_path": item["video_path"],
            "status": "failed",
            "error": repr(e),
        }


def run_gemini_phy_variant(repeat_dir: Path, out_repeat_dir: Path, variant: str, manifest: List[Dict[str, Any]], workers: int) -> None:
    module = import_module_with_variant("eval.gemini_phy.batch_eval", variant, "GEMINI_PHY_PROMPT_VARIANT")
    variant_cache = out_repeat_dir / "gemini_phy2" / "expectations_cache.json"
    out_dir = out_repeat_dir / "gemini_phy2" / repeat_dir.name
    rows: List[Dict[str, Any]] = []
    with ThreadPoolExecutor(max_workers=workers) as ex:
        futures = [ex.submit(_phy_worker, module, variant, variant_cache, out_dir, item) for item in manifest]
        for fut in as_completed(futures):
            rows.append(fut.result())
    rows.sort(key=lambda x: (x.get("category", ""), x.get("content", "")))
    ok_rows = [r for r in rows if r["status"] == "ok"]
    write_csv(
        out_repeat_dir / "gemini_phy2" / repeat_dir.name / "results.csv",
        rows,
        ["category", "content", "video_path", "status", "overall_score", "semantic_physics_score", "visual_physics_score", "confidence", "error"],
    )
    write_csv(
        out_repeat_dir / "gemini_phy2" / repeat_dir.name / "summary.csv",
        [
            {
                "variant": variant,
                "n": len(ok_rows),
                "avg_overall_score": mean([float(r["overall_score"]) for r in ok_rows]),
                "avg_semantic_physics_score": mean([float(r["semantic_physics_score"]) for r in ok_rows]),
                "avg_visual_physics_score": mean([float(r["visual_physics_score"]) for r in ok_rows]),
            }
        ],
        ["variant", "n", "avg_overall_score", "avg_semantic_physics_score", "avg_visual_physics_score"],
    )
    write_json(
        out_repeat_dir / "gemini_phy2" / repeat_dir.name / "summary.json",
        {
            "variant": variant,
            "count_total": len(rows),
            "count_ok": len(ok_rows),
            "avg_overall_score": mean([float(r["overall_score"]) for r in ok_rows]),
            "avg_semantic_physics_score": mean([float(r["semantic_physics_score"]) for r in ok_rows]),
            "avg_visual_physics_score": mean([float(r["visual_physics_score"]) for r in ok_rows]),
            "by_category": build_summary_by_category(ok_rows, "overall_score"),
            "expectations_cache": str(variant_cache),
            "note": "Stage-1 expectations were regenerated with the selected prompt variant and cached under this variant-specific expectations_cache.json.",
        },
    )


def _extract_expected_minmax(spec: Dict[str, Any]) -> Tuple[int, int]:
    pf = spec.get("primary_faces", {}) or {}
    mn = int(pf.get("min_ids", 0) or 0)
    mx = int(pf.get("max_ids", mn) or mn)
    if mx < mn:
        mx = mn
    return mn, mx


def run_facial_variant(repeat_dir: Path, out_repeat_dir: Path, variant: str, workers: int) -> None:
    label_module = import_module_with_variant("eval.facial_consistency.gemini_label", variant, "FACIAL_PROMPT_VARIANT")
    facial_batch = import_module_with_variant("eval.facial_consistency.batch_eval", variant, "FACIAL_PROMPT_VARIANT")
    src_results = read_json(repeat_dir / "eval_output" / "facial" / repeat_dir.name / "eval_results.json")
    cached_rows = src_results.get("results", [])
    facial_workers = max(1, min(workers, int(os.getenv("FACIAL_GEMINI_WORKERS", "8"))))
    facial_retries = max(1, int(os.getenv("FACIAL_GEMINI_RETRIES", "4")))
    facial_timeout_s = max(30, int(os.getenv("FACIAL_GEMINI_TIMEOUT_S", "300")))

    def worker(row: Dict[str, Any]) -> Dict[str, Any]:
        if row.get("status") != "ok":
            return row
        last_err: Optional[Exception] = None
        for attempt in range(facial_retries):
            try:
                spec = label_module.gemini_analyze_prompt(row["prompt"], timeout_s=facial_timeout_s)
                break
            except Exception as exc:
                last_err = exc
                if attempt == facial_retries - 1:
                    raise
                sleep_s = (2 ** attempt) + random.uniform(0.0, 0.5)
                time.sleep(sleep_s)
        else:
            raise last_err if last_err is not None else RuntimeError("Unknown facial Gemini failure")
        mn, mx = _extract_expected_minmax(spec)
        scored = facial_batch.compute_final_score(
            expected_min=mn,
            expected_max=mx,
            metrics=row["metrics"],
            primary_ratio_thresh=src_results["config"]["primary_ratio_thresh"],
            min_cluster_dets=src_results["config"]["min_primary_cluster_dets"],
        )
        updated = dict(row)
        updated["expected_primary_faces"] = {"min_ids": mn, "max_ids": mx}
        updated["gemini_prompt_variant"] = variant
        updated["face_spec"] = spec
        updated["final_score"] = scored["final_score"]
        updated["breakdown"] = scored["breakdown"]
        updated["observed"] = scored["observed"]
        return updated

    updated_rows: List[Dict[str, Any]] = []
    with ThreadPoolExecutor(max_workers=facial_workers) as ex:
        futures = [ex.submit(worker, row) for row in cached_rows]
        for fut in as_completed(futures):
            updated_rows.append(fut.result())
    updated_rows.sort(key=lambda x: (x.get("category", ""), x.get("content", "")))
    ok_rows = [r for r in updated_rows if r.get("status") == "ok"]
    out_dir = out_repeat_dir / "facial" / repeat_dir.name
    write_json(
        out_dir / "eval_results.json",
        {
            "config": src_results.get("config", {}),
            "summaries_by_category": build_summary_by_category(ok_rows, "final_score"),
            "overall": {"n": len(ok_rows), "avg_score": mean([float(r["final_score"]) for r in ok_rows])},
            "results": updated_rows,
            "variant": variant,
            "note": "Reused cached facial metrics from original eval_results.json and only reran Gemini face-spec extraction.",
        },
    )


def try_reuse_music_raw_events(module, cached_result: Dict[str, Any]) -> Tuple[Optional[List[Dict[str, Any]]], Optional[List[Dict[str, Any]]], Optional[str]]:
    artifacts = cached_result.get("_artifacts", {}) or {}
    midi_rel = artifacts.get("midi_path")
    if not midi_rel:
        return None, None, "No cached MIDI path recorded."
    midi_path = (REPO_ROOT / midi_rel).resolve()
    if not midi_path.exists():
        return None, None, f"Cached MIDI not found: {midi_path}"
    raw_events = module.parse_midi_note_events(str(midi_path))
    chord_frames = module.build_chord_frames(raw_events)
    return raw_events, chord_frames, None


def run_music_variant(repeat_dir: Path, out_repeat_dir: Path, variant: str, workers: int) -> None:
    module = import_module_with_variant("eval.music_check.music_check_gemini", variant, "MUSIC_PROMPT_VARIANT")
    src_dir = repeat_dir / "eval_output" / "music" / repeat_dir.name / "per_video_result"
    out_dir = out_repeat_dir / "music" / repeat_dir.name
    out_result_dir = out_dir / "per_video_result"
    rows: List[Dict[str, Any]] = []

    cached_files = sorted(src_dir.rglob("*.music_raw_midi_result.json"))

    def worker(src_path: Path) -> Dict[str, Any]:
        cached = read_json(src_path)
        prompt_text = cached["_input"]["prompt_text"]
        video_path = cached["_input"]["video_path"]
        category = src_path.parent.name
        content = src_path.name.replace(".music_raw_midi_result.json", "")
        out_path = out_result_dir / category / src_path.name
        constraints_obj = module.gemini_extract_constraints(prompt_text)
        if constraints_obj.get("skipped") is True:
            result = {
                "overall_score": 0,
                "confidence": 0.0,
                "skipped": True,
                "skip_reason": constraints_obj.get("skip_reason"),
                "constraints": constraints_obj.get("constraints", {}),
                "checks": [{"id": "skip", "aspect": "other", "status": "uncertain", "severity": 1, "evidence": constraints_obj.get("skip_reason", "")}],
                "violations": [],
                "subscores": {"chord": None, "sequence_or_scale": None},
                "interpretation": {"best_chord_frame": None, "best_chord_pitch_classes": [], "best_sequence_pitch_classes": [], "notes_used_as_evidence": []},
                "summary": constraints_obj.get("skip_reason", "skipped"),
                "_input": {"video_path": video_path, "prompt_text": prompt_text, "variant": variant},
                "_reuse": {"basic_pitch_reused": False, "reason": "Skipped by stage-1 Gemini."},
            }
            write_json(out_path, result)
            return {"category": category, "content": content, "status": "ok", "overall_score": 0, "reuse_reason": "stage1_skip"}

        raw_events, chord_frames, reuse_err = try_reuse_music_raw_events(module, cached)
        if raw_events is not None and chord_frames is not None:
            judgement = module.gemini_validate(prompt_text, constraints_obj, raw_events, chord_frames)
            result = {
                **judgement,
                "skipped": False,
                "skip_reason": None,
                "constraints": constraints_obj.get("constraints", constraints_obj),
                "_input": {"video_path": video_path, "prompt_text": prompt_text, "variant": variant},
                "_reuse": {"basic_pitch_reused": True, "reason": "Rebuilt raw events from cached MIDI artifact."},
            }
            write_json(out_path, result)
            return {"category": category, "content": content, "status": "ok", "overall_score": result["overall_score"], "reuse_reason": "cached_midi"}

        result = module.judge_video_music_raw_midi(
            video_path=video_path,
            prompt_text=prompt_text,
            out_path=str(out_path),
            constraints_cache_dir=str(out_dir / "music_prompt_constraints"),
            timeout_s=module.DEFAULT_TIMEOUT_S,
        )
        result["_reuse"] = {"basic_pitch_reused": False, "reason": reuse_err or "Cached MIDI unavailable; Basic Pitch rerun was required."}
        write_json(out_path, result)
        return {
            "category": category,
            "content": content,
            "status": "ok",
            "overall_score": result.get("overall_score", 0),
            "reuse_reason": result["_reuse"]["reason"],
        }

    with ThreadPoolExecutor(max_workers=workers) as ex:
        futures = [ex.submit(worker, p) for p in cached_files]
        for fut in as_completed(futures):
            rows.append(fut.result())

    rows.sort(key=lambda x: (x.get("category", ""), x.get("content", "")))
    summary = {
        "variant": variant,
        "count_total": len(rows),
        "avg_score": mean([float(r["overall_score"]) for r in rows if r.get("status") == "ok"]),
        "by_category": build_summary_by_category([r for r in rows if r.get("status") == "ok"], "overall_score"),
        "reuse_breakdown": {
            key: sum(1 for r in rows if r.get("reuse_reason") == key)
            for key in sorted({str(r.get("reuse_reason", "")) for r in rows})
        },
        "note": "Music reuses cached MIDI only if the old .mid artifact still exists; otherwise it falls back to rerunning Basic Pitch.",
    }
    write_json(out_dir / "summary.json", summary)


def run_repeat(
    repeat_dir: Path,
    out_root: Path,
    variant: str,
    prompts_dir: Path,
    workers: int,
    modules: List[str],
) -> None:
    out_repeat_dir = out_root / variant / repeat_dir.name
    manifest = load_master_manifest(repeat_dir)
    videos_root = infer_videos_root(manifest)
    prompts_index = build_prompt_index(prompts_dir)

    if "ocr" in modules:
        run_ocr_variant(repeat_dir, out_repeat_dir, variant, prompts_index, videos_root, workers)
    if "speech" in modules:
        run_speech_variant(repeat_dir, out_repeat_dir, variant, videos_root, prompts_dir, workers)
    if "plot_matching" in modules:
        run_plot_variant(repeat_dir, out_repeat_dir, variant, manifest, workers)
    if "gemini_phy" in modules:
        run_gemini_phy_variant(repeat_dir, out_repeat_dir, variant, manifest, workers)
    if "facial" in modules:
        run_facial_variant(repeat_dir, out_repeat_dir, variant, workers)
    if "music" in modules:
        run_music_variant(repeat_dir, out_repeat_dir, variant, workers)


def main() -> None:
    parser = argparse.ArgumentParser(description="Rerun Gemini prompt variants while reusing cached small-model artifacts from an existing stability session.")
    parser.add_argument(
        "--session_model_dir",
        type=Path,
        default=Path("/home/v-wangrui5/AVGen-Bench/stability_eval_runs/session_20260328T180932Z_4d0f84/Veo_3.1_fast"),
        help="Path to the model-specific stability session directory. This directory is treated as the original-prompt baseline.",
    )
    parser.add_argument(
        "--prompts_dir",
        type=Path,
        default=Path("/home/v-wangrui5/AVGen-Bench/prompts"),
    )
    parser.add_argument(
        "--variants",
        nargs="+",
        default=["v1", "v2"],
    )
    parser.add_argument(
        "--modules",
        nargs="+",
        default=["ocr", "speech", "plot_matching", "gemini_phy", "facial", "music"],
        choices=["ocr", "speech", "plot_matching", "gemini_phy", "facial", "music"],
    )
    parser.add_argument("--workers", type=int, default=16)
    parser.add_argument(
        "--out_root",
        type=Path,
        default=None,
        help="Default: <session_model_dir>/prompt_variants",
    )
    args = parser.parse_args()

    session_model_dir = args.session_model_dir.resolve()
    out_root = (args.out_root or (session_model_dir / "prompt_variants")).resolve()
    repeats = find_repeats(session_model_dir)
    if not repeats:
        raise RuntimeError(f"No repeat_* directories found under {session_model_dir}")

    baseline = {
        "baseline_type": "original_prompt_results",
        "baseline_path": str(session_model_dir),
        "variants": args.variants,
        "modules": args.modules,
        "repeats": [p.name for p in repeats],
    }
    write_json(out_root / "baseline.json", baseline)

    for variant in args.variants:
        for repeat_dir in repeats:
            print(f"[variant={variant}] [repeat={repeat_dir.name}] starting", flush=True)
            run_repeat(
                repeat_dir=repeat_dir,
                out_root=out_root,
                variant=variant,
                prompts_dir=args.prompts_dir.resolve(),
                workers=args.workers,
                modules=args.modules,
            )
            print(f"[variant={variant}] [repeat={repeat_dir.name}] done", flush=True)


if __name__ == "__main__":
    main()
