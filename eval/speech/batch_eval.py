#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import re
import json
import time
import argparse
from typing import Dict, Any, List, Tuple
from concurrent.futures import ThreadPoolExecutor, as_completed

import torch
from faster_whisper import WhisperModel
import google.generativeai as genai


# -------------------------
# Utils
# -------------------------
def safe_filename(name: str, max_len: int = 180) -> str:
    name = str(name).strip()
    name = re.sub(r"[\/\\\:\*\?\"\<\>\|\n\r\t]", "_", name)
    name = re.sub(r"\s+", " ", name).strip()
    if not name:
        name = "untitled"
    if len(name) > max_len:
        name = name[:max_len].rstrip()
    return name


def read_json(path: str) -> Any:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def write_json(path: str, obj: Any):
    d = os.path.dirname(path)
    if d:
        os.makedirs(d, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)


def append_jsonl(path: str, obj: Any):
    d = os.path.dirname(path)
    if d:
        os.makedirs(d, exist_ok=True)
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(obj, ensure_ascii=False) + "\n")


# -------------------------
# Faster-Whisper (no manual filtering)
# -------------------------
def transcribe_with_faster_whisper(
    model: WhisperModel,
    video_path: str,
    language: str = "en",
    beam_size: int = 1,
    temperature: float = 0.0,
    vad_filter: bool = True,
    vad_min_silence_ms: int = 800,
    vad_speech_pad_ms: int = 200,
) -> Dict[str, Any]:
    if not os.path.exists(video_path):
        raise FileNotFoundError(f"Video not found: {video_path}")

    segments_iter, info = model.transcribe(
        video_path,
        language=language,
        task="transcribe",
        beam_size=beam_size,
        temperature=temperature,
        vad_filter=vad_filter,
        vad_parameters={
            "min_silence_duration_ms": vad_min_silence_ms,
            "speech_pad_ms": vad_speech_pad_ms,
            "threshold": 0.5,
            "min_speech_duration_ms":100,
        } if vad_filter else None,
    )

    segs = []
    texts = []
    for s in segments_iter:
        txt = (s.text or "").strip()
        if txt:
            texts.append(txt)
        segs.append(
            {
                "start": float(s.start),
                "end": float(s.end),
                "text": txt,
                "avg_logprob": float(getattr(s, "avg_logprob", 0.0)),
                "compression_ratio": float(getattr(s, "compression_ratio", 0.0)),
                "no_speech_prob": float(getattr(s, "no_speech_prob", 0.0)),
            }
        )

    return {
        "text": " ".join(texts).strip(),
        "segments": segs,
        "language": getattr(info, "language", language),
        "language_probability": float(getattr(info, "language_probability", 0.0)),
        "duration": float(getattr(info, "duration", 0.0)),
    }


# -------------------------
# Gemini
# -------------------------
def _strip_code_fences(text: str) -> str:
    t = (text or "").strip()
    if t.startswith("```"):
        t = t.split("\n", 1)[-1]
        if t.endswith("```"):
            t = t[:-3]
        t = t.strip()
    return t


def evaluate_speech_with_gemini(
    generation_prompt: str,
    transcript_text: str,
    model_name: str,
    api_key: str,
    timeout_s: int = 600,
    max_retries: int = 3,
    retry_backoff_s: float = 2.0,
) -> Dict[str, Any]:
    if not api_key:
        raise RuntimeError("Missing GEMINI_API_KEY.")
    genai.configure(api_key=api_key)
    model = genai.GenerativeModel(model_name=model_name)

    evaluator_prompt = f"""
You are a speech transcript compliance auditor.

Goal:
Given (A) the prompt used to generate an AI video and (B) the speech transcript,
judge how well the spoken content satisfies the prompt.

Key definition (IMPORTANT):
- "explicit_speech_required" MUST be true if the prompt implies there should be spoken audio.
  This includes BOTH:
  (1) Explicit exact lines / a script / quotes / dialogue that must be spoken.
  (2) Any clear implication that there is speech, even without exact wording, e.g.:
      "voiceover", "narration", "the character says ...", "dialogue", "announcer reads ...",
      "the speaker explains ...", "she introduces the product", "he delivers a monologue",
      or any instruction that someone speaks on-camera.
- "explicit_speech_required" MUST be false only if the prompt does not require speech
  (e.g., purely visual prompt), OR it explicitly requests silence/no voice/no dialogue.

Scoring modes:
- If the prompt provides required dialogue/voiceover lines or demands exact wording,
  evaluate in VERBATIM mode: required line(s) must appear with identical wording
  (ignore minor punctuation/case/spacing only).
- Otherwise evaluate in CONTEXTUAL mode: speech only needs to be consistent with the prompt’s
  scene/message/constraints.
  If explicit_speech_required is false and the transcript is empty or near-empty, that can still be compliant.
  If explicit_speech_required is true and the transcript is empty or near-empty, that should score very low.

Return STRICT JSON ONLY (no Markdown, no code fences) using this schema:
{{
  "explicit_speech_required": true/false,
  "match_type": "verbatim" | "contextual",
  "score": 0,
  "pass": true/false,
  "required_speech_lines": ["..."],
  "verbatim_match_details": [{{"required_line":"...","found":true/false,"matched_text":"...","diff_summary":"..."}}],
  "missing_lines": ["..."],
  "extra_or_mismatched_segments": ["..."],
  "normalized_transcript": "...",
  "score_rationale": "...",
  "suggested_fix": "..."
}}

Hard constraints:
- score must be an integer 0..100.
- pass must be true iff score >= 80.

Inputs:
PROMPT_USED_TO_GENERATE_VIDEO:
{generation_prompt}

TRANSCRIPT_RAW:
{transcript_text}
""".strip()

    last_err = None
    for attempt in range(1, max_retries + 1):
        try:
            resp = model.generate_content(
                evaluator_prompt,
                request_options={"timeout": timeout_s},
            )
            text = _strip_code_fences(resp.text)
            data = json.loads(text)

            try:
                score = int(data.get("score", 0))
            except Exception:
                score = 0
            score = max(0, min(100, score))
            data["score"] = score
            data["pass"] = bool(score >= 80)

            if "explicit_speech_required" not in data:
                data["explicit_speech_required"] = False
            if data.get("match_type") not in ("verbatim", "contextual"):
                data["match_type"] = "contextual"

            return data
        except Exception as e:
            last_err = e
            if attempt < max_retries:
                time.sleep(retry_backoff_s * attempt)
            else:
                raise last_err


# -------------------------
# Data discovery / mapping
# -------------------------
def load_items(prompts_dir: str) -> List[Dict[str, Any]]:
    items = []
    for fn in sorted(os.listdir(prompts_dir)):
        if not fn.endswith(".json"):
            continue
        category = os.path.splitext(fn)[0]
        path = os.path.join(prompts_dir, fn)
        arr = read_json(path)
        if not isinstance(arr, list):
            continue
        for idx, it in enumerate(arr):
            if not isinstance(it, dict):
                continue
            content = it.get("content", "")
            prompt = it.get("prompt", "")
            if not content or not prompt:
                continue
            items.append(
                {
                    "category": category,
                    "index_in_file": idx,
                    "content": content,
                    "prompt": prompt,
                    "prompt_file": path,
                }
            )
    return items


def resolve_video_path(videos_root: str, category: str, content: str, ext: str = ".mp4") -> str:
    return os.path.join(videos_root, category, safe_filename(content) + ext)


# -------------------------
# Main
# -------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--videos_root", type=str, required=True)
    ap.add_argument("--prompts_dir", type=str, required=True)
    ap.add_argument("--out_dir", type=str, required=True)
    ap.add_argument("--video_ext", type=str, default=".mp4")

    # faster-whisper model
    ap.add_argument("--whisper_model", type=str, default=os.getenv("WHISPER_MODEL_NAME", "large-v3"))
    ap.add_argument("--compute_type", type=str, default=os.getenv("FW_COMPUTE_TYPE", "float16"),
                    help="ctranslate2 compute_type: float16/int8_float16/int8 etc.")
    ap.add_argument("--beam_size", type=int, default=1)
    ap.add_argument("--temperature", type=float, default=0.0)
    ap.add_argument("--whisper_language", type=str, default="en")

    # VAD
    ap.add_argument("--vad_filter", action="store_true", default=True)
    ap.add_argument("--no_vad_filter", action="store_true", default=False)
    ap.add_argument("--vad_min_silence_ms", type=int, default=300)
    ap.add_argument("--vad_speech_pad_ms", type=int, default=500)

    # Gemini
    ap.add_argument("--gemini_model", type=str, default=os.getenv("GEMINI_MODEL_NAME", "gemini-3-flash-preview"))
    ap.add_argument("--gemini_workers", type=int, default=int(os.getenv("GEMINI_WORKERS", "32")))
    ap.add_argument("--gemini_timeout", type=int, default=600)
    ap.add_argument("--gemini_retries", type=int, default=3)

    ap.add_argument("--skip_existing", action="store_true")
    args = ap.parse_args()

    api_key = os.getenv("GEMINI_API_KEY", "")
    if not api_key:
        raise RuntimeError("Missing GEMINI_API_KEY env var.")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[info] device={device}")
    print(f"[info] faster_whisper_model={args.whisper_model} language={args.whisper_language} beam={args.beam_size} temp={args.temperature}")
    vad_filter = False if args.no_vad_filter else args.vad_filter
    print(f"[info] vad_filter={vad_filter} compute_type={args.compute_type}")
    print(f"[info] gemini_model={args.gemini_model} workers={args.gemini_workers}")

    items = load_items(args.prompts_dir)
    if not items:
        raise RuntimeError(f"No prompt items found in: {args.prompts_dir}")

    for it in items:
        it["video_path"] = resolve_video_path(args.videos_root, it["category"], it["content"], ext=args.video_ext)

    results_jsonl = os.path.join(args.out_dir, "results.jsonl")
    existing = set()
    if args.skip_existing and os.path.exists(results_jsonl):
        with open(results_jsonl, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    obj = json.loads(line)
                    vp = obj.get("video_path")
                    if vp:
                        existing.add(vp)
                except Exception:
                    continue
        before = len(items)
        items = [it for it in items if it["video_path"] not in existing]
        print(f"[info] skip_existing: {before-len(items)} skipped, {len(items)} remaining")

    missing = [it for it in items if not os.path.exists(it["video_path"])]
    if missing:
        miss_path = os.path.join(args.out_dir, "missing_videos.json")
        write_json(miss_path, missing)
        print(f"[warn] {len(missing)} videos missing. Wrote: {miss_path}")

    items = [it for it in items if os.path.exists(it["video_path"])]
    if not items:
        print("[info] nothing to process.")
        return

    # 1) Transcribe (sequential)
    print(f"[stage] Transcribing {len(items)} videos with faster-whisper...")
    fw = WhisperModel(
        args.whisper_model,
        device=device,
        compute_type=args.compute_type if device == "cuda" else "int8",
    )

    transcripts: List[Dict[str, Any]] = []
    t0 = time.time()
    for i, it in enumerate(items, 1):
        vp = it["video_path"]
        try:
            wres = transcribe_with_faster_whisper(
                fw,
                vp,
                language=args.whisper_language,
                beam_size=args.beam_size,
                temperature=args.temperature,
                vad_filter=vad_filter,
                vad_min_silence_ms=args.vad_min_silence_ms,
                vad_speech_pad_ms=args.vad_speech_pad_ms,
            )

            transcripts.append(
                {
                    **it,
                    "whisper_meta": {
                        "engine": "faster-whisper",
                        "language": wres.get("language"),
                        "language_probability": wres.get("language_probability"),
                        "duration": wres.get("duration"),
                        "beam_size": args.beam_size,
                        "temperature": args.temperature,
                        "vad_filter": vad_filter,
                        "vad_min_silence_ms": args.vad_min_silence_ms,
                        "vad_speech_pad_ms": args.vad_speech_pad_ms,
                        "compute_type": args.compute_type,
                    },
                    "transcript": (wres.get("text") or "").strip(),
                }
            )
        except Exception as e:
            append_jsonl(
                results_jsonl,
                {
                    **it,
                    "video_path": vp,
                    "error_stage": "faster_whisper",
                    "error": repr(e),
                },
            )

        if i % 10 == 0 or i == len(items):
            print(f"  - {i}/{len(items)} done")

    t1 = time.time()
    write_json(os.path.join(args.out_dir, "transcripts.json"), transcripts)
    print(f"[stage] faster-whisper done. ok={len(transcripts)}/{len(items)} time={t1-t0:.1f}s")

    # 2) Gemini evaluate in parallel
    print(f"[stage] Gemini evaluating {len(transcripts)} items (workers={args.gemini_workers})...")
    scored: List[Dict[str, Any]] = []

    def _worker(rec: Dict[str, Any]) -> Tuple[str, Dict[str, Any]]:
        j = evaluate_speech_with_gemini(
            generation_prompt=rec["prompt"],
            transcript_text=rec["transcript"],
            model_name=args.gemini_model,
            api_key=api_key,
            timeout_s=args.gemini_timeout,
            max_retries=args.gemini_retries,
        )
        out = {
            "video_path": rec["video_path"],
            "category": rec["category"],
            "content": rec["content"],
            "prompt": rec["prompt"],
            "transcript": rec["transcript"],
            "whisper_meta": rec.get("whisper_meta", {}),
            "judgement": j,
        }
        return rec["video_path"], out

    t2 = time.time()
    with ThreadPoolExecutor(max_workers=args.gemini_workers) as ex:
        futs = [ex.submit(_worker, rec) for rec in transcripts]
        for k, fut in enumerate(as_completed(futs), 1):
            try:
                _, out = fut.result()
                scored.append(out)
                append_jsonl(results_jsonl, out)
            except Exception as e:
                append_jsonl(results_jsonl, {"error_stage": "gemini", "error": repr(e)})

            if k % 10 == 0 or k == len(futs):
                print(f"  - {k}/{len(futs)} done")

    t3 = time.time()
    print(f"[stage] Gemini done. ok={len(scored)}/{len(transcripts)} time={t3-t2:.1f}s")

    # 3) Summary
    summary = {
        "videos_root": args.videos_root,
        "prompts_dir": args.prompts_dir,
        "whisper_model": args.whisper_model,
        "whisper_language": args.whisper_language,
        "gemini_model": args.gemini_model,
        "count_total_videos_found": len(items),
        "count_transcribed": len(transcripts),
        "count_scored": len(scored),
        "avg_score": (
            sum(x["judgement"].get("score", 0) for x in scored) / len(scored) if scored else None
        ),
        "by_category": {},
    }

    by_cat: Dict[str, List[int]] = {}
    for x in scored:
        cat = x.get("category", "unknown")
        by_cat.setdefault(cat, []).append(int(x["judgement"].get("score", 0)))
    for cat, arr in by_cat.items():
        summary["by_category"][cat] = {
            "n": len(arr),
            "avg_score": sum(arr) / len(arr) if arr else None,
            "pass_rate": sum(1 for s in arr if s >= 80) / len(arr) if arr else None,
        }

    write_json(os.path.join(args.out_dir, "summary.json"), summary)
    write_json(os.path.join(args.out_dir, "scored.json"), scored)
    print(
        "[done] wrote:\n"
        f"  - {results_jsonl}\n"
        f"  - {os.path.join(args.out_dir,'transcripts.json')}\n"
        f"  - {os.path.join(args.out_dir,'scored.json')}\n"
        f"  - {os.path.join(args.out_dir,'summary.json')}"
    )


if __name__ == "__main__":
    main()
