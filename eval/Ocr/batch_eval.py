import os
import re
import json
import time
import math
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import cv2
from paddleocr import PaddleOCR
import google.generativeai as genai


# =========================
# Env / Config
# =========================
MODEL_NAME = "gemini-3-flash-preview"
TIMEOUT_S = int(os.getenv("GEMINI_TIMEOUT_S", "300"))

# OCR sampling (speed vs accuracy)
STRIDE = int(os.getenv("OCR_STRIDE", "3"))          # run OCR every N frames
MAX_FRAMES = int(os.getenv("OCR_MAX_FRAMES", "900"))# cap processed frames
START_FRAME = int(os.getenv("OCR_START_FRAME", "0"))

# Same-frame merge params
MERGE_Y_IOU_THRESH = float(os.getenv("OCR_MERGE_Y_IOU_THRESH", "0.5"))
MERGE_X_GAP_NORM = float(os.getenv("OCR_MERGE_X_GAP_NORM", "0.02"))
MERGE_MAX_HEIGHT_RATIO = float(os.getenv("OCR_MERGE_MAX_HEIGHT_RATIO", "2.5"))

# Tracking params
IOU_THRESH = float(os.getenv("OCR_IOU_THRESH", "0.3"))
MAX_GAP_FRAMES = int(os.getenv("OCR_MAX_GAP_FRAMES", "5"))
MIN_TRACK_FRAMES = int(os.getenv("OCR_MIN_TRACK_FRAMES", "3"))
MAX_TRACKS_TO_SEND = int(os.getenv("OCR_MAX_TRACKS_TO_SEND", "40"))
MIN_OCR_CONF = float(os.getenv("OCR_MIN_CONF", "0.2"))

# Text segmentation
TEXT_SIM_THRESH = float(os.getenv("OCR_TEXT_SIM_THRESH", "0.75"))

# OCR init (match your earlier settings)
OCR_ENGINE = PaddleOCR(
    use_doc_orientation_classify=False,
    use_doc_unwarping=False,
    use_textline_orientation=False
)


# =========================
# Gemini prompt
# =========================
GEMINI_SYSTEM = """
You are a strict judge for generated-video text quality (OCR-based evaluation).

IMPORTANT SCORING RULE:
- ALL scores are integers in the range [0, 100].
- This includes overall_text_quality_score and every subscores field.
- Do NOT output percentages with '%' sign; output plain integers.

Input:
- Prompt text for audio-video generation
- Tracked text regions ("text tracks") summarized across frames:
  Each track includes normalized bbox, duration, confidence stats, and text-change segments.

Task: Evaluate VISIBLE on-screen text generation quality (OCR-based):

ADDITIONAL REQUIRED OUTPUT FIELDS (classification; not scores):
- prompt_requires_visible_text (boolean):
  True ONLY if the prompt explicitly requires visible on-screen text (e.g., subtitles/captions, title card, on-screen labels, UI text, signs to read).
  False if text is not explicitly required. Spoken dialogue/quotes/voiceover do NOT imply visible text unless subtitles/captions are explicitly requested.
- text_presence (string enum: "none" | "noise" | "incidental" | "required"):
  Use ONLY the provided track summaries as evidence.
  "none": No credible visible text tracks.
  "noise": Only obvious OCR false positives / noise (e.g.a, single-character like "O/0/I", random fragments from textures/reflections/edges, huge/fullscreen boxes, very short-lived tracks) and no credible readable text.
  "incidental": Credible readable visible text exists, but the prompt does NOT explicitly require visible text (e.g., natural signs, packaging, UI elements that fit the scene).
  "required": Credible readable visible text exists AND the prompt explicitly requires visible text.
- missing_required_text (boolean):
  True if prompt_requires_visible_text is True but text_presence is "none" or "noise" (i.e., required visible text is missing).
  False otherwise.
- incidental_text_is_contextual (boolean or null):
  If text_presence is "incidental", set True if the incidental text is plausible/contextual and not distracting; set False if it is nonsensical/out-of-place.
  If text_presence is "none" or "noise" or "required", set this to null.

Then evaluate visible on-screen text generation quality (OCR-based):
- Legibility/accuracy
- Temporal stability
- Spatial stability
- Completeness
- Prompt match (ONLY if the prompt explicitly requires visible on-screen text such as subtitles/captions/titles/signs/UI/labels.
Do NOT treat spoken dialogue/voiceover/quotes in the prompt as a requirement for on-screen text unless subtitles/captions are explicitly requested.)
- Noise handling: If the prompt does NOT require on-screen text, treat obvious OCR false-positives as noise (e.g., single-character like "O/0/I" caused by circular patterns, reflections, textures, or edges; huge/fullscreen boxes; very short-lived tracks). Do not penalize overall score for such noise; instead report them in issues as false_positive/noise.

Rules:
- Use only the provided track summaries as evidence.
- Output STRICT JSON only (no markdown, no code fences).
""".strip()




GEMINI_USER_TEMPLATE = """
PROMPT_TEXT:
{prompt_text}

VIDEO_TEXT_TRACKS_JSON:
{tracks_json}

Output STRICT JSON (ALL score fields must be integer 0-100; confidence must be float 0-1):
{{
  "prompt_requires_visible_text": false,
  "text_presence": "none",
  "missing_required_text": false,
  "incidental_text_is_contextual": null,

  "overall_text_quality_score": 0,
  "subscores": {{
    "legibility_accuracy": 0,
    "temporal_stability": 0,
    "spatial_stability": 0,
    "completeness": 0,
    "prompt_text_match": null
  }},
  "confidence": 0.0,
  "top_issues": [
    {{
      "track_id": "T1",
      "issue_type": "flicker|character_drift|bbox_jitter|missing|nonsense_text|low_confidence|false_positive|noise|other",
      "severity": 1,
      "evidence": "cite segments and stats from the track"
    }}
  ],
  "good_tracks": ["T3","T7"],
  "summary": "short assessment"
}}
""".strip()




# =========================
# Utils
# =========================
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

def clamp01(x: Any) -> float:
    try:
        v = float(x)
    except Exception:
        v = 0.0
    return max(0.0, min(1.0, v))

def clamp100(x: Any) -> int:
    try:
        v = int(x)
    except Exception:
        v = 0
    return max(0, min(100, v))

def poly_to_aabb(poly: List[List[float]]) -> Tuple[float,float,float,float]:
    xs = [p[0] for p in poly]
    ys = [p[1] for p in poly]
    return (min(xs), min(ys), max(xs), max(ys))

def normalize_aabb(a: Tuple[float,float,float,float], w: int, h: int) -> Tuple[float,float,float,float]:
    x1,y1,x2,y2 = a
    w = max(1, int(w))
    h = max(1, int(h))
    return (x1/w, y1/h, x2/w, y2/h)

def aabb_iou(a: Tuple[float,float,float,float], b: Tuple[float,float,float,float]) -> float:
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    iw, ih = max(0.0, ix2 - ix1), max(0.0, iy2 - iy1)
    inter = iw * ih
    area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    union = area_a + area_b - inter
    return inter / union if union > 0 else 0.0

def aabb_center(a: Tuple[float,float,float,float]) -> Tuple[float,float]:
    x1,y1,x2,y2 = a
    return ((x1+x2)/2.0, (y1+y2)/2.0)

def aabb_height(a: Tuple[float,float,float,float]) -> float:
    return max(0.0, a[3]-a[1])

def aabb_width(a: Tuple[float,float,float,float]) -> float:
    return max(0.0, a[2]-a[0])

def y_iou(a: Tuple[float,float,float,float], b: Tuple[float,float,float,float]) -> float:
    ay1, ay2 = a[1], a[3]
    by1, by2 = b[1], b[3]
    iy1, iy2 = max(ay1, by1), min(ay2, by2)
    inter = max(0.0, iy2 - iy1)
    union = max(0.0, ay2 - ay1) + max(0.0, by2 - by1) - inter
    return inter / union if union > 0 else 0.0

def text_similarity(a: str, b: str) -> float:
    a = (a or "").strip()
    b = (b or "").strip()
    if not a and not b:
        return 1.0
    if not a or not b:
        return 0.0
    try:
        from rapidfuzz.fuzz import ratio
        return ratio(a, b) / 100.0
    except Exception:
        import difflib
        return difflib.SequenceMatcher(None, a, b).ratio()


# =========================
# Data structures
# =========================
@dataclass
class Det:
    frame: int
    t: float
    text: str
    conf: float
    poly: List[List[float]]
    aabb: Tuple[float,float,float,float]
    aabb_norm: Tuple[float,float,float,float]

@dataclass
class Track:
    track_id: str
    dets: List[Det] = field(default_factory=list)
    last_frame: int = -1

    def add(self, d: Det):
        self.dets.append(d)
        self.last_frame = d.frame


# =========================
# OCR per frame
# =========================
def ocr_frame(frame_img: np.ndarray) -> List[Tuple[List[List[float]], str, float]]:
    """
    Returns list of (poly_points, text, conf).
    Compatible with PaddleOCR predict output variants.
    """
    result = OCR_ENGINE.predict(input=frame_img)
    out = []

    for res in result:
        # PaddleX style dict
        if isinstance(res, dict) and "rec_texts" in res and "rec_scores" in res:
            texts = res.get("rec_texts", [])
            scores = res.get("rec_scores", [])
            polys = res.get("dt_polys", []) or []
            for text, score, poly in zip(texts, scores, polys):
                poly = poly.tolist() if hasattr(poly, "tolist") else poly
                out.append((poly, str(text), float(score)))
        # Older list style [[poly, (text, score)], ...]
        elif isinstance(res, list):
            for line in res:
                poly = line[0]
                text = line[1][0]
                score = line[1][1]
                poly = poly.tolist() if hasattr(poly, "tolist") else poly
                out.append((poly, str(text), float(score)))

    return out


# =========================
# Same-frame merge
# =========================
def _merge_group(group: List[Det]) -> Det:
    frame = group[0].frame
    t = group[0].t
    text = "".join([g.text for g in group])  # no separator (works ok for CN/EN mixed)

    xs1 = [g.aabb_norm[0] for g in group]
    ys1 = [g.aabb_norm[1] for g in group]
    xs2 = [g.aabb_norm[2] for g in group]
    ys2 = [g.aabb_norm[3] for g in group]
    aabb_norm = (min(xs1), min(ys1), max(xs2), max(ys2))

    px1 = [g.aabb[0] for g in group]
    py1 = [g.aabb[1] for g in group]
    px2 = [g.aabb[2] for g in group]
    py2 = [g.aabb[3] for g in group]
    aabb = (min(px1), min(py1), max(px2), max(py2))

    areas = []
    for g in group:
        areas.append(max(1e-6, aabb_width(g.aabb_norm) * aabb_height(g.aabb_norm)))
    conf = float(np.average([g.conf for g in group], weights=areas))

    return Det(frame=frame, t=t, text=text, conf=conf, poly=group[0].poly, aabb=aabb, aabb_norm=aabb_norm)


def merge_dets_in_frame(dets: List[Det]) -> List[Det]:
    if not dets:
        return []

    dets_sorted = sorted(dets, key=lambda d: (aabb_center(d.aabb_norm)[1], d.aabb_norm[0]))

    # line clustering
    lines: List[List[Det]] = []
    for d in dets_sorted:
        placed = False
        for line in lines:
            rep = line[-1]
            if y_iou(rep.aabb_norm, d.aabb_norm) >= MERGE_Y_IOU_THRESH:
                h1 = aabb_height(rep.aabb_norm)
                h2 = aabb_height(d.aabb_norm)
                if h1 > 0 and h2 > 0:
                    ratio = max(h1, h2) / max(1e-6, min(h1, h2))
                    if ratio <= MERGE_MAX_HEIGHT_RATIO:
                        line.append(d)
                        placed = True
                        break
        if not placed:
            lines.append([d])

    merged: List[Det] = []
    for line in lines:
        line = sorted(line, key=lambda d: d.aabb_norm[0])
        cur = [line[0]]
        for d in line[1:]:
            prev = cur[-1]
            gap = d.aabb_norm[0] - prev.aabb_norm[2]
            if gap <= MERGE_X_GAP_NORM:
                cur.append(d)
            else:
                merged.append(_merge_group(cur))
                cur = [d]
        merged.append(_merge_group(cur))

    merged.sort(key=lambda d: (d.frame, -d.conf))
    return merged


# =========================
# Tracking
# =========================
def build_tracks(dets: List[Det]) -> List[Track]:
    by_frame: Dict[int, List[Det]] = {}
    for d in dets:
        by_frame.setdefault(d.frame, []).append(d)

    active: List[Track] = []
    finished: List[Track] = []
    next_id = 1

    for f in sorted(by_frame.keys()):
        frame_dets = by_frame[f]

        # expire
        keep = []
        for tr in active:
            if f - tr.last_frame <= MAX_GAP_FRAMES:
                keep.append(tr)
            else:
                finished.append(tr)
        active = keep

        used = set()
        for d in sorted(frame_dets, key=lambda x: -x.conf):
            best_iou = 0.0
            best_tr = None
            for tr in active:
                if tr.track_id in used:
                    continue
                iou = aabb_iou(tr.dets[-1].aabb_norm, d.aabb_norm)
                if iou > best_iou:
                    best_iou = iou
                    best_tr = tr

            if best_tr is not None and best_iou >= IOU_THRESH:
                best_tr.add(d)
                used.add(best_tr.track_id)
            else:
                tr = Track(track_id=f"T{next_id}")
                next_id += 1
                tr.add(d)
                active.append(tr)
                used.add(tr.track_id)

    finished.extend(active)
    return finished


# =========================
# Track summarization
# =========================
def summarize_track(tr: Track) -> Dict[str, Any]:
    dets = sorted(tr.dets, key=lambda d: d.frame)
    frames = [d.frame for d in dets]
    if not frames:
        return {}

    centers = np.array([aabb_center(d.aabb_norm) for d in dets], dtype=np.float32)
    cx_std = float(np.std(centers[:, 0])) if len(dets) > 1 else 0.0
    cy_std = float(np.std(centers[:, 1])) if len(dets) > 1 else 0.0

    areas = []
    for d in dets:
        x1, y1, x2, y2 = d.aabb_norm
        areas.append(max(0.0, (x2-x1) * (y2-y1)))
    area_std = float(np.std(areas)) if len(areas) > 1 else 0.0

    confs = [d.conf for d in dets]
    conf_mean = float(np.mean(confs)) if confs else 0.0
    conf_p10 = float(np.percentile(confs, 10)) if len(confs) >= 2 else (confs[0] if confs else 0.0)

    # segments
    segments = []
    seg_start = dets[0].frame
    seg_text = dets[0].text
    seg_confs = [dets[0].conf]

    for d in dets[1:]:
        sim = text_similarity(seg_text, d.text)
        if sim >= TEXT_SIM_THRESH:
            seg_confs.append(d.conf)
        else:
            segments.append({
                "start_frame": int(seg_start),
                "end_frame": int(d.frame - 1),
                "text": seg_text,
                "conf_mean": float(np.mean(seg_confs)),
            })
            seg_start = d.frame
            seg_text = d.text
            seg_confs = [d.conf]

    segments.append({
        "start_frame": int(seg_start),
        "end_frame": int(dets[-1].frame),
        "text": seg_text,
        "conf_mean": float(np.mean(seg_confs)),
    })

    # representative bbox
    aabbs = np.array([d.aabb_norm for d in dets], dtype=np.float32)
    rep = np.median(aabbs, axis=0).tolist()

    return {
        "track_id": tr.track_id,
        "framespan": {"start": int(frames[0]), "end": int(frames[-1]), "num_dets": len(dets)},
        "bbox_norm": {"x1": rep[0], "y1": rep[1], "x2": rep[2], "y2": rep[3]},
        "confidence": {"mean": conf_mean, "p10": conf_p10},
        "stability": {
            "bbox_center_std": {"x": cx_std, "y": cy_std},
            "bbox_area_std": area_std,
            "text_num_changes": max(0, len(segments) - 1),
            "segment_count": len(segments),
        },
        "text_segments": segments,
        "samples": [
            {"frame": int(dets[i].frame), "text": dets[i].text, "conf": float(dets[i].conf)}
            for i in np.linspace(0, len(dets)-1, num=min(5, len(dets))).astype(int)
        ],
    }


def summarize_tracks(tracks: List[Track]) -> List[Dict[str, Any]]:
    summaries = []
    for tr in tracks:
        if len(tr.dets) >= MIN_TRACK_FRAMES:
            summaries.append(summarize_track(tr))

    def importance(s: Dict[str, Any]) -> float:
        span = s["framespan"]["end"] - s["framespan"]["start"] + 1
        conf = s["confidence"]["mean"]
        b = s["bbox_norm"]
        area = max(0.0, (b["x2"]-b["x1"]) * (b["y2"]-b["y1"]))
        return span * (0.5 + conf) * (0.5 + area)

    summaries.sort(key=importance, reverse=True)
    return summaries[:MAX_TRACKS_TO_SEND]


# =========================
# Gemini judge
# =========================
def gemini_judge(prompt_text: str, payload: Dict[str, Any]) -> Dict[str, Any]:
    model = _get_gemini_model()

    user_text = GEMINI_USER_TEMPLATE.format(
        prompt_text=prompt_text,
        tracks_json=json.dumps(payload, ensure_ascii=False),
    )
    resp = model.generate_content(user_text, request_options={"timeout": TIMEOUT_S})
    data = json.loads(_extract_json(resp.text))

    # ---- NEW: normalize added fields ----
    def to_bool(x, default=False):
        if isinstance(x, bool):
            return x
        if isinstance(x, str):
            if x.lower() in ("true", "1", "yes"):
                return True
            if x.lower() in ("false", "0", "no"):
                return False
        return default

    allowed_presence = {"none", "noise", "incidental", "required"}
    text_presence = data.get("text_presence", "none")
    if isinstance(text_presence, str):
        text_presence = text_presence.strip().lower()
    if text_presence not in allowed_presence:
        text_presence = "none"

    data["prompt_requires_visible_text"] = to_bool(data.get("prompt_requires_visible_text", False), default=False)
    data["text_presence"] = text_presence
    data["missing_required_text"] = to_bool(data.get("missing_required_text", False), default=False)

    itc = data.get("incidental_text_is_contextual", None)
    if itc is None:
        data["incidental_text_is_contextual"] = None
    else:
        data["incidental_text_is_contextual"] = to_bool(itc, default=False)

    # ---- existing score normalization ----
    data["overall_text_quality_score"] = clamp100(data.get("overall_text_quality_score", 0))
    subs = data.get("subscores", {}) or {}
    data["subscores"] = {
        "legibility_accuracy": clamp100(subs.get("legibility_accuracy", 0)),
        "temporal_stability": clamp100(subs.get("temporal_stability", 0)),
        "spatial_stability": clamp100(subs.get("spatial_stability", 0)),
        "completeness": clamp100(subs.get("completeness", 0)),
        "prompt_text_match": subs.get("prompt_text_match", None),
    }
    data["confidence"] = clamp01(data.get("confidence", 0.0))
    return data



# =========================
# Main pipeline (video + prompt only)
# =========================
def run(video_path: str, prompt_text: str, out_path: str, payload_path: Optional[str] = None) -> Dict[str, Any]:
    """
    Compatibility wrapper.
    If payload_path exists, load it; otherwise run OCR to produce it.
    Then call Gemini and save final result json to out_path.
    """
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)

    if payload_path and os.path.exists(payload_path):
        payload = json.loads(open(payload_path, "r", encoding="utf-8").read())
    else:
        # fallback: run OCR now
        payload_path = payload_path or (out_path + ".payload.json")
        payload = run_ocr_only(video_path, payload_path)

    verdict_part = run_gemini_only(prompt_text, payload)

    result = {
        **verdict_part,
        "tracks_summary": payload,
        "_input": {"video_path": video_path, "prompt_text": prompt_text, "model": MODEL_NAME},
    }

    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)

    return result

import os
import re
import json
import csv
import time
import argparse
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple
def _gemini_worker(
    video_path: str,
    video_stem: str,
    prompt_text: str,
    payload_path: str,
    out_json_path: str,
) -> Dict[str, Any]:
    with open(payload_path, "r", encoding="utf-8") as f:
        payload = json.load(f)

    verdict = run_gemini_only(prompt_text, payload)

    result = {
        # ---- NEW classification fields ----
        "prompt_requires_visible_text": verdict.get("prompt_requires_visible_text", False),
        "text_presence": verdict.get("text_presence", "none"),
        "missing_required_text": verdict.get("missing_required_text", False),
        "incidental_text_is_contextual": verdict.get("incidental_text_is_contextual", None),

        # ---- scores & explanation ----
        "overall_text_quality_score": verdict.get("overall_text_quality_score", 0),
        "subscores": verdict.get("subscores", {}),
        "confidence": verdict.get("confidence", 0.0),
        "top_issues": verdict.get("top_issues", []),
        "good_tracks": verdict.get("good_tracks", []),
        "summary": verdict.get("summary", ""),

        # ---- provenance ----
        "tracks_summary": payload,
        "_input": {"video_path": video_path, "prompt_text": prompt_text, "model": MODEL_NAME},
    }

    with open(out_json_path, "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)

    subs = result.get("subscores") or {}
    return {
        "video_path": video_path,
        "video_stem": video_stem,
        "overall_text_quality_score": result.get("overall_text_quality_score", ""),
        "legibility_accuracy": subs.get("legibility_accuracy", ""),
        "temporal_stability": subs.get("temporal_stability", ""),
        "spatial_stability": subs.get("spatial_stability", ""),
        "completeness": subs.get("completeness", ""),
        "prompt_text_match": subs.get("prompt_text_match", ""),
        "confidence": result.get("confidence", ""),
    }

import threading
_THREAD_LOCAL = threading.local()


def _get_gemini_api_key() -> str:
    api_key = os.getenv("GEMINI_API_KEY", "").strip()
    if not api_key:
        api_key = os.getenv("GOOGLE_API_KEY", "").strip()
    if not api_key:
        raise RuntimeError("Missing GEMINI_API_KEY or GOOGLE_API_KEY")
    return api_key

def _get_gemini_model():
    m = getattr(_THREAD_LOCAL, "model", None)
    if m is None:
        genai.configure(api_key=_get_gemini_api_key())
        m = genai.GenerativeModel(model_name=MODEL_NAME, system_instruction=GEMINI_SYSTEM)
        _THREAD_LOCAL.model = m
    return m


# =========================
# 2) Filename matching helpers
# =========================
def safe_filename(name: str, max_len: int = 180) -> str:
    name = str(name).strip()
    name = re.sub(r"[\/\\\:\*\?\"\<\>\|\n\r\t]", "_", name)
    name = re.sub(r"\s+", " ", name).strip()
    if not name:
        name = "untitled"
    if len(name) > max_len:
        name = name[:max_len].rstrip()
    return name


def load_prompts(prompts_dir: Path, max_len: int = 180) -> Dict[str, Dict[str, str]]:
    """
    Return mapping: video_stem(safe_filename(content)) -> {content, prompt, source}
    """
    m: Dict[str, Dict[str, str]] = {}
    for jp in sorted(prompts_dir.glob("*.json")):
        data = json.loads(jp.read_text(encoding="utf-8"))
        for x in data:
            content = x["content"]
            prompt = x["prompt"]
            key = safe_filename(content, max_len=max_len)
            if key in m and m[key]["content"] != content:
                # collision warning
                print(f"[WARN] key collision for '{key}': '{m[key]['content']}' vs '{content}'")
                # keep the first one by default
                continue
            m[key] = {"content": content, "prompt": prompt, "source": jp.name}
    return m


def find_mp4s(root: Path) -> List[Path]:
    return sorted(root.rglob("*.mp4"))


def run_ocr_only(video_path: str, out_payload_path: str) -> Dict[str, Any]:
    """
    Run OCR+merge+track+summarize only.
    Save payload (tracks_summary) to out_payload_path.
    Return payload dict.
    """
    os.makedirs(os.path.dirname(out_payload_path) or ".", exist_ok=True)

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {video_path}")

    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0)
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)

    dets_all: List[Det] = []
    frame_idx = 0
    processed = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        if frame_idx < START_FRAME:
            frame_idx += 1
            continue
        if processed >= MAX_FRAMES:
            break
        if (frame_idx - START_FRAME) % STRIDE != 0:
            frame_idx += 1
            continue

        t = frame_idx / fps if fps > 0 else 0.0

        try:
            ocr_out = ocr_frame(frame)
        except Exception:
            frame_idx += 1
            processed += 1
            continue

        frame_dets: List[Det] = []
        for poly, text, conf in ocr_out:
            try:
                conf = float(conf)
                if conf < MIN_OCR_CONF:
                    continue
                poly = [[float(p[0]), float(p[1])] for p in poly]
                aabb = poly_to_aabb(poly)
                aabb_n = normalize_aabb(aabb, w, h)
                frame_dets.append(
                    Det(frame=frame_idx, t=t, text=str(text), conf=conf,
                        poly=poly, aabb=aabb, aabb_norm=aabb_n)
                )
            except Exception:
                continue

        frame_merged = merge_dets_in_frame(frame_dets)
        dets_all.extend(frame_merged)

        frame_idx += 1
        processed += 1

    cap.release()

    dets_all.sort(key=lambda d: (d.frame, -d.conf))
    tracks = build_tracks(dets_all)
    track_summaries = summarize_tracks(tracks)

    payload = {
        "video_meta": {"width": w, "height": h, "fps": fps, "total_frames": total},
        "pipeline_meta": {
            "sampling": {"stride": STRIDE, "max_frames": MAX_FRAMES, "start_frame": START_FRAME},
            "same_frame_merge": {
                "merge_y_iou_thresh": MERGE_Y_IOU_THRESH,
                "merge_x_gap_norm": MERGE_X_GAP_NORM,
                "merge_max_height_ratio": MERGE_MAX_HEIGHT_RATIO,
            },
            "tracking": {"iou_thresh": IOU_THRESH, "max_gap_frames": MAX_GAP_FRAMES},
            "segmentation": {"text_sim_thresh": TEXT_SIM_THRESH},
            "counts": {
                "merged_detections": len(dets_all),
                "tracks_total": len(tracks),
                "tracks_sent": len(track_summaries),
            },
        },
        "tracks": track_summaries,
    }

    with open(out_payload_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)

    return payload

def run_gemini_only(prompt_text: str, payload: Dict[str, Any]) -> Dict[str, Any]:
    return gemini_judge(prompt_text, payload)



# =========================
# 3) Batch eval
# =========================
def main():
    import os
    import json
    import csv
    import time
    import argparse
    from pathlib import Path
    from typing import Any, Dict, List, Tuple
    from concurrent.futures import ThreadPoolExecutor, as_completed

    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True, help="root directory that contains mp4s (recursive)")
    ap.add_argument("--prompts_dir", required=True, help="directory containing *.json prompts")

    ap.add_argument("--out_dir", default="outputs_text_quality_batch", help="output directory")
    ap.add_argument("--save_csv", default="results_text_quality.csv",
                    help="per-video CSV path (relative to out_dir unless absolute)")
    ap.add_argument("--save_json_dir", default="per_video_json",
                    help="dir to save per-video detailed json (inside out_dir)")
    ap.add_argument("--payload_dir", default="payload_json",
                    help="dir to save per-video OCR payload json (inside out_dir)")

    ap.add_argument("--skip_existing", action="store_true",
                    help="skip gemini if final per-video json already exists (still may run OCR if payload missing)")
    ap.add_argument("--safe_max_len", type=int, default=200,
                    help="MUST match generation-time safe_filename max_len")

    ap.add_argument("--gemini_workers", type=int, default=int(os.getenv("GEMINI_WORKERS", "8")),
                    help="number of concurrent Gemini calls")
    ap.add_argument("--gemini_sleep", type=float, default=0.0,
                    help="optional sleep inside each Gemini worker (to reduce rate-limit risk)")

    ap.add_argument("--skip_ocr", action="store_true",
                    help="skip OCR stage; require payload json already exists")
    ap.add_argument("--skip_gemini", action="store_true",
                    help="skip Gemini stage (only generate payloads)")

    args = ap.parse_args()

    root = Path(args.root).expanduser().resolve()
    prompts_dir = Path(args.prompts_dir).expanduser().resolve()

    out_dir = Path(args.out_dir).expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    save_json_dir = out_dir / args.save_json_dir
    save_json_dir.mkdir(parents=True, exist_ok=True)

    payload_dir = out_dir / args.payload_dir
    payload_dir.mkdir(parents=True, exist_ok=True)

    csv_path = Path(args.save_csv)
    if not csv_path.is_absolute():
        csv_path = out_dir / csv_path
    csv_path.parent.mkdir(parents=True, exist_ok=True)

    # Load prompts and discover videos
    prompt_map = load_prompts(prompts_dir, max_len=args.safe_max_len)
    mp4s = find_mp4s(root)
    if not mp4s:
        print("No mp4 found.")
        return

    # Stage 0: build per-video plan
    missing_prompt = 0
    planned: List[Tuple[Path, Dict[str, str], Path, Path]] = []
    # tuple: (video_path, meta, payload_path, out_json_path)

    for vp in mp4s:
        video_stem = vp.stem
        meta = prompt_map.get(video_stem)
        if meta is None:
            missing_prompt += 1
            print(video_stem)
            continue

        payload_path = payload_dir / f"{video_stem}.payload.json"
        out_json = save_json_dir / f"{video_stem}.json"

        planned.append((vp, meta, payload_path, out_json))

    print(f"Found videos: {len(mp4s)} | with prompt: {len(planned)} | missing prompt: {missing_prompt}")

    # Stage 1: OCR (SERIAL)
    ocr_done = 0
    ocr_skipped = 0
    ocr_failed = 0

    if not args.skip_ocr:
        for vp, meta, payload_path, out_json in planned:
            if payload_path.exists():
                ocr_skipped += 1
                continue
            try:
                run_ocr_only(str(vp), str(payload_path))
                ocr_done += 1
            except Exception as e:
                ocr_failed += 1
                # still keep going; gemini stage will fail for this one unless you skip it
                print(f"[OCR FAIL] {vp} | {repr(e)}")
    else:
        # ensure payload exists if skipping OCR
        for vp, meta, payload_path, out_json in planned:
            if not payload_path.exists():
                ocr_failed += 1
                print(f"[OCR MISSING] payload not found but --skip_ocr set: {payload_path}")

    print(f"OCR stage: done={ocr_done}, skipped_existing={ocr_skipped}, failed/missing={ocr_failed}")

    if args.skip_gemini:
        print("Skip Gemini stage (--skip_gemini).")
        return



    # Stage 2: Gemini (CONCURRENT)
    rows: List[Dict[str, Any]] = []
    gemini_planned = 0
    gemini_skipped = 0
    gemini_failed = 0

    def _submit_eligible(ex):
        fut2meta = {}
        for vp, meta, payload_path, out_json in planned:
            # skip if final result exists
            if args.skip_existing and out_json.exists():
                gemini_skipped += 1  # outer scope? can't assign here cleanly
                continue
            if not payload_path.exists():
                continue

            fut = ex.submit(
                _gemini_worker,
                str(vp),
                vp.stem,
                meta["prompt"],
                str(payload_path),
                str(out_json),
                # if you updated _gemini_worker signature to accept sleep, pass it; otherwise ignore
            )
            fut2meta[fut] = (vp, meta)
        return fut2meta

    # Because Python scope rules: count skips in a separate pass
    eligible = []
    for vp, meta, payload_path, out_json in planned:
        if args.skip_existing and out_json.exists():
            gemini_skipped += 1
            continue
        if not payload_path.exists():
            # OCR failed/missing
            rows.append({
                "video_path": str(vp),
                "video_stem": vp.stem,
                "content": meta["content"],
                "prompt_source": meta["source"],
                "error": f"Missing payload: {payload_path}",
            })
            gemini_failed += 1
            continue
        eligible.append((vp, meta, payload_path, out_json))

    gemini_planned = len(eligible)
    print(f"Gemini stage: planned={gemini_planned}, skipped_existing={gemini_skipped}")

    with ThreadPoolExecutor(max_workers=args.gemini_workers) as ex:
        futs = []
        for vp, meta, payload_path, out_json in eligible:
            futs.append(ex.submit(
                _gemini_worker,
                str(vp),
                vp.stem,
                meta["prompt"],
                str(payload_path),
                str(out_json),
            ))

        for fut in as_completed(futs):
            try:
                row = fut.result()
                # attach prompt metadata for CSV
                # We need to find meta again: simplest is to use prompt_map with video_stem
                meta2 = prompt_map.get(row.get("video_stem", ""), {})
                row["content"] = meta2.get("content", "")
                row["prompt_source"] = meta2.get("source", "")
                rows.append(row)
            except Exception as e:
                gemini_failed += 1
                rows.append({"error": repr(e)})

            if args.gemini_sleep > 0:
                time.sleep(args.gemini_sleep)

    # Additionally, if you want to include rows for skipped_existing, you can load existing jsons:
    if args.skip_existing and gemini_skipped > 0:
        for vp, meta, payload_path, out_json in planned:
            if not out_json.exists():
                continue
            try:
                j = json.loads(out_json.read_text(encoding="utf-8"))
                subs = j.get("subscores") or {}
                rows.append({
                    "video_path": str(vp),
                    "video_stem": vp.stem,
                    "content": meta["content"],
                    "prompt_source": meta["source"],
                    "overall_text_quality_score": j.get("overall_text_quality_score", ""),
                    "legibility_accuracy": subs.get("legibility_accuracy", ""),
                    "temporal_stability": subs.get("temporal_stability", ""),
                    "spatial_stability": subs.get("spatial_stability", ""),
                    "completeness": subs.get("completeness", ""),
                    "prompt_text_match": subs.get("prompt_text_match", ""),
                    "confidence": j.get("confidence", ""),
                })
            except Exception:
                pass

    # write csv
    if rows:
        fieldnames = sorted({k for r in rows for k in r.keys()})
        with csv_path.open("w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=fieldnames)
            w.writeheader()
            w.writerows(rows)

    print(
        f"Done. videos={len(mp4s)}, with_prompt={len(planned)}, missing_prompt={missing_prompt}\n"
        f"OCR: done={ocr_done}, skipped_existing={ocr_skipped}, failed/missing={ocr_failed}\n"
        f"Gemini: planned={gemini_planned}, skipped_existing={gemini_skipped}, failed={gemini_failed}\n"
        f"Saved CSV: {csv_path}\n"
        f"Payload dir: {payload_dir}\n"
        f"Result json dir: {save_json_dir}"
    )


if __name__ == "__main__":
    main()
