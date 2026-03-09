
import os
import re
import json
import hashlib
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import librosa
from moviepy.editor import VideoFileClip

import google.generativeai as genai
from basic_pitch.inference import predict_and_save
import basic_pitch


# =========================
# Config
# =========================
MODEL_NAME = "gemini-3-flash-preview" 
DEFAULT_TIMEOUT_S = int(os.getenv("GEMINI_TIMEOUT_S", "180"))

TEMP_DIR = Path(os.getenv("MUSIC_TEMP_DIR", "./temp_music_analysis"))
TEMP_DIR.mkdir(parents=True, exist_ok=True)

MAX_EVENTS_TO_SEND = int(os.getenv("MUSIC_MAX_EVENTS_TO_SEND", "600"))
CHORD_ONSET_WINDOW_S = float(os.getenv("MUSIC_CHORD_ONSET_WINDOW_S", "0.08"))  # group notes within 80ms as a chord frame


def _get_gemini_api_key() -> str:
    api_key = os.getenv("GEMINI_API_KEY", "").strip()
    if not api_key:
        api_key = os.getenv("GOOGLE_API_KEY", "").strip()
    if not api_key:
        raise RuntimeError("Missing GEMINI_API_KEY env var. Please: export GEMINI_API_KEY='...'")
    return api_key


# =========================
# Gemini prompts (2-stage)
# =========================
PROMPT_TO_CONSTRAINTS_SYSTEM = """
You are a strict benchmark judge for music-generation compliance.

Your first job is to decide whether MIDI-based evaluation is appropriate at all.

MIDI-based evaluation is appropriate ONLY when the prompt contains explicit, testable musical note/chord/scale constraints
(e.g., named notes like C4, pitch classes like C major, chord names like Am7, scale degrees/solfege, short note sequences),
AND the described sound source is reasonably "midi-transcribable" with Basic Pitch (monophonic/clear pitched instruments).

Do NOT use MIDI evaluation when:
- The prompt contains no explicit notes/chords/scales (e.g., "nice music", "beautiful melody", "upbeat song", "jazzy vibe").
- The prompt is mainly about rhythm/genre/tempo/mood without pitch constraints.
- The prompt is mainly about instruments that are hard to reliably transcribe to MIDI here, especially: bowed strings (violin/viola/cello/double bass), vocal singing/choir, or complex ensembles/orchestras.
- The prompt asks for sound design / SFX / drums-only / noise / ambience.

If MIDI evaluation is not appropriate, output should_midi_eval=false with a short reason, and keep constraints minimal (task_type="unknown").

If MIDI evaluation IS appropriate, extract ONLY testable constraints that can be verified from:
- A raw list of detected MIDI note events: (time_seconds, duration_seconds, note_name_with_octave, midi_number)
- Derived chord frames built by grouping near-simultaneous note onsets

Important:
- The transcription may contain noise/extra notes. Be robust: allow tolerance unless the prompt demands strictness.
- Treat octave as strict ONLY if the prompt explicitly mentions an octave (e.g., C4) or says "one octave starting at ...".

Output STRICT JSON only. No markdown, no code fences, no extra text.
""".strip()


PROMPT_TO_CONSTRAINTS_USER = """
From PROMPT_TEXT, output STRICT JSON:

{
  "prompt_text": "...",

  "should_midi_eval": true,
  "skip_reason": null,

  "instrument_family": "keyboard|guitar|bass|wind|brass|mallet|synth|drums|voice|bowed_strings|ensemble|unknown",
  "task_type": "single_chord|chord_progression|note_sequence|scale|mixed|unknown",
  "octave_strict": false,

  "constraints": {
    "expected_chord": null,
    "expected_chord_quality": null,
    "expected_pitch_classes_in_chord": [],
    "expected_scale": null,
    "expected_pitch_class_sequence": [],
    "expected_note_sequence": [],

    "tolerance": {
      "allow_extra_notes": true,
      "allow_inversions": true,
      "match_by_pitch_class": true,
      "required_pc_coverage": 1.0,
      "max_extra_pcs_in_chord": 3
    }
  }
}

Decision rules (IMPORTANT):
1) Set should_midi_eval=false if PROMPT_TEXT lacks explicit note/chord/scale constraints.
2) Set should_midi_eval=false if instrument_family is voice, bowed_strings, or ensemble (strings/orchestra/choir etc.).
3) If should_midi_eval=false:
   - set skip_reason to a short string (e.g., "no explicit notes/chords/scales" or "bowed strings not evaluated")
   - set task_type="unknown"
   - keep constraints minimal (mostly null/empty)
4) Only extract what is explicitly required by the prompt.
5) Do NOT require exact voicing/octave unless specified.

PROMPT_TEXT:
""".strip()


RAW_EVENTS_VALIDATE_SYSTEM = """
You are a strict, evidence-based music judge.

Inputs:
- Original prompt text
- Extracted constraints JSON
- RAW MIDI note events (may include noise)
- Derived chord frames (grouped by onset proximity)

Task:
Determine if the music content matches the prompt (e.g., C major chord, 5-note ascending scale).

Scoring (IMPORTANT):
- overall_score is an INTEGER from 0 to 100 (full score = 100).
- 100 means all applicable extracted constraints are satisfied with strong evidence.
- 0 means clearly not satisfied or no usable evidence.
- If constraints are partially satisfied, assign an intermediate score proportional to compliance.
- If the provided constraints_json indicates skipping (e.g., should_midi_eval=false or skipped=true), still output JSON and set:
  overall_score = 0, confidence = 0, and add a check explaining it was skipped.

You MUST:
1) Use chord frames to judge chord claims (major/minor/etc.) primarily by pitch class set, allowing inversions unless disallowed.
2) Use raw events to judge sequences/scales, but be robust to noise: focus on dominant/most plausible structure.
3) Produce checks and violations with evidence grounded in the provided data.
4) Output overall_score (0-100, full score=100) and confidence (0-1).

Rules:
- Use ONLY the provided data as evidence.
- If multiple plausible interpretations exist due to noise, choose the best-supported one and lower confidence.
- Output STRICT JSON only. No markdown, no code fences, no extra text.
""".strip()


RAW_EVENTS_VALIDATE_USER_TEMPLATE = """
PROMPT_TEXT:
{prompt_text}

CONSTRAINTS_JSON:
{constraints_json}

RAW_NOTE_EVENTS_JSON (truncated):
{raw_events_json}

CHORD_FRAMES_JSON:
{chord_frames_json}

Output STRICT JSON:
{{
  "overall_score": 0,
  "confidence": 0.0,
  "subscores": {{
    "chord": null,
    "sequence_or_scale": null
  }},
  "interpretation": {{
    "best_chord_frame": null,
    "best_chord_pitch_classes": [],
    "best_sequence_pitch_classes": [],
    "notes_used_as_evidence": []
  }},
  "checks": [
    {{
      "id": "C1",
      "aspect": "chord|scale|sequence|other",
      "status": "match|mismatch|missing|uncertain",
      "severity": 1,
      "evidence": "cite specific event times/notes or chord frame pcs"
    }}
  ],
  "violations": [
    {{
      "type": "wrong_chord|missing_chord_tones|too_many_extra_tones|wrong_scale|wrong_direction|wrong_order|missing_notes|noisy_transcription|other",
      "severity": 1,
      "evidence": "grounded in RAW_NOTE_EVENTS_JSON / CHORD_FRAMES_JSON"
    }}
  ],
  "summary": "short assessment"
}}
""".strip()


# =========================
# Utilities
# =========================
PITCH_CLASS_NAMES_SHARP = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]

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

def _stable_id(s: str) -> str:
    return hashlib.sha256(s.strip().encode("utf-8")).hexdigest()[:16]

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

def midi_note_to_name(midi_num: int) -> str:
    pc = midi_num % 12
    octave = (midi_num // 12) - 1  # 60 = C4
    return f"{PITCH_CLASS_NAMES_SHARP[pc]}{octave}"

def pitch_class_name(midi_num: int) -> str:
    return PITCH_CLASS_NAMES_SHARP[midi_num % 12]


# =========================
# Basic Pitch transcription
# =========================
class BasicPitchTranscriber:
    def __init__(self, temp_dir: Path = TEMP_DIR):
        self.temp_dir = Path(temp_dir)
        self.temp_dir.mkdir(parents=True, exist_ok=True)

        base_path = Path(basic_pitch.__file__).parent
        onnx_path = base_path / "saved_models/icassp_2022/nmp.onnx"
        self.model_path = str(onnx_path) if onnx_path.exists() else basic_pitch.ICASSP_2022_MODEL_PATH

    def extract_audio(self, video_path: str) -> str:
        audio_path = self.temp_dir / (Path(video_path).stem + ".wav")
        if audio_path.exists():
            try: audio_path.unlink()
            except Exception: pass

        clip = VideoFileClip(str(video_path))
        if clip.audio is None:
            clip.close()
            raise RuntimeError("Video has no audio track.")
        clip.audio.write_audiofile(str(audio_path), logger=None)
        clip.close()
        return str(audio_path)

    def audio_to_midi(self, audio_path: str) -> str:
        base_name = Path(audio_path).stem
        midi_path = self.temp_dir / f"{base_name}_basic_pitch.mid"
        if midi_path.exists():
            try: midi_path.unlink()
            except Exception: pass

        predict_and_save(
            audio_path_list=[str(audio_path)],
            output_directory=str(self.temp_dir),
            save_midi=True,
            sonify_midi=False,
            save_model_outputs=False,
            save_notes=False,
            model_or_model_path=self.model_path,
            onset_threshold=0.6,
            frame_threshold=0.3,
            minimum_note_length=50,  # ms
        )
        if not midi_path.exists():
            raise RuntimeError("Basic Pitch did not produce a MIDI file.")
        return str(midi_path)


# =========================
# MIDI parse (mido) -> raw events
# =========================
def parse_midi_note_events(midi_path: str) -> List[Dict[str, Any]]:
    try:
        import mido
    except ImportError as e:
        raise ImportError("Please install mido: pip install mido") from e

    mid = mido.MidiFile(midi_path)
    ticks_per_beat = mid.ticks_per_beat

    tempo_us_per_beat = 500000  # default 120bpm
    for tr in mid.tracks:
        for msg in tr:
            if msg.type == "set_tempo":
                tempo_us_per_beat = msg.tempo
                break

    def ticks_to_seconds(ticks: int) -> float:
        return mido.tick2second(ticks, ticks_per_beat, tempo_us_per_beat)

    ongoing = {}  # (track, midi_note) -> (start_s, velocity)
    evts: List[Dict[str, Any]] = []

    for ti, track in enumerate(mid.tracks):
        t_ticks = 0
        for msg in track:
            t_ticks += msg.time
            t_s = ticks_to_seconds(t_ticks)

            if msg.type == "note_on" and msg.velocity > 0:
                ongoing[(ti, msg.note)] = (t_s, msg.velocity)

            elif msg.type == "note_off" or (msg.type == "note_on" and msg.velocity == 0):
                key = (ti, msg.note)
                if key in ongoing:
                    start_s, vel = ongoing.pop(key)
                    dur_s = max(0.0, t_s - start_s)
                    evts.append({
                        "time": round(float(start_s), 4),
                        "dur": round(float(dur_s), 4),
                        "midi": int(msg.note),
                        "note": midi_note_to_name(int(msg.note)),
                        "pc": pitch_class_name(int(msg.note)),
                        "velocity": int(vel),
                        "track": int(ti),
                    })

    evts.sort(key=lambda e: (e["time"], -e["dur"], -e.get("velocity", 0)))
    return evts


# =========================
# Chord frames from raw events (group by onset proximity)
# =========================
def build_chord_frames(events: List[Dict[str, Any]], onset_window_s: float = CHORD_ONSET_WINDOW_S) -> List[Dict[str, Any]]:
    """
    Group notes with onsets within onset_window_s into a "frame".
    This is NOT filtering notes away; it's just a derived view to help chord judgement.
    """
    if not events:
        return []

    # sort by time
    evs = sorted(events, key=lambda e: e["time"])
    frames = []
    cur = {
        "start_time": evs[0]["time"],
        "events": [evs[0]],
    }

    for e in evs[1:]:
        if (e["time"] - cur["start_time"]) <= onset_window_s:
            cur["events"].append(e)
        else:
            frames.append(cur)
            cur = {"start_time": e["time"], "events": [e]}
    frames.append(cur)

    # summarize frames
    out = []
    for i, fr in enumerate(frames[:200]):  # cap
        pcs = sorted({ev["pc"] for ev in fr["events"]})
        notes = [ev["note"] for ev in fr["events"]]
        out.append({
            "frame_id": f"F{i+1}",
            "start_time": fr["start_time"],
            "pitch_classes": pcs,
            "notes": notes,
            "num_notes": len(fr["events"]),
            # include a few raw events for evidence
            "events": fr["events"][:12],
        })
    return out


# =========================
# Gemini stage 1 & 2
# =========================
def gemini_extract_constraints(prompt_text: str, timeout_s: int = DEFAULT_TIMEOUT_S) -> Dict[str, Any]:
    genai.configure(api_key=_get_gemini_api_key())
    model = genai.GenerativeModel(model_name=MODEL_NAME, system_instruction=PROMPT_TO_CONSTRAINTS_SYSTEM)
    resp = model.generate_content(PROMPT_TO_CONSTRAINTS_USER + prompt_text, request_options={"timeout": timeout_s})
    constraints_obj = json.loads(_extract_json(resp.text))

    if not constraints_obj.get("should_midi_eval", True):
        # Skip Basic Pitch / MIDI / stage-2 evaluation directly.
        return {
            "skipped": True,
            "skip_reason": constraints_obj.get("skip_reason"),
            "prompt_text": prompt_text,
            "constraints": constraints_obj,
        }

    return json.loads(_extract_json(resp.text))


def gemini_validate(prompt_text: str, constraints: Dict[str, Any],
                    raw_events: List[Dict[str, Any]], chord_frames: List[Dict[str, Any]],
                    timeout_s: int = DEFAULT_TIMEOUT_S) -> Dict[str, Any]:
    genai.configure(api_key=_get_gemini_api_key())
    model = genai.GenerativeModel(model_name=MODEL_NAME, system_instruction=RAW_EVENTS_VALIDATE_SYSTEM)

    user_text = RAW_EVENTS_VALIDATE_USER_TEMPLATE.format(
        prompt_text=prompt_text,
        constraints_json=json.dumps(constraints, ensure_ascii=False),
        raw_events_json=json.dumps(raw_events[:MAX_EVENTS_TO_SEND], ensure_ascii=False),
        chord_frames_json=json.dumps(chord_frames, ensure_ascii=False),
    )
    resp = model.generate_content(user_text, request_options={"timeout": timeout_s})
    data = json.loads(_extract_json(resp.text))
    data["overall_score"] = _clamp_int_0_100(data.get("overall_score", 0))
    data["confidence"] = _clamp_float_0_1(data.get("confidence", 0.0))
    return data


# =========================
# Orchestration
# =========================
def _constraints_cache_path(cache_dir: str, prompt_id: str) -> Path:
    d = Path(cache_dir)
    d.mkdir(parents=True, exist_ok=True)
    return d / f"{prompt_id}.json"

def _load_constraints_from_file(p: Path) -> Optional[Dict[str, Any]]:
    try:
        if p.exists():
            return json.loads(p.read_text(encoding="utf-8"))
    except Exception:
        # Ignore corrupted cache and regenerate.
        return None
    return None

def _atomic_write_json(path: Path, obj: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + f".tmp.{os.getpid()}")
    tmp.write_text(json.dumps(obj, ensure_ascii=False, indent=2), encoding="utf-8")
    os.replace(tmp, path)  # atomic on same filesystem


def judge_video_music_raw_midi(
    video_path: str,
    prompt_text: str,
    out_path: str = "outputs/music_raw_midi_result.json",
    constraints_cache_dir: str = "cache/music_prompt_constraints",
    timeout_s: int = DEFAULT_TIMEOUT_S,
) -> Dict[str, Any]:
    """
    Evaluate a single video against a prompt using:
    - Stage 1 (Gemini): decide if MIDI eval is appropriate + extract testable constraints
    - If appropriate: BasicPitch -> MIDI -> raw events + chord frames -> Stage 2 (Gemini) scoring
    - If not appropriate: output a unified JSON result with skipped=true and do NOT run transcription.

    Output schema is unified regardless of skip.
    Cache is per-prompt_id file, parallel-safe with atomic replace.
    """
    def _constraints_cache_path(cache_dir: str, prompt_id: str) -> Path:
        d = Path(cache_dir)
        d.mkdir(parents=True, exist_ok=True)
        return d / f"{prompt_id}.json"

    def _load_constraints_from_file(p: Path) -> Optional[Dict[str, Any]]:
        try:
            if p.exists():
                return json.loads(p.read_text(encoding="utf-8"))
        except Exception:
            # corrupted cache -> ignore
            return None
        return None

    def _atomic_write_json(path: Path, obj: Dict[str, Any]) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(path.suffix + f".tmp.{os.getpid()}")
        tmp.write_text(json.dumps(obj, ensure_ascii=False, indent=2), encoding="utf-8")
        os.replace(tmp, path)  # atomic on same filesystem

    def _normalize_stage1(obj: Any) -> Dict[str, Any]:
        """
        Normalize cached/stage1 object into the raw stage1 JSON:
        {
          "should_midi_eval": bool,
          "skip_reason": str|None,
          "constraints": {...},
          ...
        }
        Your gemini_extract_constraints may return either:
        - raw stage1 json (normal)
        - {"skipped": True, "constraints": <raw stage1 json>, ...}
        """
        if not isinstance(obj, dict):
            return {}
        if obj.get("skipped") is True and isinstance(obj.get("constraints"), dict):
            return obj["constraints"]
        return obj

    # ensure output dir exists
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    os.makedirs("cache", exist_ok=True)

    prompt_id = _stable_id(prompt_text)

    # -------- per-prompt cache (parallel-safe) --------
    cache_file = _constraints_cache_path(constraints_cache_dir, prompt_id)
    cached = _load_constraints_from_file(cache_file)
    if cached is None:
        cached = gemini_extract_constraints(prompt_text, timeout_s=timeout_s)
        _atomic_write_json(cache_file, cached)

    stage1 = _normalize_stage1(cached)
    should_midi_eval = bool(stage1.get("should_midi_eval", True))
    skip_reason = stage1.get("skip_reason")

    # constraints block (the part we want to expose consistently)
    constraints_block = stage1.get("constraints", {}) if isinstance(stage1.get("constraints", {}), dict) else {}

    # -------- early skip: do NOT run BasicPitch / MIDI / stage2 --------
    if not should_midi_eval:
        result = {
            "overall_score": 0,
            "confidence": 0.0,
            "skipped": True,
            "skip_reason": skip_reason or "should_midi_eval=false",
            "constraints": constraints_block,

            "checks": [
                {
                    "id": "SKIP",
                    "aspect": "other",
                    "status": "missing",
                    "severity": 1,
                    "evidence": f"Skipped MIDI evaluation: {skip_reason or 'not suitable per stage1 rules'}",
                }
            ],
            "violations": [],
            "subscores": {
                "chord": None,
                "sequence_or_scale": None
            },
            "interpretation": {
                "best_chord_frame": None,
                "best_chord_pitch_classes": [],
                "best_sequence_pitch_classes": [],
                "notes_used_as_evidence": []
            },
            "summary": "skipped (prompt not suitable for MIDI evaluation)",

            "_artifacts": {
                "audio_path": None,
                "midi_path": None,
                "raw_events_count": 0,
                "chord_frames_count": 0,
                "chord_onset_window_s": CHORD_ONSET_WINDOW_S,
                "max_events_sent": MAX_EVENTS_TO_SEND,
            },
            "_input": {
                "video_path": video_path,
                "prompt_text": prompt_text,
                "prompt_id": prompt_id,
                "model": MODEL_NAME,
            }
        }
        Path(out_path).write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
        return result

    # -------- transcription + validation --------
    transcriber = BasicPitchTranscriber(temp_dir=TEMP_DIR)
    audio_path = transcriber.extract_audio(video_path)
    midi_path = transcriber.audio_to_midi(audio_path)

    raw_events = parse_midi_note_events(midi_path)
    chord_frames = build_chord_frames(raw_events, onset_window_s=CHORD_ONSET_WINDOW_S)

    # IMPORTANT: pass raw stage1 JSON to validator (not wrapped cache object)
    verdict = gemini_validate(
        prompt_text=prompt_text,
        constraints=stage1,
        raw_events=raw_events,
        chord_frames=chord_frames,
        timeout_s=timeout_s,
    )

    result = {
        "overall_score": verdict.get("overall_score", 0),
        "confidence": verdict.get("confidence", 0.0),
        "skipped": False,
        "skip_reason": None,
        "constraints": constraints_block,

        "checks": verdict.get("checks", []),
        "violations": verdict.get("violations", []),
        "subscores": verdict.get("subscores", {"chord": None, "sequence_or_scale": None}),
        "interpretation": verdict.get("interpretation", {
            "best_chord_frame": None,
            "best_chord_pitch_classes": [],
            "best_sequence_pitch_classes": [],
            "notes_used_as_evidence": []
        }),
        "summary": verdict.get("summary", ""),

        "_artifacts": {
            "audio_path": audio_path,
            "midi_path": midi_path,
            "raw_events_count": len(raw_events),
            "chord_frames_count": len(chord_frames),
            "chord_onset_window_s": CHORD_ONSET_WINDOW_S,
            "max_events_sent": MAX_EVENTS_TO_SEND,
        },
        "_input": {
            "video_path": video_path,
            "prompt_text": prompt_text,
            "prompt_id": prompt_id,
            "model": MODEL_NAME,
        }
    }

    Path(out_path).write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    return result




# =========================
# CLI
# =========================
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("video_path", type=str)
    parser.add_argument("--prompt", type=str, default=None, help="Prompt text (plain text).")
    parser.add_argument("--prompt-file", type=str, default=None, help="Path to a text file containing the prompt.")
    parser.add_argument("--out", type=str, default="outputs/music_raw_midi_result.json")
    args = parser.parse_args()

    if args.prompt_file:
        prompt_text = Path(args.prompt_file).read_text(encoding="utf-8").strip()
    elif args.prompt is not None:
        prompt_text = args.prompt.strip()
    else:
        raise SystemExit("Provide --prompt or --prompt-file")

    res = judge_video_music_raw_midi(
        video_path=args.video_path,
        prompt_text=prompt_text,
        out_path=args.out,
    )
    print(res["overall_score"])
