import os
import json
from typing import Dict, Any, Optional

import torch
import whisper
import google.generativeai as genai


# =========================
# Config
# =========================
WHISPER_MODEL_NAME = os.getenv("WHISPER_MODEL_NAME", "large-v3")
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

GEMINI_MODEL_NAME = os.getenv("GEMINI_MODEL_NAME", "gemini-3-flash-preview")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "")


# =========================
# Whisper
# =========================
def transcribe_with_whisper(
    video_path: str,
    model_name: str = WHISPER_MODEL_NAME,
    device: str = DEVICE,
    beam_size: int = 5,
    fp16: Optional[bool] = None,
) -> Dict[str, Any]:
    if not os.path.exists(video_path):
        raise FileNotFoundError(f"Video not found: {video_path}")

    if fp16 is None:
        fp16 = (device == "cuda")

    model = whisper.load_model(model_name, device=device)
    return model.transcribe(
        video_path,
        beam_size=beam_size,
        fp16=fp16,
        verbose=False,
    )


# =========================
# Gemini: score 0-100
# =========================
def evaluate_speech_with_gemini(
    generation_prompt: str,
    transcript_text: str,
    model_name: str = GEMINI_MODEL_NAME,
) -> Dict[str, Any]:
    if not GEMINI_API_KEY:
        raise RuntimeError("Missing GEMINI_API_KEY env var (export GEMINI_API_KEY=...).")

    genai.configure(api_key=GEMINI_API_KEY)
    model = genai.GenerativeModel(model_name=model_name)

    evaluator_prompt = f"""
You are a speech transcript compliance auditor.

Goal:
Given (A) the prompt used to generate an AI video and (B) the Whisper speech transcript,
judge how well the spoken content satisfies the prompt.

You MUST determine from the prompt whether it explicitly requires specific speech:
- If the prompt explicitly provides required dialogue/voiceover lines or demands exact wording,
  evaluate in VERBATIM mode: required line(s) must appear with identical wording (ignore minor punctuation/case/spacing only).
- Otherwise evaluate in CONTEXTUAL mode: speech only needs to be consistent with the prompt’s scene/message/constraints.
  If the prompt does not require speech and the transcript is empty or near-empty, that can still be compliant.

Return STRICT JSON ONLY (no Markdown, no code fences) using this schema:
{{
  "explicit_speech_required": true/false,
  "match_type": "verbatim" | "contextual",

  "score": 0,                 // integer 0-100
  "pass": true/false,         // pass iff score >= 80

  "required_speech_lines": ["..."],
  "verbatim_match_details": [
    {{
      "required_line": "...",
      "found": true/false,
      "matched_text": "...",
      "diff_summary": "..."
    }}
  ],

  "missing_lines": ["..."],
  "extra_or_mismatched_segments": ["..."],

  "normalized_transcript": "...",
  "score_rationale": "...",    // brief, bullet-like sentences OK
  "suggested_fix": "..."
}}

Scoring rubric:
- VERBATIM mode:
  * 100: all required lines present and verbatim (allow only punctuation/case/spacing differences)
  * 80-99: tiny formatting differences only; no word substitutions/additions/deletions
  * 1-79: any missing required line OR any wording change (paraphrase/synonym/added/removed word)
  * 0: transcript empty but speech is explicitly required
- CONTEXTUAL mode:
  * 90-100: clearly aligns with prompt intent and constraints
  * 70-89: mostly aligned but minor omissions/weak alignment
  * 1-69: significant mismatch/off-topic/contradictions
  * 0: completely contradictory or contains disallowed claims that break core requirements

Normalization rule for normalized_transcript:
- Trim and collapse whitespace; remove obvious non-speech tags like [music], [applause]; do NOT rewrite content.

Inputs:
PROMPT_USED_TO_GENERATE_VIDEO:
{generation_prompt}

WHISPER_TRANSCRIPT_RAW:
{transcript_text}
""".strip()

    resp = model.generate_content(evaluator_prompt, request_options={"timeout": 600})
    text = (resp.text or "").strip()

    # tolerate ```json wrappers
    if text.startswith("```"):
        text = text.split("\n", 1)[-1]
        if text.endswith("```"):
            text = text[:-3]
        text = text.strip()

    data = json.loads(text)

    # Optional hardening: ensure int 0-100 and pass rule consistent
    try:
        score = int(data.get("score", 0))
    except Exception:
        score = 0
    score = max(0, min(100, score))
    data["score"] = score
    data["pass"] = bool(score >= 80)

    return data


# =========================
# End-to-end
# =========================
def check_video_speech_against_prompt(
    video_path: str,
    generation_prompt: str,
    whisper_model_name: str = WHISPER_MODEL_NAME,
    gemini_model_name: str = GEMINI_MODEL_NAME,
) -> Dict[str, Any]:
    whisper_result = transcribe_with_whisper(
        video_path=video_path,
        model_name=whisper_model_name,
        device=DEVICE,
        beam_size=5,
    )
    transcript = (whisper_result.get("text") or "").strip()

    judgement = evaluate_speech_with_gemini(
        generation_prompt=generation_prompt,
        transcript_text=transcript,
        model_name=gemini_model_name,
    )

    return {
        "video_path": video_path,
        "device": DEVICE,
        "whisper_model": whisper_model_name,
        "gemini_model": gemini_model_name,
        "transcript": transcript,
        "whisper_meta": {"language": whisper_result.get("language")},
        "judgement": judgement,
    }


if __name__ == "__main__":
    # export GEMINI_API_KEY="..."
    video_path = "/path/to/video_generation/ovi/news/Archaeology_ Ancient Discovery.mp4"

    generation_prompt = "A professional news segment starting with a female anchor in a bright studio, holding a paper and announcing a major historical find. Suddenly, the video cuts to shaky documentary-style footage inside a dusty cave, showing a gloved hand brushing dirt off a golden ancient artifact buried in the ground. The anchor's voiceover explains the significance of the discovery throughout the clip."

    result = check_video_speech_against_prompt(
        video_path=video_path,
        generation_prompt=generation_prompt,
        whisper_model_name="large-v3",
        gemini_model_name="gemini-3-flash-preview",
    )

    print(json.dumps(result, ensure_ascii=False, indent=2))
