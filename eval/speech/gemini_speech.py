import os
import json
import sys
from pathlib import Path
from typing import Dict, Any, Optional

import torch
import whisper

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from gemini_client import generate_content_text, resolve_api_key


# =========================
# Config
# =========================
WHISPER_MODEL_NAME = os.getenv("WHISPER_MODEL_NAME", "large-v3")
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

GEMINI_MODEL_NAME = os.getenv("GEMINI_MODEL_NAME", "gemini-3-flash-preview")
GEMINI_API_KEY = resolve_api_key(required=False) or ""
PROMPT_VARIANT = os.getenv("SPEECH_PROMPT_VARIANT", "original").strip().lower()


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
SPEECH_EVALUATOR_TEMPLATE_ORIGINAL = """
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

  "score": 0,
  "pass": true/false,

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
  "score_rationale": "...",
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

SPEECH_EVALUATOR_TEMPLATE_V1 = """
You are a speech transcript compliance auditor.

## Goal
Given (A) the prompt used to generate an AI video and (B) the Whisper speech transcript,
judge how well the spoken content satisfies the prompt.

## You MUST determine from the prompt whether it explicitly requires specific speech
- If the prompt explicitly provides required dialogue/voiceover lines or demands exact wording,
  evaluate in VERBATIM mode: required line(s) must appear with identical wording (ignore minor punctuation/case/spacing only).
- Otherwise evaluate in CONTEXTUAL mode: speech only needs to be consistent with the prompt’s scene/message/constraints.
  If the prompt does not require speech and the transcript is empty or near-empty, that can still be compliant.

## Return STRICT JSON ONLY
No Markdown, no code fences.

Use this schema:
{{
  "explicit_speech_required": true/false,
  "match_type": "verbatim" | "contextual",
  "score": 0,
  "pass": true/false,
  "required_speech_lines": ["..."],
  "verbatim_match_details": [
    {{"required_line": "...", "found": true/false, "matched_text": "...", "diff_summary": "..."}}
  ],
  "missing_lines": ["..."],
  "extra_or_mismatched_segments": ["..."],
  "normalized_transcript": "...",
  "score_rationale": "...",
  "suggested_fix": "..."
}}

## Scoring rubric
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

## Normalization rule for normalized_transcript
- Trim and collapse whitespace; remove obvious non-speech tags like [music], [applause]; do NOT rewrite content.

## Inputs
PROMPT_USED_TO_GENERATE_VIDEO:
{generation_prompt}

WHISPER_TRANSCRIPT_RAW:
{transcript_text}
""".strip()

SPEECH_EVALUATOR_TEMPLATE_V2 = """
You are a speech transcript compliance auditor.

Goal: given (A) the prompt used to generate an AI video and (B) the Whisper speech transcript, judge how well the spoken content satisfies the prompt.

You must determine from the prompt whether it explicitly requires specific speech. If the prompt explicitly provides required dialogue/voiceover lines or demands exact wording, evaluate in VERBATIM mode: required line(s) must appear with identical wording, ignoring minor punctuation/case/spacing only. Otherwise evaluate in CONTEXTUAL mode: speech only needs to be consistent with the prompt’s scene/message/constraints. If the prompt does not require speech and the transcript is empty or near-empty, that can still be compliant.

Return STRICT JSON ONLY with no Markdown and no code fences, using this schema:
{{
  "explicit_speech_required": true/false,
  "match_type": "verbatim" | "contextual",
  "score": 0,
  "pass": true/false,
  "required_speech_lines": ["..."],
  "verbatim_match_details": [
    {{"required_line": "...", "found": true/false, "matched_text": "...", "diff_summary": "..."}}
  ],
  "missing_lines": ["..."],
  "extra_or_mismatched_segments": ["..."],
  "normalized_transcript": "...",
  "score_rationale": "...",
  "suggested_fix": "..."
}}

Scoring rubric: in VERBATIM mode, 100 means all required lines are present and verbatim with only punctuation/case/spacing differences allowed, 80-99 means tiny formatting differences only with no word substitutions/additions/deletions, 1-79 means any missing required line or any wording change such as paraphrase/synonym/added/removed word, and 0 means the transcript is empty but speech is explicitly required. In CONTEXTUAL mode, 90-100 means the speech clearly aligns with prompt intent and constraints, 70-89 means mostly aligned but with minor omissions or weak alignment, 1-69 means significant mismatch, off-topic content, or contradictions, and 0 means completely contradictory content or disallowed claims that break core requirements.

Normalization rule for normalized_transcript: trim and collapse whitespace; remove obvious non-speech tags like [music], [applause]; do NOT rewrite content.

Inputs:
PROMPT_USED_TO_GENERATE_VIDEO:
{generation_prompt}

WHISPER_TRANSCRIPT_RAW:
{transcript_text}
""".strip()

SPEECH_PROMPT_TEMPLATES = {
    "original": SPEECH_EVALUATOR_TEMPLATE_ORIGINAL,
    "v1": SPEECH_EVALUATOR_TEMPLATE_V1,
    "v2": SPEECH_EVALUATOR_TEMPLATE_V2,
}

if PROMPT_VARIANT not in SPEECH_PROMPT_TEMPLATES:
    raise ValueError(f"Unsupported SPEECH_PROMPT_VARIANT: {PROMPT_VARIANT}")

def evaluate_speech_with_gemini(
    generation_prompt: str,
    transcript_text: str,
    model_name: str = GEMINI_MODEL_NAME,
) -> Dict[str, Any]:
    if not GEMINI_API_KEY:
        raise RuntimeError("Missing Gemini API key env var (for example GEMINI_API_KEY=...).")

    evaluator_prompt = SPEECH_PROMPT_TEMPLATES[PROMPT_VARIANT].format(
        generation_prompt=generation_prompt,
        transcript_text=transcript_text,
    )

    text = generate_content_text(
        model_name=model_name,
        user_parts=[evaluator_prompt],
        api_key=GEMINI_API_KEY,
        timeout_s=600,
    ).strip()

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
