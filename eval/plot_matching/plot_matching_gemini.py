"""
Gemini judge for Plot Matching (text -> audio-video alignment)

Input:
- prompt_text (plot / storyboard requirements in natural language)
- mp4 video file

Output:
- plot_alignment_score [0,100]
- event_log (story beats)
- checks (plot constraints matched/mismatched)
- violations (counts/colors/cuts/audio mismatch etc.)
- confidence
- (optional) visual_physics violations can be added, but this script focuses on plot matching.

Install:
  no special Gemini SDK required; requests use the official Gemini REST API.

Env:
  export GEMINI_API_KEY="..."
  export GEMINI_MODEL="gemini-3-pro"   # optional

Run:
  python plot_matching_judge.py /path/to/video.mp4 --prompt-file prompt.txt
  python plot_matching_judge.py /path/to/video.mp4 --prompt "..."



python plot_matching_gemini.py "/path/to/video_generation/sora2_generated/ads/TATTOO _ Platform Advertisers (Anonymized).mp4" --prompt "Two men sit at a diner counter with soft ambient clatter, and one asks if the other is selling the boat. A clip shows a man sailing a boat as a voice asks, 'Are you selling my boat?' Back at the diner, the man asks how he can keep the memory of the boat. The video cuts to him screaming in a tattoo parlor as a buzzing tattoo machine etches a small sailboat onto his arm, the buzz rising and falling as the needle touches skin."
"""

import os
import time
import json
import re
import hashlib
import sys
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from gemini_client import generate_content_text, inline_file_part, resolve_api_key


# ========= Config =========
MODEL_NAME = os.getenv("PLOT_MATCHING_GEMINI_MODEL", "gemini-3-flash-preview")
TIMEOUT_S = int(os.getenv("GEMINI_TIMEOUT_S", "900"))
PROMPT_VARIANT = os.getenv("PLOT_MATCHING_PROMPT_VARIANT", "original").strip().lower()
# =========================


# ========= Prompts =========
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

PLOT_JUDGE_SYSTEM_V1 = """
You are a strict, evidence-based judge for a text-to-audio-video benchmark.

## Goal
Evaluate how well the provided MP4 matches the given PROMPT in terms of plot/story alignment,
including:
- story beats / event sequence (plot development)
- shot changes / continuity / pacing (cuts)
- objects and attributes (counts, colors, identities)
- actions and causal relations
- audio alignment (dialogue/sfx/music/ambience) and whether audio matches what is shown

## Rules
- Use ONLY observable evidence from the video and audio.
- If something is not clearly observable (low resolution, fast motion, occlusion, darkness), mark UNCERTAIN.
- Do not invent details not supported by evidence.
- If the prompt is underspecified, do not penalize missing details; focus on core requirements.
- Output STRICT JSON only. No markdown. No code fences. No extra text.

## Scoring
Return plot_alignment_score in [0,100].
Use subscores in [0,100]:
- narrative_alignment: plot beats and ordering
- shot_alignment: shot changes/cuts consistent with prompt (if specified) and continuity
- visual_attribute_alignment: object presence, counts, colors, key attributes
- audio_alignment: audio events that should occur vs actually heard; audio-visual consistency

Overall plot_alignment_score = round(0.35*narrative_alignment + 0.15*shot_alignment
                                    + 0.30*visual_attribute_alignment + 0.20*audio_alignment)

## Deduction guidance
- Severe mismatch of a core plot beat: -25
- Moderate mismatch: -12
- Minor mismatch/subtle: -5
- Missing core requirement: -10
- Uncertain: -0 (but reduce confidence)
""".strip()

PLOT_JUDGE_USER_TEMPLATE_V1 = """
## PROMPT (what the video should depict)
{prompt_text}

## Output STRICT JSON
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
      {{"id": "B1", "beat": "required story event", "strength": "strong|medium|weak"}}
    ],
    "required_visual_attributes": [
      {{
        "id": "V1",
        "object": "entity name",
        "attributes": {{"count": "integer or null", "color": "string or null", "other": ["..."]}},
        "strength": "strong|medium|weak"
      }}
    ],
    "required_audio_events": [
      {{"id": "A1", "audio_event": "required sound/dialogue/music cue", "strength": "strong|medium|weak"}}
    ],
    "editing_requirements": [
      {{"id": "E1", "requirement": "shot/cut/continuity requirement", "strength": "strong|medium|weak"}}
    ]
  }},
  "observed_event_log": [
    {{"where": "beginning|middle|end|event_k", "visual": "observable visual evidence", "audio": "observable audio evidence"}}
  ],
  "checks": [
    {{
      "constraint_id": "B1|V1|A1|E1",
      "category": "beat|visual_attribute|audio|editing",
      "status": "match|mismatch|missing|uncertain",
      "severity": 1,
      "evidence": "direct evidence only"
    }}
  ],
  "violations": [
    {{
      "type": "beat_order_error|missing_beat|wrong_object|wrong_count|wrong_color|identity_drift|shot_continuity|audio_mismatch|offscreen_audio_mismatch|other",
      "severity": 1,
      "evidence": "direct evidence only"
    }}
  ],
  "summary": "brief overall assessment"
}}

## Constraints
- Keep required_beats / required_visual_attributes / required_audio_events / editing_requirements each to 3-10 items max.
- checks must cover ALL extracted constraints (B*, V*, A*, E*).
- severity is 1-5 (5 most severe).
- confidence is 0-1.
""".strip()

PLOT_JUDGE_SYSTEM_V2 = """
You are a strict, evidence-based judge for a text-to-audio-video benchmark.

Goal: evaluate how well the provided MP4 matches the given PROMPT in terms of plot/story alignment, including story beats / event sequence (plot development), shot changes / continuity / pacing (cuts), objects and attributes (counts, colors, identities), actions and causal relations, and audio alignment (dialogue/sfx/music/ambience) and whether audio matches what is shown.

Rules: use ONLY observable evidence from the video and audio. If something is not clearly observable (low resolution, fast motion, occlusion, darkness), mark UNCERTAIN. Do not invent details not supported by evidence. If the prompt is underspecified, do not penalize missing details; focus on core requirements. Output STRICT JSON only. No markdown. No code fences. No extra text.

Scoring: return plot_alignment_score in [0,100]. Use subscores in [0,100] for narrative_alignment, shot_alignment, visual_attribute_alignment, and audio_alignment. narrative_alignment is plot beats and ordering. shot_alignment is shot changes/cuts consistent with prompt (if specified) and continuity. visual_attribute_alignment is object presence, counts, colors, key attributes. audio_alignment is audio events that should occur vs actually heard, plus audio-visual consistency. Overall plot_alignment_score = round(0.35*narrative_alignment + 0.15*shot_alignment + 0.30*visual_attribute_alignment + 0.20*audio_alignment).

Deduction guidance: severe mismatch of a core plot beat is -25. Moderate mismatch is -12. Minor mismatch/subtle is -5. Missing core requirement is -10. Uncertain is -0, but reduce confidence.
""".strip()

PLOT_JUDGE_USER_TEMPLATE_V2 = """
Prompt to check:
{prompt_text}

Analyze the clip against the prompt and output STRICT JSON:
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
    "required_beats": [{{"id": "B1", "beat": "required beat", "strength": "strong|medium|weak"}}],
    "required_visual_attributes": [{{"id": "V1", "object": "object/entity", "attributes": {{"count": "integer or null", "color": "string or null", "other": ["..."]}}, "strength": "strong|medium|weak"}}],
    "required_audio_events": [{{"id": "A1", "audio_event": "required audio event", "strength": "strong|medium|weak"}}],
    "editing_requirements": [{{"id": "E1", "requirement": "editing requirement", "strength": "strong|medium|weak"}}]
  }},
  "observed_event_log": [
    {{"where": "beginning|middle|end|event_k", "visual": "what is visibly happening", "audio": "what is audibly happening"}}
  ],
  "checks": [
    {{"constraint_id": "B1|V1|A1|E1", "category": "beat|visual_attribute|audio|editing", "status": "match|mismatch|missing|uncertain", "severity": 1, "evidence": "evidence from the clip"}}
  ],
  "violations": [
    {{"type": "beat_order_error|missing_beat|wrong_object|wrong_count|wrong_color|identity_drift|shot_continuity|audio_mismatch|offscreen_audio_mismatch|other", "severity": 1, "evidence": "evidence from the clip"}}
  ],
  "summary": "short judgment"
}}

Requirements: keep required_beats / required_visual_attributes / required_audio_events / editing_requirements each to 3-10 items max. checks must cover ALL extracted constraints (B*, V*, A*, E*). severity is 1-5 (5 most severe). confidence is 0-1.
""".strip()

PLOT_PROMPT_VARIANTS = {
    "original": (PLOT_JUDGE_SYSTEM, PLOT_JUDGE_USER_TEMPLATE),
    "v1": (PLOT_JUDGE_SYSTEM_V1, PLOT_JUDGE_USER_TEMPLATE_V1),
    "v2": (PLOT_JUDGE_SYSTEM_V2, PLOT_JUDGE_USER_TEMPLATE_V2),
}

if PROMPT_VARIANT not in PLOT_PROMPT_VARIANTS:
    raise ValueError(f"Unsupported PLOT_MATCHING_PROMPT_VARIANT: {PROMPT_VARIANT}")
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


def _get_gemini_api_key() -> str:
    return resolve_api_key()


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


def judge_plot_matching(video_path: str, prompt_text: str, timeout_s: int = TIMEOUT_S) -> Dict[str, Any]:
    system_instruction, user_template = PLOT_PROMPT_VARIANTS[PROMPT_VARIANT]
    raw_text = generate_content_text(
        model_name=MODEL_NAME,
        user_parts=[
            inline_file_part(video_path, mime_type="video/mp4"),
            user_template.format(prompt_text=prompt_text),
        ],
        api_key=_get_gemini_api_key(),
        timeout_s=timeout_s,
        system_instruction=system_instruction,
    )
    data = json.loads(_extract_json(raw_text))

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
        "model": MODEL_NAME,
    }

    return data


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("video_path", type=str)
    parser.add_argument("--prompt", type=str, default=None, help="Prompt text (plain text).")
    parser.add_argument("--prompt-file", type=str, default=None, help="Path to a text file containing the prompt.")
    parser.add_argument("--out", type=str, default="outputs/plot_matching_result.json")
    args = parser.parse_args()

    if args.prompt_file:
        prompt_text = Path(args.prompt_file).read_text(encoding="utf-8").strip()
    elif args.prompt is not None:
        prompt_text = args.prompt.strip()
    else:
        raise SystemExit("Provide --prompt or --prompt-file")

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)

    result = judge_plot_matching(args.video_path, prompt_text)

    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)

    print(result["plot_alignment_score"])
