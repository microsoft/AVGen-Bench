#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import csv
import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple


GROUP_WEIGHTS: Dict[str, float] = {
    "basic": 0.2,
    "cross": 0.2,
    "fine": 0.6,
}

GROUP_ORDER = ["basic", "cross", "fine"]
GROUP_DISPLAY = {
    "basic": "Basic Uni-modal",
    "cross": "Basic Cross-modal",
    "fine": "Fine-grained",
}


def _clamp(x: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, x))


def _to_float(x: Any) -> Optional[float]:
    if x is None:
        return None
    if isinstance(x, (int, float)):
        return float(x)
    s = str(x).strip()
    if not s:
        return None
    try:
        return float(s)
    except Exception:
        return None


def _safe_mean(vals: List[float]) -> Optional[float]:
    if not vals:
        return None
    return float(sum(vals) / len(vals))


def _read_json(path: Path) -> Optional[Dict[str, Any]]:
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None


def _read_csv_row_by_key(path: Path, key_field: str, key_value: str) -> Optional[Dict[str, str]]:
    if not path.exists():
        return None
    try:
        with path.open("r", encoding="utf-8", newline="") as f:
            reader = csv.DictReader(f)
            for row in reader:
                if str(row.get(key_field, "")).strip() == key_value:
                    return row
    except Exception:
        return None
    return None


def _pick_existing_path(candidates: List[Path]) -> Path:
    for p in candidates:
        if p.exists():
            return p
    return candidates[0]


def _read_all_numeric_from_csv_col(path: Path, col: str) -> List[float]:
    vals: List[float] = []
    if not path.exists():
        return vals
    try:
        with path.open("r", encoding="utf-8", newline="") as f:
            reader = csv.DictReader(f)
            for row in reader:
                v = _to_float(row.get(col))
                if v is not None:
                    vals.append(v)
    except Exception:
        return []
    return vals


def _norm_higher_identity_100(x: float) -> float:
    return _clamp(x, 0.0, 100.0)


def _norm_vis(x: float) -> float:
    return _clamp(x * 100.0, 0.0, 100.0)


def _norm_aud_pq(x: float) -> float:
    return _clamp(x * 10.0, 0.0, 100.0)


def _norm_lophy(x: float) -> float:
    return _clamp(x * 20.0, 0.0, 100.0)


def _norm_low_better_linear(x: float, threshold: float) -> float:
    if threshold <= 0:
        return 0.0
    return _clamp(100.0 * (1.0 - x / threshold), 0.0, 100.0)


def _read_vis_qalign(root: Path, run_tag: str) -> Tuple[Optional[float], str]:
    candidates = []
    if run_tag:
        candidates.append(root / "q_align" / f"{run_tag}.csv")
        candidates.append(root / "q_align" / run_tag / "summary.csv")
    candidates.append(root / "q_align" / "summary.csv")
    p = _pick_existing_path(candidates)
    row = _read_csv_row_by_key(p, "folder", "__ALL__")
    return (_to_float(row.get("mean_score")) if row else None, str(p))


def _read_aud_pq(root: Path, run_tag: str) -> Tuple[Optional[float], str]:
    candidates = []
    if run_tag:
        candidates.append(root / "audiobox_aesthetic" / f"{run_tag}.csv")
        candidates.append(root / "audiobox_aesthetic" / run_tag / "summary.csv")
    candidates.append(root / "audiobox_aesthetic" / "summary.csv")
    p = _pick_existing_path(candidates)
    row = _read_csv_row_by_key(p, "folder", "__ALL__")
    return (_to_float(row.get("mean_PQ")) if row else None, str(p))


def _read_av_sync(root: Path, run_tag: str) -> Tuple[Optional[float], str]:
    candidates = []
    if run_tag:
        candidates.append(root / "av_sync" / f"{run_tag}.csv")
        candidates.append(root / "av_sync" / run_tag / "summary.csv")
    candidates.append(root / "av_sync" / "summary.csv")
    p = _pick_existing_path(candidates)
    row = _read_csv_row_by_key(p, "folder", "__ALL__")
    if not row:
        return None, str(p)
    v = _to_float(row.get("mean_abs_offset_cont_sec"))
    if v is None:
        v = _to_float(row.get("mean_abs_offset_argmax_sec"))
    return v, str(p)


def _read_lip_sync(root: Path, run_tag: str) -> Tuple[Optional[float], str]:
    candidates = []
    if run_tag:
        candidates.append(root / "syncnet" / run_tag / "result.csv")
    candidates.append(root / "syncnet" / "result.csv")
    p = _pick_existing_path(candidates)
    row = _read_csv_row_by_key(p, "class", "TOTAL")
    return (_to_float(row.get("mean_abs_offset_frames")) if row else None, str(p))


def _read_text_ocr(root: Path, run_tag: str) -> Tuple[Optional[float], str]:
    candidates = []
    if run_tag:
        candidates.append(root / "ocr" / run_tag / "results_text_quality.csv")
    candidates.append(root / "ocr" / "results_text_quality.csv")
    p = _pick_existing_path(candidates)
    vals = _read_all_numeric_from_csv_col(p, "overall_text_quality_score")
    return _safe_mean(vals), str(p)


def _read_face(root: Path, run_tag: str) -> Tuple[Optional[float], str]:
    candidates = []
    if run_tag:
        candidates.append(root / "facial" / run_tag / "eval_results.json")
    candidates.append(root / "facial" / "eval_results.json")
    p = _pick_existing_path(candidates)
    data = _read_json(p)
    if not data:
        return None, str(p)
    overall = data.get("overall", {}) or {}
    return _to_float(overall.get("mean_score_total")), str(p)


def _read_music(root: Path, run_tag: str) -> Tuple[Optional[float], str]:
    candidates = []
    if run_tag:
        candidates.append(root / "music" / run_tag / "summary.json")
    candidates.append(root / "music" / "summary.json")
    p = _pick_existing_path(candidates)
    data = _read_json(p)
    if not data:
        return None, str(p)
    return _to_float(data.get("mean_score")), str(p)


def _read_speech(root: Path, run_tag: str) -> Tuple[Optional[float], str]:
    candidates = []
    if run_tag:
        candidates.append(root / "speech" / run_tag / "summary.json")
    candidates.append(root / "speech" / "summary.json")
    p = _pick_existing_path(candidates)
    data = _read_json(p)
    if not data:
        return None, str(p)
    return _to_float(data.get("avg_score")), str(p)


def _read_lophy(root: Path, run_tag: str) -> Tuple[Optional[float], str]:
    candidates = []
    if run_tag:
        candidates.append(root / "videophy2" / f"{run_tag}.csv")
        candidates.append(root / "videophy2" / run_tag / "summary.csv")
    candidates.append(root / "videophy2" / "summary.csv")
    p = _pick_existing_path(candidates)
    row = _read_csv_row_by_key(p, "folder", "__ALL__")
    return (_to_float(row.get("mean_score")) if row else None, str(p))


def _read_hiphy(root: Path, run_tag: str) -> Tuple[Optional[float], str]:
    candidates = []
    if run_tag:
        candidates.append(root / "gemini_phy2" / run_tag / "summary.csv")
        candidates.append(root / "gemini_phy" / run_tag / "summary.csv")
    candidates.append(root / "gemini_phy" / "summary.csv")
    p = _pick_existing_path(candidates)
    row = _read_csv_row_by_key(p, "folder", "__ALL__")
    return (_to_float(row.get("mean_overall")) if row else None, str(p))


def _read_holistic(root: Path, run_tag: str) -> Tuple[Optional[float], str]:
    candidates = []
    if run_tag:
        candidates.append(root / "plot_matching" / run_tag / "eval_results.json")
    candidates.append(root / "plot_matching" / "eval_results.json")
    p = _pick_existing_path(candidates)
    data = _read_json(p)
    if not data:
        return None, str(p)
    overall = data.get("overall", {}) or {}
    return _to_float(overall.get("mean_plot_alignment_score_total")), str(p)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Aggregate AVGen-Bench module outputs into a unified total score (Scheme 2)."
    )
    parser.add_argument("--output-dir", type=str, required=True, help="Evaluation output root, e.g. eval_results")
    parser.add_argument("--av-threshold-sec", type=float, default=0.5, help="AV offset threshold for score=0")
    parser.add_argument("--lip-threshold-frames", type=float, default=8.0, help="Lip offset threshold for score=0")
    parser.add_argument("--run-tag", type=str, default="", help="Optional run tag for avgenbench-style outputs")
    parser.add_argument("--save-json", type=str, default="", help="Optional: save aggregate result JSON")
    parser.add_argument("--save-csv", type=str, default="", help="Optional: save aggregate one-row CSV")
    args = parser.parse_args()

    output_dir = Path(args.output_dir).expanduser().resolve()
    if not output_dir.exists():
        raise FileNotFoundError(f"output dir not found: {output_dir}")

    metric_defs: List[Dict[str, Any]] = [
        {"name": "Vis", "group": "basic", "read": _read_vis_qalign, "norm": _norm_vis},
        {"name": "Aud", "group": "basic", "read": _read_aud_pq, "norm": _norm_aud_pq},
        {
            "name": "AV",
            "group": "cross",
            "read": _read_av_sync,
            "norm": lambda x: _norm_low_better_linear(x, args.av_threshold_sec),
        },
        {
            "name": "Lip",
            "group": "cross",
            "read": _read_lip_sync,
            "norm": lambda x: _norm_low_better_linear(x, args.lip_threshold_frames),
        },
        {"name": "Text", "group": "fine", "read": _read_text_ocr, "norm": _norm_higher_identity_100},
        {"name": "Face", "group": "fine", "read": _read_face, "norm": _norm_higher_identity_100},
        {"name": "Music", "group": "fine", "read": _read_music, "norm": _norm_higher_identity_100},
        {"name": "Speech", "group": "fine", "read": _read_speech, "norm": _norm_higher_identity_100},
        {"name": "Lo-Phy", "group": "fine", "read": _read_lophy, "norm": _norm_lophy},
        {"name": "Hi-Phy", "group": "fine", "read": _read_hiphy, "norm": _norm_higher_identity_100},
        {"name": "Holistic", "group": "fine", "read": _read_holistic, "norm": _norm_higher_identity_100},
    ]

    n_by_group: Dict[str, int] = {g: 0 for g in GROUP_WEIGHTS}
    for m in metric_defs:
        n_by_group[m["group"]] += 1

    metrics_out: List[Dict[str, Any]] = []
    weighted_num = 0.0
    weighted_den = 0.0
    group_norm_values: Dict[str, List[float]] = {g: [] for g in GROUP_WEIGHTS}

    for m in metric_defs:
        raw, source = m["read"](output_dir, args.run_tag)
        norm: Optional[float] = None
        if raw is not None:
            try:
                norm = float(m["norm"](float(raw)))
            except Exception:
                norm = None

        g = m["group"]
        global_w = GROUP_WEIGHTS[g] / float(n_by_group[g]) if n_by_group[g] > 0 else 0.0
        available = norm is not None
        if available:
            weighted_num += global_w * float(norm)
            weighted_den += global_w
            group_norm_values[g].append(float(norm))

        metrics_out.append(
            {
                "name": m["name"],
                "group": g,
                "raw_value": raw,
                "normalized_0_100": norm,
                "global_weight": global_w,
                "available": available,
                "source": source,
            }
        )

    group_scores: Dict[str, Optional[float]] = {}
    for g in GROUP_ORDER:
        group_scores[g] = _safe_mean(group_norm_values[g])

    total_score = (weighted_num / weighted_den) if weighted_den > 0 else None
    coverage = weighted_den  # total global weight sums to 1.0
    coverage = _clamp(coverage, 0.0, 1.0)

    missing_metrics = [m["name"] for m in metrics_out if not m["available"]]

    result = {
        "scheme": "Scheme-2",
        "formula": {
            "group_weights": GROUP_WEIGHTS,
            "metric_weighting": "equal weight per metric within each group",
            "normalization": {
                "Vis": "Vis * 100",
                "Aud": "Aud(PQ) * 10",
                "AV": f"100 * max(0, 1 - AV / {args.av_threshold_sec})",
                "Lip": f"100 * max(0, 1 - Lip / {args.lip_threshold_frames})",
                "Lo-Phy": "Lo-Phy * 20",
                "others": "already 0-100",
            },
        },
        "output_dir": str(output_dir),
        "total_score": total_score,
        "coverage": coverage,
        "group_scores": group_scores,
        "metrics": metrics_out,
        "missing_metrics": missing_metrics,
    }

    print("========== Aggregate Score (Scheme 2) ==========")
    for g in GROUP_ORDER:
        print(f"[{GROUP_DISPLAY[g]}]")
        for m in [x for x in metrics_out if x["group"] == g]:
            raw_s = "N/A" if m["raw_value"] is None else f"{float(m['raw_value']):.6f}"
            norm_s = "N/A" if m["normalized_0_100"] is None else f"{float(m['normalized_0_100']):.2f}"
            print(
                f"  - {m['name']:<8s} raw={raw_s:<10s} norm={norm_s:<7s} w={m['global_weight']:.4f}"
            )
        gs = group_scores[g]
        gs_s = "N/A" if gs is None else f"{gs:.2f}"
        print(f"  -> {GROUP_DISPLAY[g]} score: {gs_s}")

    print("------------------------------------------------")
    total_s = "N/A" if total_score is None else f"{total_score:.2f}"
    print(f"Total Score: {total_s}")
    print(f"Coverage   : {coverage * 100.0:.1f}%")
    if missing_metrics:
        print(f"Missing metrics: {', '.join(missing_metrics)}")

    if args.save_json:
        out_json = Path(args.save_json).expanduser().resolve()
        out_json.parent.mkdir(parents=True, exist_ok=True)
        out_json.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"Saved aggregate JSON: {out_json}")

    if args.save_csv:
        out_csv = Path(args.save_csv).expanduser().resolve()
        out_csv.parent.mkdir(parents=True, exist_ok=True)
        row: Dict[str, Any] = {
            "total_score": total_score if total_score is not None else "",
            "coverage": coverage,
            "group_basic": group_scores["basic"] if group_scores["basic"] is not None else "",
            "group_cross": group_scores["cross"] if group_scores["cross"] is not None else "",
            "group_fine": group_scores["fine"] if group_scores["fine"] is not None else "",
        }
        for m in metrics_out:
            key_raw = f"raw_{m['name'].replace('-', '_').lower()}"
            key_norm = f"norm_{m['name'].replace('-', '_').lower()}"
            row[key_raw] = m["raw_value"] if m["raw_value"] is not None else ""
            row[key_norm] = m["normalized_0_100"] if m["normalized_0_100"] is not None else ""
        with out_csv.open("w", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(row.keys()))
            writer.writeheader()
            writer.writerow(row)
        print(f"Saved aggregate CSV: {out_csv}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
