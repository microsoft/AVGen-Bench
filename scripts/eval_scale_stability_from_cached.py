#!/usr/bin/env python3

import argparse
import csv
import json
import math
import random
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple


GROUP_WEIGHTS: Dict[str, float] = {
    "basic": 0.2,
    "cross": 0.2,
    "fine": 0.6,
}

GROUP_DIMENSIONS: Dict[str, Tuple[str, ...]] = {
    "basic": ("Vis", "Aud"),
    "cross": ("AV", "Lip"),
    "fine": ("Text", "Face", "Music", "Speech", "Lo-Phy", "Hi-Phy", "Holistic"),
}

METRIC_DEFS = [
    {"name": "Vis", "group": "basic", "dimension": "Vis", "kind": "category", "key": "vis", "norm": lambda x: clamp(x * 100.0, 0.0, 100.0)},
    {"name": "Aud", "group": "basic", "dimension": "Aud", "kind": "category", "key": "aud", "norm": lambda x: clamp(x * 10.0, 0.0, 100.0)},
    {"name": "AV", "group": "cross", "dimension": "AV", "kind": "category", "key": "av", "norm": lambda x: clamp(100.0 * (1.0 - x / 0.5), 0.0, 100.0)},
    {"name": "Lip", "group": "cross", "dimension": "Lip", "kind": "category", "key": "lip", "norm": lambda x: clamp(100.0 * (1.0 - x / 8.0), 0.0, 100.0)},
    {"name": "Text", "group": "fine", "dimension": "Text", "kind": "exact", "key": "text", "norm": lambda x: clamp(x, 0.0, 100.0)},
    {"name": "Face", "group": "fine", "dimension": "Face", "kind": "exact", "key": "face", "norm": lambda x: clamp(x, 0.0, 100.0)},
    {"name": "Music", "group": "fine", "dimension": "Music", "kind": "exact", "key": "music", "norm": lambda x: clamp(x, 0.0, 100.0)},
    {"name": "Speech", "group": "fine", "dimension": "Speech", "kind": "exact", "key": "speech", "norm": lambda x: clamp(x, 0.0, 100.0)},
    {"name": "Lo-Phy", "group": "fine", "dimension": "Lo-Phy", "kind": "category", "key": "lo_phy", "norm": lambda x: clamp(x * 20.0, 0.0, 100.0)},
    {"name": "Hi-Phy", "group": "fine", "dimension": "Hi-Phy", "kind": "exact", "key": "hi_phy", "norm": lambda x: clamp(x, 0.0, 100.0)},
    {"name": "Holistic", "group": "fine", "dimension": "Holistic", "kind": "exact", "key": "holistic", "norm": lambda x: clamp(x, 0.0, 100.0)},
]


def clamp(x: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, x))


def to_float(x: Any) -> Optional[float]:
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


def mean(xs: List[float]) -> Optional[float]:
    return (sum(xs) / len(xs)) if xs else None


def variance(xs: List[float]) -> Optional[float]:
    if not xs:
        return None
    m = mean(xs)
    if m is None:
        return None
    return sum((x - m) ** 2 for x in xs) / len(xs)


def stddev(xs: List[float]) -> Optional[float]:
    v = variance(xs)
    return math.sqrt(v) if v is not None else None


def read_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def read_csv_rows(path: Path) -> List[Dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def category_from_video_path(video_path: str) -> str:
    p = Path(video_path)
    return p.parent.name


def load_category_metric_map(csv_path: Path, value_field: str) -> Dict[str, float]:
    out: Dict[str, float] = {}
    for row in read_csv_rows(csv_path):
        category = str(row.get("folder", "")).strip()
        if not category or category == "__ALL__":
            continue
        value = to_float(row.get(value_field))
        if value is not None:
            out[category] = value
    return out


def load_exact_text(csv_path: Path) -> Dict[str, Dict[str, Any]]:
    out: Dict[str, Dict[str, Any]] = {}
    for row in read_csv_rows(csv_path):
        video_path = str(row.get("video_path", "")).strip()
        score = to_float(row.get("overall_text_quality_score"))
        if not video_path or score is None:
            continue
        prompt_requires = str(row.get("prompt_requires_visible_text", "")).strip().lower() == "true"
        text_presence = str(row.get("text_presence", "") or "").strip().lower()
        out[video_path] = {
            "category": str(row.get("folder", "")).strip() or category_from_video_path(video_path),
            "score": score,
            "include_filtered": prompt_requires or text_presence == "incidental",
        }
    return out


def load_exact_speech(scored_json: Path) -> Dict[str, Dict[str, Any]]:
    out: Dict[str, Dict[str, Any]] = {}
    for item in read_json(scored_json):
        video_path = str(item.get("video_path", "")).strip()
        score = to_float((item.get("judgement") or {}).get("score"))
        if not video_path or score is None:
            continue
        out[video_path] = {
            "category": str(item.get("category", "")).strip() or category_from_video_path(video_path),
            "score": score,
        }
    return out


def load_exact_facial(eval_json: Path) -> Dict[str, Dict[str, Any]]:
    out: Dict[str, Dict[str, Any]] = {}
    data = read_json(eval_json)
    for item in data.get("results", []):
        if item.get("status") != "ok":
            continue
        video_path = str(item.get("video_path", "")).strip()
        score = to_float(item.get("final_score"))
        if not video_path or score is None:
            continue
        out[video_path] = {
            "category": str(item.get("category", "")).strip() or category_from_video_path(video_path),
            "score": score,
        }
    return out


def load_exact_music(summary_json: Path) -> Dict[str, Dict[str, Any]]:
    out: Dict[str, Dict[str, Any]] = {}
    data = read_json(summary_json)
    for item in data.get("results", []):
        if item.get("status") != "ok":
            continue
        video_path = str(item.get("video_path", "")).strip()
        score = to_float(item.get("overall_score"))
        if not video_path or score is None:
            continue
        out[video_path] = {
            "category": str(item.get("category", "")).strip() or category_from_video_path(video_path),
            "score": score,
        }
    return out


def load_exact_plot(eval_json: Path) -> Dict[str, Dict[str, Any]]:
    out: Dict[str, Dict[str, Any]] = {}
    data = read_json(eval_json)
    for item in data.get("results", []):
        if item.get("status") != "ok":
            continue
        video_path = str(item.get("video_path", "")).strip()
        score = to_float(item.get("plot_alignment_score"))
        if not video_path or score is None:
            continue
        out[video_path] = {
            "category": str(item.get("category", "")).strip() or category_from_video_path(video_path),
            "score": score,
        }
    return out


def load_exact_hi_phy(csv_path: Path) -> Dict[str, Dict[str, Any]]:
    out: Dict[str, Dict[str, Any]] = {}
    for row in read_csv_rows(csv_path):
        if str(row.get("status", "")).strip() != "ok":
            continue
        video_path = str(row.get("video_path", "")).strip()
        score = to_float(row.get("overall_score"))
        if not video_path or score is None:
            continue
        out[video_path] = {
            "category": str(row.get("folder", "")).strip() or category_from_video_path(video_path),
            "score": score,
        }
    return out


def build_model_data(repeat_dir: Path) -> Dict[str, Any]:
    eval_root = repeat_dir / "eval_output"
    run_tag = repeat_dir.name

    exact = {
        "text": load_exact_text(eval_root / "ocr" / run_tag / "results_text_quality.csv"),
        "speech": load_exact_speech(eval_root / "speech" / run_tag / "scored.json"),
        "face": load_exact_facial(eval_root / "facial" / run_tag / "eval_results.json"),
        "music": load_exact_music(eval_root / "music" / run_tag / "summary.json"),
        "holistic": load_exact_plot(eval_root / "plot_matching" / run_tag / "eval_results.json"),
        "hi_phy": load_exact_hi_phy(eval_root / "gemini_phy2" / run_tag / "results.csv"),
    }
    category_metrics = {
        "vis": load_category_metric_map(eval_root / "q_align" / f"{run_tag}.csv", "mean_score"),
        "aud": load_category_metric_map(eval_root / "audiobox_aesthetic" / f"{run_tag}.csv", "mean_PQ"),
        "av": load_category_metric_map(eval_root / "av_sync" / f"{run_tag}.csv", "mean_abs_offset_cont_sec"),
        "lip": load_category_metric_map(eval_root / "syncnet" / run_tag / "result.csv", "mean_abs_offset_frames"),
        "lo_phy": load_category_metric_map(eval_root / "videophy2" / f"{run_tag}.csv", "mean_score"),
    }

    universe: Dict[str, str] = {}
    for metric_map in exact.values():
        for video_path, meta in metric_map.items():
            universe[video_path] = meta["category"]

    return {
        "repeat_dir": str(repeat_dir),
        "exact": exact,
        "category_metrics": category_metrics,
        "videos": sorted(universe.keys()),
        "video_to_category": universe,
    }


def subset_raw_exact(metric_key: str, data: Dict[str, Any], selected_videos: List[str]) -> Optional[float]:
    metric_map = data["exact"][metric_key]
    if metric_key == "text":
        filtered_scores = []
        all_scores = []
        for video_path in selected_videos:
            meta = metric_map.get(video_path)
            if not meta:
                continue
            all_scores.append(float(meta["score"]))
            if meta["include_filtered"]:
                filtered_scores.append(float(meta["score"]))
        if filtered_scores:
            return mean(filtered_scores)
        return mean(all_scores)

    vals = [float(metric_map[video_path]["score"]) for video_path in selected_videos if video_path in metric_map]
    return mean(vals)


def subset_raw_category(metric_key: str, data: Dict[str, Any], selected_videos: List[str]) -> Optional[float]:
    cat_map = data["category_metrics"][metric_key]
    vals = [cat_map[data["video_to_category"][video_path]] for video_path in selected_videos if data["video_to_category"].get(video_path) in cat_map]
    return mean(vals)


def score_subset(data: Dict[str, Any], selected_videos: List[str]) -> Dict[str, Any]:
    n_by_dimension: Dict[Tuple[str, str], int] = {}
    for spec in METRIC_DEFS:
        key = (spec["group"], spec["dimension"])
        n_by_dimension[key] = n_by_dimension.get(key, 0) + 1

    weighted_num = 0.0
    weighted_den = 0.0
    group_dimension_values: Dict[str, Dict[str, List[float]]] = {
        g: {d: [] for d in dims} for g, dims in GROUP_DIMENSIONS.items()
    }
    metrics_out: Dict[str, Dict[str, Any]] = {}

    for spec in METRIC_DEFS:
        if spec["kind"] == "exact":
            raw = subset_raw_exact(spec["key"], data, selected_videos)
        else:
            raw = subset_raw_category(spec["key"], data, selected_videos)
        norm = float(spec["norm"](raw)) if raw is not None else None
        dimension = spec["dimension"]
        dimension_weight = GROUP_WEIGHTS[spec["group"]] / float(len(GROUP_DIMENSIONS[spec["group"]]))
        global_w = dimension_weight / float(n_by_dimension[(spec["group"], dimension)])
        if norm is not None:
            weighted_num += global_w * norm
            weighted_den += global_w
            group_dimension_values[spec["group"]][dimension].append(norm)
        metrics_out[spec["name"]] = {
            "group": spec["group"],
            "dimension": dimension,
            "raw": raw,
            "norm": norm,
            "global_weight": global_w,
            "available": norm is not None,
        }

    group_scores = {}
    for group, dim_values in group_dimension_values.items():
        dimension_scores = [mean(vals) for vals in dim_values.values() if vals]
        group_scores[group] = mean([x for x in dimension_scores if x is not None])
    total_score = (weighted_num / weighted_den) if weighted_den > 0 else None
    return {
        "total_score": total_score,
        "coverage": weighted_den,
        "group_basic": group_scores["basic"],
        "group_cross": group_scores["cross"],
        "group_fine": group_scores["fine"],
        "metrics": metrics_out,
    }


def summarize_trials(scores: List[float]) -> Dict[str, Optional[float]]:
    return {
        "mean": mean(scores),
        "variance": variance(scores),
        "stddev": stddev(scores),
        "min": min(scores) if scores else None,
        "max": max(scores) if scores else None,
    }


def fmt(x: Optional[float], ndigits: int = 4) -> str:
    if x is None:
        return "NA"
    return f"{x:.{ndigits}f}"


def write_markdown(path: Path, all_rows: List[Dict[str, Any]], trials: int, seed: int) -> None:
    by_model: Dict[str, List[Dict[str, Any]]] = {}
    for row in all_rows:
        by_model.setdefault(row["model"], []).append(row)

    lines = [
        "# Scale Stability From Cached Repeat01 Results",
        "",
        "- Method: random subset rescoring from cached eval outputs",
        "- Trials per subset ratio: {}".format(trials),
        "- Random seed: {}".format(seed),
        "- Exact per-video metrics: Text, Face, Music, Speech, Hi-Phy, Holistic",
        "- Category-weighted approximation only: Vis, Aud, AV, Lip, Lo-Phy",
        "",
    ]
    for model, rows in by_model.items():
        rows = sorted(rows, key=lambda r: r["subset_ratio"], reverse=True)
        lines.extend([
            f"## {model}",
            "",
            "| Subset | n_subset | Full Score | Mean | Variance | Stddev | Min | Max | Mean-Full |",
            "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
        ])
        for row in rows:
            lines.append(
                "| {subset_label} | {subset_size} | {full_score} | {mean_score} | {variance_score} | {stddev_score} | {min_score} | {max_score} | {delta_mean} |".format(
                    subset_label=row["subset_label"],
                    subset_size=row["subset_size"],
                    full_score=fmt(row["full_score"]),
                    mean_score=fmt(row["mean_score"]),
                    variance_score=fmt(row["variance_score"]),
                    stddev_score=fmt(row["stddev_score"]),
                    min_score=fmt(row["min_score"]),
                    max_score=fmt(row["max_score"]),
                    delta_mean=fmt(row["mean_score"] - row["full_score"] if row["mean_score"] is not None and row["full_score"] is not None else None),
                )
            )
        lines.append("")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--session-dir", required=True)
    ap.add_argument("--trials", type=int, default=200)
    ap.add_argument("--seed", type=int, default=20260331)
    args = ap.parse_args()

    session_dir = Path(args.session_dir).resolve()
    model_repeats = [
        ("Veo_3.1_fast", session_dir / "Veo_3.1_fast" / "repeat_01_180932_65fc"),
        ("LTX-2", session_dir / "LTX-2" / "repeat_01_225449_f052"),
    ]
    subset_ratios = [0.8, 0.6, 0.4, 0.2]

    rng = random.Random(args.seed)
    all_rows: List[Dict[str, Any]] = []
    all_trials: List[Dict[str, Any]] = []

    for model_name, repeat_dir in model_repeats:
        data = build_model_data(repeat_dir)
        videos = data["videos"]
        full_score = score_subset(data, videos)["total_score"]
        for ratio in subset_ratios:
            subset_size = max(1, int(round(len(videos) * ratio)))
            trial_scores: List[float] = []
            for trial_idx in range(args.trials):
                selected = rng.sample(videos, subset_size)
                result = score_subset(data, selected)
                score = result["total_score"]
                if score is None:
                    continue
                trial_scores.append(score)
                all_trials.append({
                    "model": model_name,
                    "repeat_dir": str(repeat_dir),
                    "subset_ratio": ratio,
                    "subset_size": subset_size,
                    "trial_index": trial_idx,
                    "total_score": score,
                    "group_basic": result["group_basic"],
                    "group_cross": result["group_cross"],
                    "group_fine": result["group_fine"],
                    "coverage": result["coverage"],
                })
            stats = summarize_trials(trial_scores)
            all_rows.append({
                "model": model_name,
                "repeat_dir": str(repeat_dir),
                "subset_ratio": ratio,
                "subset_label": f"{int(ratio * 100)}%",
                "subset_size": subset_size,
                "full_score": full_score,
                "mean_score": stats["mean"],
                "variance_score": stats["variance"],
                "stddev_score": stats["stddev"],
                "min_score": stats["min"],
                "max_score": stats["max"],
                "num_trials": len(trial_scores),
            })

    out_csv = session_dir / "scale_stability_subset_repeat01.csv"
    out_json = session_dir / "scale_stability_subset_repeat01.json"
    out_md = session_dir / "scale_stability_subset_repeat01.md"

    with out_csv.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "model",
                "subset_ratio",
                "subset_label",
                "subset_size",
                "full_score",
                "mean_score",
                "variance_score",
                "stddev_score",
                "min_score",
                "max_score",
                "num_trials",
                "repeat_dir",
            ],
        )
        writer.writeheader()
        writer.writerows(all_rows)

    out_json.write_text(
        json.dumps(
            {
                "session_dir": str(session_dir),
                "trials": args.trials,
                "seed": args.seed,
                "summary_rows": all_rows,
                "trial_rows": all_trials,
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    write_markdown(out_md, all_rows, args.trials, args.seed)


if __name__ == "__main__":
    main()
