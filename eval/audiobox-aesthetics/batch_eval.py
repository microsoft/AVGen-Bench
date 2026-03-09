#!/usr/bin/env python3
# -*- coding: utf-8 -*-


import csv
import argparse
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import torchaudio
from audiobox_aesthetics.infer import initialize_predictor


METRICS = ["CE", "CU", "PC", "PQ"]


def safe_mean(vals: List[float]) -> Optional[float]:
    return (sum(vals) / len(vals)) if vals else None


def find_mp4s_in_subfolders(root_dir: Path) -> List[Tuple[Path, List[Path]]]:
    subfolders = [p for p in root_dir.iterdir() if p.is_dir()]
    groups = []
    for sf in sorted(subfolders):
        mp4s = sorted(sf.rglob("*.mp4"))
        if mp4s:
            groups.append((sf.relative_to(root_dir), mp4s))
    return groups


def find_mp4s_recursive_grouped(root_dir: Path) -> List[Tuple[Path, List[Path]]]:
    folder_to_mp4s: Dict[Path, List[Path]] = {}
    for vp in sorted(root_dir.rglob("*.mp4")):
        rel_parent = vp.parent.relative_to(root_dir)
        folder_to_mp4s.setdefault(rel_parent, []).append(vp)
    return sorted(folder_to_mp4s.items(), key=lambda x: str(x[0]))


def score_one_video_audio(predictor, video_path: Path) -> Dict[str, float]:

    wav, sr = torchaudio.load(str(video_path))  
    input_data = [{"path": wav, "sample_rate": sr}]
    preds = predictor.forward(input_data)
    score = preds[0]
    return {k: float(score[k]) for k in METRICS}


def main():
    parser = argparse.ArgumentParser("Batch eval AudioBox Aesthetics for mp4 videos.")
    parser.add_argument("--root", type=str, required=True, help="Root directory: subfolders contain mp4 files")
    parser.add_argument("--recursive_root", action="store_true",
                        help="Recursively find all mp4 files under root and group by relative parent directory")
    parser.add_argument("--save_csv", type=str, default="",
                        help="Optional: save per-video results CSV, e.g. results.csv")
    parser.add_argument("--save_summary_csv", type=str, default="",
                        help="Optional: save summary CSV, e.g. summary.csv")
    parser.add_argument("--fail_fast", action="store_true", help="Exit immediately on error; otherwise skip the failed video")
    args = parser.parse_args()

    root_dir = Path(args.root).expanduser().resolve()
    if not root_dir.exists():
        raise FileNotFoundError(f"root dir not found: {root_dir}")

    print("Initializing AudioBox Aesthetics predictor (the first run may download checkpoints)...")
    predictor = initialize_predictor()

    if args.recursive_root:
        groups = find_mp4s_recursive_grouped(root_dir)
    else:
        groups = find_mp4s_in_subfolders(root_dir)

    if not groups:
        print(f"No mp4 files found under {root_dir}.")
        return

    total_videos = sum(len(vs) for _, vs in groups)
    done = 0

    folder_metrics: Dict[str, Dict[str, List[float]]] = {}
    all_metrics: Dict[str, List[float]] = {m: [] for m in METRICS}

    per_video_rows = []

    for rel_folder, mp4s in groups:
        folder_key = str(rel_folder)
        if folder_key not in folder_metrics:
            folder_metrics[folder_key] = {m: [] for m in METRICS}

        print("=" * 60)
        print(f"Subfolder: {folder_key} | videos: {len(mp4s)}")

        for vp in mp4s:
            done += 1
            try:
                s = score_one_video_audio(predictor, vp)

                for m in METRICS:
                    folder_metrics[folder_key][m].append(s[m])
                    all_metrics[m].append(s[m])

                per_video_rows.append({
                    "folder": folder_key,
                    "video_path": str(vp),
                    **{m: s[m] for m in METRICS}
                })

                print(f"[{done}/{total_videos}] {vp.name} -> "
                      f"CE={s['CE']:.4f} CU={s['CU']:.4f} PC={s['PC']:.4f} PQ={s['PQ']:.4f}")

            except Exception as e:
                msg = f"Scoring failed: {vp} | error: {repr(e)}"
                if args.fail_fast:
                    raise RuntimeError(msg) from e
                print(msg)

        n_valid = len(folder_metrics[folder_key]["CE"])
        if n_valid == 0:
            print("Subfolder mean: N/A (no valid results)")
        else:
            means = {m: safe_mean(folder_metrics[folder_key][m]) for m in METRICS}
            print(f"Subfolder mean (n={n_valid}): "
                  f"CE={means['CE']:.4f} CU={means['CU']:.4f} PC={means['PC']:.4f} PQ={means['PQ']:.4f}")

    print("\n" + "#" * 60)
    n_all = len(all_metrics["CE"])
    if n_all == 0:
        print("Overall mean: N/A (no valid results)")
    else:
        overall = {m: safe_mean(all_metrics[m]) for m in METRICS}
        print(f"Overall mean (n={n_all}): "
              f"CE={overall['CE']:.4f} CU={overall['CU']:.4f} PC={overall['PC']:.4f} PQ={overall['PQ']:.4f}")

    if args.save_csv:
        out_csv = Path(args.save_csv).expanduser().resolve()
        out_csv.parent.mkdir(parents=True, exist_ok=True)
        with out_csv.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=["folder", "video_path"] + METRICS)
            writer.writeheader()
            writer.writerows(per_video_rows)
        print(f"Saved per-video results: {out_csv}")

    if args.save_summary_csv:
        out_sum = Path(args.save_summary_csv).expanduser().resolve()
        out_sum.parent.mkdir(parents=True, exist_ok=True)
        with out_sum.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=["folder", "n"] + [f"mean_{m}" for m in METRICS])
            writer.writeheader()

            for folder, mdict in sorted(folder_metrics.items(), key=lambda x: x[0]):
                n = len(mdict["CE"])
                row = {"folder": folder, "n": n}
                for m in METRICS:
                    row[f"mean_{m}"] = safe_mean(mdict[m]) if n > 0 else ""
                writer.writerow(row)

            # overall
            row = {"folder": "__ALL__", "n": n_all}
            for m in METRICS:
                row[f"mean_{m}"] = safe_mean(all_metrics[m]) if n_all > 0 else ""
            writer.writerow(row)

        print(f"Saved summary results: {out_sum}")


if __name__ == "__main__":
    main()
