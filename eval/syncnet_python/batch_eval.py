#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import re
import csv
import argparse
import subprocess
from pathlib import Path
from collections import defaultdict

TRACK_RE = re.compile(
    r"AV offset:\s*([+-]?\d+)\s*frames.*?"
    r"Min dist:\s*([0-9]*\.?[0-9]+).*?"
    r"Confidence:\s*([0-9]*\.?[0-9]+)",
    re.DOTALL
)

def find_videos(video_root: Path):
    items = []
    for class_dir in sorted([p for p in video_root.iterdir() if p.is_dir()]):
        class_name = class_dir.name
        for mp4 in sorted(class_dir.rglob("*.mp4")):
            items.append((class_name, mp4))
    return items

def run_inference(inference_py: Path, videofile: Path, data_dir: Path,
                  batch_size: int, vshift: int, frame_rate: int,
                  facedet_scale: float, crop_scale: float, min_track: int,
                  num_failed_det: int, min_face_size: int,
                  timeout_sec: int = 3600):
    cmd = [
        "python", str(inference_py),
        "--videofile", str(videofile),
        "--data_dir", str(data_dir),
        "--batch_size", str(batch_size),
        "--vshift", str(vshift),
        "--frame_rate", str(frame_rate),
        "--facedet_scale", str(facedet_scale),
        "--crop_scale", str(crop_scale),
        "--min_track", str(min_track),
        "--num_failed_det", str(num_failed_det),
        "--min_face_size", str(min_face_size),
    ]
    p = subprocess.run(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        timeout=timeout_sec,
    )
    return p.returncode, p.stdout

def parse_tracks(output_text: str):
    tracks = []
    for m in TRACK_RE.finditer(output_text):
        tracks.append({
            "offset": int(m.group(1)),
            "min_dist": float(m.group(2)),
            "conf": float(m.group(3)),
        })
    return tracks

def safe_mean(vals):
    return sum(vals) / len(vals) if vals else None

def new_stat():
    return {
        "n_videos_total": 0,
        "n_videos_ran": 0,
        "n_videos_with_tracks": 0,
        "n_videos_with_pass_tracks": 0,
        "n_tracks_total": 0,
        "n_tracks_pass_conf": 0,
        "failures": 0,
        "abs_offset": [],
        "min_dist": [],
        "conf": [],
    }

def main():
    ap = argparse.ArgumentParser("Batch SyncNet evaluator (count all tracks with conf>=th)")
    ap.add_argument("--video_root", type=str, required=True, help="Root video directory (subfolders are classes)")
    ap.add_argument("--save_csv", type=str, required=True, help="Path to save summary CSV")
    ap.add_argument("--inference_py", type=str, default="inference.py", help="Path to inference.py")
    ap.add_argument("--data_dir", type=str, default="data/work_batch", help="Intermediate output directory (may become large)")
    ap.add_argument("--conf_th", type=float, default=5.0, help="Count only tracks with confidence >= this threshold")
    ap.add_argument("--timeout_sec", type=int, default=3600, help="Per-video timeout in seconds")

    ap.add_argument("--batch_size", type=int, default=20)
    ap.add_argument("--vshift", type=int, default=15)
    ap.add_argument("--frame_rate", type=int, default=25)
    ap.add_argument("--facedet_scale", type=float, default=0.25)
    ap.add_argument("--crop_scale", type=float, default=0.40)
    ap.add_argument("--min_track", type=int, default=30)
    ap.add_argument("--num_failed_det", type=int, default=25)
    ap.add_argument("--min_face_size", type=int, default=100)

    args = ap.parse_args()

    video_root = Path(args.video_root)
    save_csv = Path(args.save_csv)
    inference_py = Path(args.inference_py)
    data_dir = Path(args.data_dir)

    assert video_root.exists(), f"video_root not found: {video_root}"
    assert inference_py.exists(), f"inference_py not found: {inference_py}"
    save_csv.parent.mkdir(parents=True, exist_ok=True)
    data_dir.mkdir(parents=True, exist_ok=True)

    videos = find_videos(video_root)
    if not videos:
        raise RuntimeError(f"No mp4 videos found under: {video_root}")

    per_class = defaultdict(new_stat)
    total = new_stat()

    for i, (cls, vpath) in enumerate(videos, 1):
        per_class[cls]["n_videos_total"] += 1
        total["n_videos_total"] += 1

        print(f"[{i}/{len(videos)}] class={cls} video={vpath}")

        try:
            code, out = run_inference(
                inference_py=inference_py,
                videofile=vpath,
                data_dir=data_dir,
                batch_size=args.batch_size,
                vshift=args.vshift,
                frame_rate=args.frame_rate,
                facedet_scale=args.facedet_scale,
                crop_scale=args.crop_scale,
                min_track=args.min_track,
                num_failed_det=args.num_failed_det,
                min_face_size=args.min_face_size,
                timeout_sec=args.timeout_sec,
            )
        except subprocess.TimeoutExpired:
            per_class[cls]["failures"] += 1
            total["failures"] += 1
            print("  -> TIMEOUT")
            continue
        except Exception as e:
            per_class[cls]["failures"] += 1
            total["failures"] += 1
            print(f"  -> ERROR: {e}")
            continue

        per_class[cls]["n_videos_ran"] += 1
        total["n_videos_ran"] += 1

        if code != 0:
            per_class[cls]["failures"] += 1
            total["failures"] += 1
            print(f"  -> inference.py exit code={code}")
            continue

        tracks = parse_tracks(out)
        if not tracks:
            print("  -> no tracks parsed")
            continue

        per_class[cls]["n_videos_with_tracks"] += 1
        total["n_videos_with_tracks"] += 1

        per_class[cls]["n_tracks_total"] += len(tracks)
        total["n_tracks_total"] += len(tracks)

        pass_tracks = [t for t in tracks if t["conf"] >= args.conf_th]
        if not pass_tracks:
            print(f"  -> 0/{len(tracks)} tracks pass conf_th={args.conf_th}")
            continue

        per_class[cls]["n_videos_with_pass_tracks"] += 1
        total["n_videos_with_pass_tracks"] += 1

        per_class[cls]["n_tracks_pass_conf"] += len(pass_tracks)
        total["n_tracks_pass_conf"] += len(pass_tracks)

        for t in pass_tracks:
            per_class[cls]["abs_offset"].append(abs(t["offset"]))
            per_class[cls]["min_dist"].append(t["min_dist"])
            per_class[cls]["conf"].append(t["conf"])

            total["abs_offset"].append(abs(t["offset"]))
            total["min_dist"].append(t["min_dist"])
            total["conf"].append(t["conf"])

        print(f"  -> keep tracks: {len(pass_tracks)}/{len(tracks)}")

    fieldnames = [
        "class",
        "conf_th",
        "n_videos_total", "n_videos_ran",
        "n_videos_with_tracks", "n_videos_with_pass_tracks",
        "n_tracks_total", "n_tracks_pass_conf",
        "failures",
        "mean_abs_offset_frames",
        "mean_min_dist",
        "mean_confidence",
    ]

    with save_csv.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()

        for cls in sorted(per_class.keys()):
            st = per_class[cls]
            w.writerow({
                "class": cls,
                "conf_th": args.conf_th,
                "n_videos_total": st["n_videos_total"],
                "n_videos_ran": st["n_videos_ran"],
                "n_videos_with_tracks": st["n_videos_with_tracks"],
                "n_videos_with_pass_tracks": st["n_videos_with_pass_tracks"],
                "n_tracks_total": st["n_tracks_total"],
                "n_tracks_pass_conf": st["n_tracks_pass_conf"],
                "failures": st["failures"],
                "mean_abs_offset_frames": safe_mean(st["abs_offset"]),
                "mean_min_dist": safe_mean(st["min_dist"]),
                "mean_confidence": safe_mean(st["conf"]),
            })

        w.writerow({
            "class": "TOTAL",
            "conf_th": args.conf_th,
            "n_videos_total": total["n_videos_total"],
            "n_videos_ran": total["n_videos_ran"],
            "n_videos_with_tracks": total["n_videos_with_tracks"],
            "n_videos_with_pass_tracks": total["n_videos_with_pass_tracks"],
            "n_tracks_total": total["n_tracks_total"],
            "n_tracks_pass_conf": total["n_tracks_pass_conf"],
            "failures": total["failures"],
            "mean_abs_offset_frames": safe_mean(total["abs_offset"]),
            "mean_min_dist": safe_mean(total["min_dist"]),
            "mean_confidence": safe_mean(total["conf"]),
        })

    print(f"\nSaved: {save_csv}")
    print(f"TOTAL tracks kept: {total['n_tracks_pass_conf']} / {total['n_tracks_total']}  (conf_th={args.conf_th})")

if __name__ == "__main__":
    main()
