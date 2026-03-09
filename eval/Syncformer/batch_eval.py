#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import csv
import argparse
import subprocess
from pathlib import Path
from typing import Dict, List, Tuple, Optional

import torch
import torchvision
from omegaconf import OmegaConf

from dataset.dataset_utils import get_video_and_audio
from dataset.transforms import make_class_grid
from utils.utils import check_if_file_exists_else_download, which_ffmpeg
from scripts.train_utils import get_model, get_transforms, prepare_inputs


os.environ["VLLM_WORKER_MULTIPROC_METHOD"] = "spawn"  # harmless here; keep consistent if your env uses it


import subprocess
from pathlib import Path
from utils.utils import which_ffmpeg

def reencode_video(path, vfps=25, afps=16000, in_size=256):
    ffmpeg = which_ffmpeg()
    assert ffmpeg != "", "Is ffmpeg installed? Check conda env."

    path = str(path)
    out_dir = Path.cwd() / "vis"
    out_dir.mkdir(exist_ok=True, parents=True)

    new_mp4 = out_dir / f"{Path(path).stem}_{vfps}fps_{in_size}side_{afps}hz.mp4"
    new_wav = out_dir / f"{Path(path).stem}_{vfps}fps_{in_size}side_{afps}hz.wav"

    # 1) reencode mp4 (fps/resize/ar)
    cmd1 = [
        ffmpeg, "-hide_banner", "-loglevel", "error", "-y",
        "-i", path,
        "-vf", f"fps={vfps},scale=iw*{in_size}/min(iw\\,ih):ih*{in_size}/min(iw\\,ih),crop=trunc(iw/2)*2:trunc(ih/2)*2",
        "-ar", str(afps),
        str(new_mp4),
    ]
    subprocess.run(cmd1, check=True)

    if not new_mp4.exists():
        raise RuntimeError(f"ffmpeg did not produce mp4: {new_mp4}")

    # 2) extract mono wav (optional; keep same behavior as your original)
    cmd2 = [
        ffmpeg, "-hide_banner", "-loglevel", "error", "-y",
        "-i", str(new_mp4),
        "-acodec", "pcm_s16le", "-ac", "1",
        str(new_wav),
    ]
    subprocess.run(cmd2, check=True)

    return str(new_mp4)



def patch_config(cfg):
    cfg.model.params.afeat_extractor.params.ckpt_path = None
    cfg.model.params.vfeat_extractor.params.ckpt_path = None
    cfg.model.params.transformer.target = cfg.model.params.transformer.target.replace(
        ".modules.feature_selector.", ".sync_model."
    )
    return cfg


def safe_mean(vals: List[float]) -> Optional[float]:
    return sum(vals) / len(vals) if vals else None


def find_groups(root_dir: Path, recursive_root: bool) -> List[Tuple[Path, List[Path]]]:
    """
    Returns [(rel_folder, [mp4...]), ...]
    - recursive_root=True: group all mp4 files under root by relative parent directory
    - otherwise: only scan direct subfolders under root, recursively for mp4
    """
    if recursive_root:
        folder_to_mp4s: Dict[Path, List[Path]] = {}
        for vp in sorted(root_dir.rglob("*.mp4")):
            rel_parent = vp.parent.relative_to(root_dir)
            folder_to_mp4s.setdefault(rel_parent, []).append(vp)
        return sorted(folder_to_mp4s.items(), key=lambda x: str(x[0]))

    groups = []
    for sf in sorted([p for p in root_dir.iterdir() if p.is_dir()]):
        mp4s = sorted(sf.rglob("*.mp4"))
        if mp4s:
            groups.append((sf.relative_to(root_dir), mp4s))
    return groups


def maybe_reencode_if_needed(vid_path: str, vfps=25, afps=16000, in_size=256) -> str:
    v, _, info = torchvision.io.read_video(vid_path, pts_unit="sec")
    _, H, W, _ = v.shape
    if info.get("video_fps", None) != vfps or info.get("audio_fps", None) != afps or min(H, W) != in_size:
        return reencode_video(vid_path, vfps=vfps, afps=afps, in_size=in_size)
    return vid_path


@torch.no_grad()
def predict_offset_for_video(
    vid_path: str,
    device: torch.device,
    cfg,
    model,
    grid: torch.Tensor,
    transforms_test,
    v_start_i_sec: float = 0.0,
):
    """
    Returns:
      pred_offset_sec_cont: softmax expectation E[grid] (seconds)
      pred_offset_sec_argmax: grid value at argmax (seconds)
      pred_class_idx: argmax class index
    """
    rgb, audio, meta = get_video_and_audio(vid_path, get_meta=True)

    item = dict(
        video=rgb,
        audio=audio,
        meta=meta,
        path=vid_path,
        split="test",
        targets={"v_start_i_sec": v_start_i_sec, "offset_sec": 0.0},
    )

    item = transforms_test(item)

    batch = torch.utils.data.default_collate([item])
    aud, vid, targets = prepare_inputs(batch, device)

    with torch.autocast("cuda", enabled=getattr(cfg.training, "use_half_precision", False)):
        _, logits = model(vid, aud)  # (B=1, num_cls)

    probs = torch.softmax(logits, dim=-1)[0]  # (num_cls,)
    grid = grid.to(probs.device)

    pred_cont = float((probs * grid).sum().item())
    pred_idx = int(torch.argmax(probs).item())
    pred_argmax = float(grid[pred_idx].item())

    return pred_cont, pred_argmax, pred_idx


def main():
    parser = argparse.ArgumentParser("Batch eval A/V offset with SyncFormer (report cont + argmax).")
    parser.add_argument("--root", type=str, required=True, help="Root directory: subfolders contain mp4 files")
    parser.add_argument("--exp_name", required=True, help="syncformer exp name: xx-xx-xxTxx-xx-xx")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--recursive_root", action="store_true")
    parser.add_argument("--v_start_i_sec", type=float, default=0.0)

    parser.add_argument("--no_reencode", action="store_true", help="Disable automatic re-encoding")
    parser.add_argument("--save_csv", type=str, default="", help="Per-video results CSV")
    parser.add_argument("--save_summary_csv", type=str, default="", help="Summary results CSV")
    parser.add_argument("--fail_fast", action="store_true")
    args = parser.parse_args()

    root_dir = Path(args.root).expanduser().resolve()
    if not root_dir.exists():
        raise FileNotFoundError(root_dir)

    cfg_path = f"./logs/sync_models/{args.exp_name}/cfg-{args.exp_name}.yaml"
    ckpt_path = f"./logs/sync_models/{args.exp_name}/{args.exp_name}.pt"
    check_if_file_exists_else_download(cfg_path)
    check_if_file_exists_else_download(ckpt_path)

    cfg = patch_config(OmegaConf.load(cfg_path))

    max_off_sec = cfg.data.max_off_sec
    num_cls = cfg.model.params.transformer.params.off_head_cfg.params.out_features
    grid = make_class_grid(-max_off_sec, max_off_sec, num_cls)

    device = torch.device(args.device)
    _, model = get_model(cfg, device)
    ckpt = torch.load(ckpt_path, map_location=torch.device("cpu"), weights_only=False)
    model.load_state_dict(ckpt["model"])
    model.eval()

    transforms_test = get_transforms(cfg, ["test"])["test"]

    groups = find_groups(root_dir, args.recursive_root)
    if not groups:
        print(f"No mp4 found under: {root_dir}")
        return

    total_videos = sum(len(vs) for _, vs in groups)
    done = 0

    per_video_rows: List[Dict] = []

    folder_abs_cont: Dict[str, List[float]] = {}
    folder_abs_argmax: Dict[str, List[float]] = {}

    all_abs_cont: List[float] = []
    all_abs_argmax: List[float] = []

    vfps, afps, in_size = 25, 16000, 256

    for rel_folder, mp4s in groups:
        folder_key = str(rel_folder)
        folder_abs_cont.setdefault(folder_key, [])
        folder_abs_argmax.setdefault(folder_key, [])

        print("=" * 70)
        print(f"Folder: {folder_key} | videos: {len(mp4s)}")

        for vp in mp4s:
            done += 1
            try:
                used_path = str(vp)
                if not args.no_reencode:
                    used_path = maybe_reencode_if_needed(used_path, vfps=vfps, afps=afps, in_size=in_size)

                pred_cont, pred_argmax, pred_idx = predict_offset_for_video(
                    vid_path=used_path,
                    device=device,
                    cfg=cfg,
                    model=model,
                    grid=grid,
                    transforms_test=transforms_test,
                    v_start_i_sec=args.v_start_i_sec,
                )

                abs_cont = abs(pred_cont)
                abs_argmax = abs(pred_argmax)

                folder_abs_cont[folder_key].append(abs_cont)
                folder_abs_argmax[folder_key].append(abs_argmax)
                all_abs_cont.append(abs_cont)
                all_abs_argmax.append(abs_argmax)

                per_video_rows.append({
                    "folder": folder_key,
                    "video_path": str(vp),
                    "used_path": used_path,
                    "pred_offset_sec_cont": pred_cont,
                    "pred_offset_sec_argmax": pred_argmax,
                    "pred_class_idx": pred_idx,
                    "abs_pred_offset_sec_cont": abs_cont,
                    "abs_pred_offset_sec_argmax": abs_argmax,
                })

                print(f"[{done}/{total_videos}] {vp.name} | "
                      f"cont={pred_cont:+.3f}s abs_cont={abs_cont:.3f}s | "
                      f"argmax={pred_argmax:+.3f}s abs_argmax={abs_argmax:.3f}s")

            except Exception as e:
                msg = f"FAILED: {vp} | error={repr(e)}"
                if args.fail_fast:
                    raise RuntimeError(msg) from e
                print(msg)

        n = len(folder_abs_cont[folder_key])
        if n == 0:
            print("Folder summary: N/A (no valid videos)")
        else:
            mean_abs_cont = safe_mean(folder_abs_cont[folder_key])
            mean_abs_argmax = safe_mean(folder_abs_argmax[folder_key])
            print(f"Folder summary: n={n} "
                  f"mean_abs_offset_cont={mean_abs_cont:.4f}s "
                  f"mean_abs_offset_argmax={mean_abs_argmax:.4f}s")

    print("\n" + "#" * 70)
    n_all = len(all_abs_cont)
    if n_all == 0:
        print("Overall summary: N/A (no valid videos)")
    else:
        overall_mean_abs_cont = safe_mean(all_abs_cont)
        overall_mean_abs_argmax = safe_mean(all_abs_argmax)
        print(f"Overall summary: n={n_all} "
              f"mean_abs_offset_cont={overall_mean_abs_cont:.4f}s "
              f"mean_abs_offset_argmax={overall_mean_abs_argmax:.4f}s")

    # per-video csv
    if args.save_csv:
        out_csv = Path(args.save_csv).expanduser().resolve()
        out_csv.parent.mkdir(parents=True, exist_ok=True)
        fieldnames = [
            "folder",
            "video_path",
            "used_path",
            "pred_offset_sec_cont",
            "pred_offset_sec_argmax",
            "pred_class_idx",
            "abs_pred_offset_sec_cont",
            "abs_pred_offset_sec_argmax",
        ]
        with out_csv.open("w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=fieldnames)
            w.writeheader()
            w.writerows(per_video_rows)
        print(f"Saved per-video CSV: {out_csv}")

    # summary csv
    if args.save_summary_csv:
        out_sum = Path(args.save_summary_csv).expanduser().resolve()
        out_sum.parent.mkdir(parents=True, exist_ok=True)
        with out_sum.open("w", newline="", encoding="utf-8") as f:
            fieldnames = ["folder", "n", "mean_abs_offset_cont_sec", "mean_abs_offset_argmax_sec"]
            w = csv.DictWriter(f, fieldnames=fieldnames)
            w.writeheader()

            for folder in sorted(folder_abs_cont.keys()):
                vals_c = folder_abs_cont[folder]
                vals_a = folder_abs_argmax[folder]
                w.writerow({
                    "folder": folder,
                    "n": len(vals_c),
                    "mean_abs_offset_cont_sec": (safe_mean(vals_c) if vals_c else ""),
                    "mean_abs_offset_argmax_sec": (safe_mean(vals_a) if vals_a else ""),
                })

            w.writerow({
                "folder": "__ALL__",
                "n": n_all,
                "mean_abs_offset_cont_sec": (safe_mean(all_abs_cont) if n_all > 0 else ""),
                "mean_abs_offset_argmax_sec": (safe_mean(all_abs_argmax) if n_all > 0 else ""),
            })
        print(f"Saved summary CSV: {out_sum}")


if __name__ == "__main__":
    main()
