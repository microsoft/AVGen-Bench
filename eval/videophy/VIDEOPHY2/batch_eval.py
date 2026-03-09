#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import re
import csv
import argparse
from pathlib import Path
from collections import defaultdict

import pandas as pd
import torch
from tqdm import tqdm

from peft import LoraConfig, get_peft_model
from transformers.models.llama.tokenization_llama import LlamaTokenizer
from mplug_owl_video.modeling_mplug_owl import MplugOwlForConditionalGeneration
from mplug_owl_video.processing_mplug_owl import MplugOwlImageProcessor, MplugOwlProcessor

from template import PROMPT_SA, PROMPT_PHYSICS, PROMPT_RULE


generate_kwargs = {
    "do_sample": False,
    "top_k": 1,
    "temperature": 0.001,
    "max_length": 256,
}


def modify_keys(state_dict):
    new_state_dict = defaultdict()
    pattern = re.compile(r".*language_model.*\.(q_proj|v_proj|k_proj|o_proj|gate_proj|down_proj|up_proj).weight")

    for key, value in state_dict.items():
        if pattern.match(key):
            key_parts = key.split(".")
            key_parts.insert(-1, "base_layer")
            key = ".".join(key_parts)
        new_state_dict[key] = value
    return new_state_dict


def build_model_and_processor(checkpoint: str, lora_checkpoint: str | None, device: str, dtype=torch.bfloat16):
    tokenizer = LlamaTokenizer.from_pretrained(checkpoint)
    image_processor = MplugOwlImageProcessor.from_pretrained(checkpoint)
    processor = MplugOwlProcessor(image_processor, tokenizer)

    model = MplugOwlForConditionalGeneration.from_pretrained(
        checkpoint,
        torch_dtype=dtype,
        device_map={"": "cpu"},
    )
    model.eval()

    if lora_checkpoint:
        peft_config = LoraConfig(
            target_modules=r".*language_model.*\.(q_proj|v_proj|k_proj|o_proj|gate_proj|down_proj|up_proj)",
            inference_mode=True,
            r=32,
            lora_alpha=32,
            lora_dropout=0.05,
        )
        model = get_peft_model(model, peft_config)
        with open(lora_checkpoint, "rb") as f:
            ckpt = torch.load(f, map_location=torch.device("cpu"))
        try:
            model.load_state_dict(ckpt)
        except Exception:
            ckpt2 = modify_keys(ckpt)
            model.load_state_dict(ckpt2)
        print(f"LoRA loaded: {lora_checkpoint}")

    model = model.to(device).to(dtype)
    return model, processor, tokenizer


def build_prompt(task: str, caption: str | None = None, rule: str | None = None) -> str:
    if task == "sa":
        if caption is None:
            raise ValueError("task=sa requires caption")
        return PROMPT_SA.format(caption=caption)
    elif task == "pc":
        return PROMPT_PHYSICS
    elif task == "rule":
        if rule is None:
            raise ValueError("task=rule requires rule")
        return PROMPT_RULE.format(rule=rule)
    else:
        raise ValueError(f"Unknown task: {task}")


def parse_score_from_output(output: str) -> int:
    num_map = {
        "zero": 0, "one": 1, "two": 2, "three": 3, "four": 4, "five": 5,
        "0": 0, "1": 1, "2": 2, "3": 3, "4": 4, "5": 5,
    }
    out = output.lower().strip()

    for k, v in num_map.items():
        if k in out:
            return v

    digits = "".join([c for c in out if c.isdigit()])
    if digits:
        try:
            val = int(digits)
            if val in {0, 1, 2, 3, 4, 5}:
                return val
        except Exception:
            pass

    return 0


@torch.no_grad()
def score_video(model, processor, tokenizer, videopath: str, prompt: str, num_frames: int) -> tuple[int, str]:
    inputs = processor(
        text=[prompt],
        videos=[videopath],
        num_frames=num_frames,
        return_tensors="pt",
    )
    inputs = {k: (v.bfloat16() if v.dtype == torch.float else v) for k, v in inputs.items()}
    inputs = {k: v.to(model.device) for k, v in inputs.items()}

    res = model.generate(**inputs, **generate_kwargs)
    output = tokenizer.decode(res.tolist()[0], skip_special_tokens=True)
    score = parse_score_from_output(output)
    return score, output


def safe_mean(vals):
    return (sum(vals) / len(vals)) if vals else None


def main():
    parser = argparse.ArgumentParser(description="Batch eval Videophy2 (mplug_owl_video) for mp4 under subfolders.")
    parser.add_argument("--root", type=str, required=True, help="Root directory: each subfolder contains mp4 files")
    parser.add_argument("--checkpoint", type=str, required=True, help="Model checkpoint (e.g. videophy_2_auto)")
    parser.add_argument("--lora_checkpoint", type=str, default=None, help="Optional LoRA checkpoint path")
    parser.add_argument("--task", type=str, default="pc", choices=["sa", "pc", "rule"], help="Evaluation task")
    parser.add_argument("--num_frames", type=int, default=32)
    parser.add_argument("--device", type=str, default="cuda")

    parser.add_argument("--recursive_root", action="store_true",
                        help="Recursively collect all mp4 files under root and group by relative parent directory")
    parser.add_argument("--save_csv", type=str, default="", help="Optional: save per-video results to results.csv")
    parser.add_argument("--save_summary_csv", type=str, default="", help="Optional: save summary to summary.csv")
    parser.add_argument("--fail_fast", action="store_true", help="Exit immediately on error")
    args = parser.parse_args()

    root = Path(args.root).expanduser().resolve()
    if not root.exists():
        raise FileNotFoundError(f"root not found: {root}")

    groups = []
    if args.recursive_root:
        all_mp4s = sorted(root.rglob("*.mp4"))
        folder_map = defaultdict(list)
        for vp in all_mp4s:
            rel_parent = vp.parent.relative_to(root)
            folder_map[rel_parent].append(vp)
        groups = sorted(folder_map.items(), key=lambda x: str(x[0]))
    else:
        subfolders = sorted([p for p in root.iterdir() if p.is_dir()])
        for sf in subfolders:
            mp4s = sorted(sf.rglob("*.mp4"))
            if mp4s:
                groups.append((sf.relative_to(root), mp4s))

    if not groups:
        print(f"No mp4 found under: {root}")
        return

    if args.task != "pc":
        raise ValueError("Batch folder eval typically uses --task pc. If you need sa/rule, tell me how caption/rule are provided per video.")

    prompt = build_prompt(task=args.task)

    print("Loading model/processor...")
    model, processor, tokenizer = build_model_and_processor(
        checkpoint=args.checkpoint,
        lora_checkpoint=args.lora_checkpoint,
        device=args.device,
        dtype=torch.bfloat16,
    )
    print("Model loaded.")

    per_video_rows = []
    folder_scores = defaultdict(list)
    all_scores = []

    total = sum(len(vs) for _, vs in groups)
    done = 0

    for rel_folder, mp4s in groups:
        rel_folder_str = str(rel_folder)
        print("=" * 70)
        print(f"Folder: {rel_folder_str} | n={len(mp4s)}")

        for vp in tqdm(mp4s, desc=f"{rel_folder_str}", leave=False):
            done += 1
            try:
                score, raw = score_video(
                    model=model,
                    processor=processor,
                    tokenizer=tokenizer,
                    videopath=str(vp),
                    prompt=prompt,
                    num_frames=args.num_frames,
                )
                folder_scores[rel_folder_str].append(score)
                all_scores.append(score)

                per_video_rows.append({
                    "folder": rel_folder_str,
                    "video_path": str(vp),
                    "score": score,
                    "raw_output": raw,
                })

            except Exception as e:
                msg = f"Failed: {vp} | {repr(e)}"
                if args.fail_fast:
                    raise RuntimeError(msg) from e
                print(msg)

        m = safe_mean(folder_scores[rel_folder_str])
        if m is None:
            print("Folder mean: N/A")
        else:
            print(f"Folder mean: {m:.4f}  (n={len(folder_scores[rel_folder_str])})")

    overall = safe_mean(all_scores)
    print("\n" + "#" * 70)
    if overall is None:
        print("Overall mean: N/A")
    else:
        print(f"Overall mean: {overall:.4f}  (n={len(all_scores)})")

    if args.save_csv:
        out_csv = Path(args.save_csv).expanduser().resolve()
        out_csv.parent.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(per_video_rows).to_csv(out_csv, index=False)
        print(f"Saved per-video CSV: {out_csv}")

    if args.save_summary_csv:
        out_sum = Path(args.save_summary_csv).expanduser().resolve()
        out_sum.parent.mkdir(parents=True, exist_ok=True)

        rows = []
        for folder, scores in sorted(folder_scores.items(), key=lambda x: x[0]):
            rows.append({
                "folder": folder,
                "n": len(scores),
                "mean_score": safe_mean(scores) if scores else "",
            })
        rows.append({
            "folder": "__ALL__",
            "n": len(all_scores),
            "mean_score": overall if overall is not None else "",
        })
        pd.DataFrame(rows).to_csv(out_sum, index=False)
        print(f"Saved summary CSV: {out_sum}")


if __name__ == "__main__":
    main()
