#!/usr/bin/env python3

import argparse
import os
from pathlib import Path
from typing import Iterable


LTX_REPOS = {
    "2": "Lightricks/LTX-2",
    "2.3": "Lightricks/LTX-2.3",
}

LTX_FILES = {
    "2": {
        "dev": "ltx-2-19b-dev.safetensors",
        "dev-fp8": "ltx-2-19b-dev-fp8.safetensors",
        "dev-fp4": "ltx-2-19b-dev-fp4.safetensors",
        "distilled": "ltx-2-19b-distilled.safetensors",
        "distilled-fp8": "ltx-2-19b-distilled-fp8.safetensors",
        "distilled-lora": "ltx-2-19b-distilled-lora-384.safetensors",
        "spatial-upscaler-x2": "ltx-2-spatial-upscaler-x2-1.0.safetensors",
        "temporal-upscaler-x2": "ltx-2-temporal-upscaler-x2-1.0.safetensors",
    },
    "2.3": {
        "dev": "ltx-2.3-22b-dev.safetensors",
        "distilled": "ltx-2.3-22b-distilled.safetensors",
        "distilled-lora": "ltx-2.3-22b-distilled-lora-384.safetensors",
        "spatial-upscaler-x2": "ltx-2.3-spatial-upscaler-x2-1.0.safetensors",
        "spatial-upscaler-x1.5": "ltx-2.3-spatial-upscaler-x1.5-1.0.safetensors",
        "temporal-upscaler-x2": "ltx-2.3-temporal-upscaler-x2-1.0.safetensors",
    },
}

GEMMA_REPO = "google/gemma-3-12b-it-qat-q4_0-unquantized"


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def iter_unique(items: Iterable[str]) -> list[str]:
    seen = set()
    out = []
    for item in items:
        if item not in seen:
            out.append(item)
            seen.add(item)
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description="Download LTX-2 and LTX-2.3 weights from Hugging Face.")
    ap.add_argument(
        "--version",
        type=str,
        default="all",
        choices=["2", "2.3", "all"],
        help="Which LTX version to download.",
    )
    ap.add_argument(
        "--models",
        nargs="+",
        default=["distilled"],
        help="Which model checkpoints to download (e.g. distilled, dev, dev-fp8, dev-fp4, distilled-fp8).",
    )
    ap.add_argument(
        "--output-dir",
        type=str,
        default="models/ltx2",
        help="Directory to store downloaded weights.",
    )
    ap.add_argument(
        "--spatial-upscaler",
        type=str,
        default="x2",
        choices=["x2", "x1.5", "none"],
        help="Spatial upscaler to download (if supported by the version).",
    )
    ap.add_argument(
        "--include-temporal-upscaler",
        action="store_true",
        help="Download the temporal upscaler.",
    )
    ap.add_argument(
        "--include-distilled-lora",
        action="store_true",
        help="Download the distilled LoRA (optional).",
    )
    ap.add_argument(
        "--download-gemma",
        action="store_true",
        help="Download Gemma 3 text encoder repository.",
    )
    ap.add_argument(
        "--token",
        type=str,
        default=None,
        help="Optional Hugging Face token. Falls back to HUGGINGFACE_HUB_TOKEN.",
    )
    args = ap.parse_args()

    try:
        from huggingface_hub import hf_hub_download, snapshot_download
    except Exception as e:
        raise SystemExit(
            "huggingface_hub is required. Install it with: pip install huggingface_hub"
        ) from e

    token = args.token or os.environ.get("HUGGINGFACE_HUB_TOKEN")
    output_dir = Path(args.output_dir).expanduser().resolve()
    ensure_dir(output_dir)

    versions = ["2", "2.3"] if args.version == "all" else [args.version]

    for version in versions:
        repo_id = LTX_REPOS[version]
        file_map = LTX_FILES[version]

        want = []
        for model in args.models:
            key = model.strip()
            if key not in file_map:
                raise SystemExit(f"Model '{model}' is not available for LTX-{version}.")
            want.append(file_map[key])

        if args.include_distilled_lora:
            want.append(file_map["distilled-lora"])

        if args.spatial_upscaler != "none":
            spatial_key = f"spatial-upscaler-{args.spatial_upscaler}"
            if spatial_key not in file_map:
                raise SystemExit(f"Spatial upscaler '{args.spatial_upscaler}' is not available for LTX-{version}.")
            want.append(file_map[spatial_key])

        if args.include_temporal_upscaler:
            want.append(file_map["temporal-upscaler-x2"])

        for filename in iter_unique(want):
            print(f"Downloading {repo_id}/{filename} -> {output_dir}")
            hf_hub_download(
                repo_id=repo_id,
                filename=filename,
                local_dir=str(output_dir),
                local_dir_use_symlinks=False,
                token=token,
            )

    if args.download_gemma:
        gemma_dir = output_dir / GEMMA_REPO.split("/")[-1]
        ensure_dir(gemma_dir)
        print(f"Downloading {GEMMA_REPO} -> {gemma_dir}")
        snapshot_download(
            repo_id=GEMMA_REPO,
            local_dir=str(gemma_dir),
            local_dir_use_symlinks=False,
            token=token,
        )

    print("Done.")


if __name__ == "__main__":
    main()
