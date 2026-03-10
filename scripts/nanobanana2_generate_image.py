#!/usr/bin/env python3

import argparse
import os
from pathlib import Path

from generation.clients.nanobanana2 import Nanobanana2Client


def resolve_output_path(out_path: Path, default_ext: str) -> Path:
    if out_path.suffix:
        return out_path
    return out_path.with_suffix(default_ext)


def build_prompt(user_prompt: str) -> str:
    return f"generate the first frame of the following video:{user_prompt}"


def main() -> None:
    ap = argparse.ArgumentParser(description="Generate a single image with nanobanana2.")
    ap.add_argument("--prompt", type=str, required=True, help="Text prompt for image generation.")
    ap.add_argument(
        "--output",
        type=str,
        default="generated_image.png",
        help="Output image path.",
    )
    ap.add_argument(
        "--model",
        type=str,
        default="gemini-3.1-flash-image-preview",
        help="Nanobanana2 model name.",
    )
    ap.add_argument(
        "--image_ext",
        type=str,
        default=None,
        help="Optional image extension (e.g. .png, .jpg, .webp). Defaults to output suffix.",
    )
    ap.add_argument(
        "--api_key",
        type=str,
        default=None,
        help="Override API key (otherwise uses GOOGLE_API_KEY or GEMINI_API_KEY).",
    )
    args = ap.parse_args()

    out_path = Path(args.output).expanduser()
    image_ext = args.image_ext or out_path.suffix or ".png"

    client = Nanobanana2Client(model=args.model, api_key=args.api_key)
    final_prompt = build_prompt(args.prompt)
    artifact = client.image_generation(prompt=final_prompt, model=args.model, image_ext=image_ext)

    final_path = resolve_output_path(out_path, artifact.extension)
    final_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = final_path.with_suffix(final_path.suffix + ".part")
    with open(tmp_path, "wb") as f:
        f.write(artifact.data)
    os.replace(tmp_path, final_path)
    print(f"Saved image: {final_path}")


if __name__ == "__main__":
    main()
