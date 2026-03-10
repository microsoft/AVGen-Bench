import io
import json
import os
import random
import shutil
import subprocess
import sys
import tempfile
import time
import uuid
from pathlib import Path
from typing import Optional, Tuple

from .base import BaseGenerationClient, GenerationArtifact


def _normalize_image_ext(image_ext: Optional[str]) -> Optional[str]:
    if image_ext is None:
        return None
    ext = image_ext.strip().lower()
    if not ext:
        return None
    if not ext.startswith("."):
        ext = "." + ext
    return ext


def _format_for_ext(ext: str) -> str:
    if ext in (".jpg", ".jpeg"):
        return "JPEG"
    if ext == ".webp":
        return "WEBP"
    return "PNG"

def _parse_size(size: str) -> Tuple[int, int]:
    normalized = str(size).strip().lower().replace("*", "x")
    parts = [p.strip() for p in normalized.split("x") if p.strip()]
    if len(parts) != 2:
        raise ValueError(f"Invalid size '{size}', expected format WIDTHxHEIGHT")
    width = int(parts[0])
    height = int(parts[1])
    if width <= 0 or height <= 0:
        raise ValueError(f"Invalid size '{size}', width/height must be positive")
    return width, height


def _aspect_ratio_for_size(width: int, height: int) -> str:
    ratio = width / float(height)
    candidates = {
        "21:9": 21 / 9,
        "16:9": 16 / 9,
        "4:3": 4 / 3,
        "3:2": 3 / 2,
        "1:1": 1.0,
        "3:4": 3 / 4,
        "9:16": 9 / 16,
        "2:3": 2 / 3,
    }
    best = None
    best_diff = 1e9
    for name, val in candidates.items():
        diff = abs(ratio - val)
        if diff < best_diff:
            best = name
            best_diff = diff
    if best is None or best_diff > 0.02:
        return "auto"
    return best


def _build_config_content(
    prompt: str,
    save_path: str,
    model_path: str,
    vq_path: str,
    tokenizer_path: str,
    vq_type: str,
    task_type: str,
    use_image: bool,
    aspect_ratio: str,
    hf_device: str,
    vq_device: str,
    classifier_free_guidance: float,
    max_new_tokens: int,
    image_area: int,
    image_cfg_scale: float,
    seed: int,
) -> str:
    prompt_literal = json.dumps(prompt, ensure_ascii=False)
    return f"""from pathlib import Path
from src.utils.logging_utils import setup_logger

cfg_name = Path(__file__).stem

model_path = {json.dumps(model_path)}
vq_path = {json.dumps(vq_path)}

tokenizer_path = {json.dumps(tokenizer_path)}
vq_type = {json.dumps(vq_type)}

task_type = {json.dumps(task_type)}
use_image = {str(bool(use_image))}

exp_name = "emu3p5-image"
save_path = {json.dumps(save_path)}
save_to_proto = True
setup_logger(save_path)

hf_device = {json.dumps(hf_device)}
vq_device = {json.dumps(vq_device)}
streaming = False
unconditional_type = "no_text"
classifier_free_guidance = {float(classifier_free_guidance)}
max_new_tokens = {int(max_new_tokens)}
image_area = {int(image_area)}
image_cfg_scale = {float(image_cfg_scale)}

aspect_ratios = {{
    "4:3": "55*73",
    "21:9": "41*97",
    "16:9": "47*85",
    "3:2": "52*78",
    "1:1": "64*64",
    "3:4": "73*55",
    "9:16": "85*47",
    "2:3": "78*52",
    "default": "55*73",
    "auto": None,
}}


def get_target_size(aspect_ratio: str):
    value = aspect_ratios.get(aspect_ratio, None)
    if value is None:
        return None, None

    h, w = map(int, value.split("*"))
    return h, w


aspect_ratio = {json.dumps(aspect_ratio)}
target_height, target_width = get_target_size(aspect_ratio)


def build_unc_and_template(task: str, with_image: bool):
    task_str = task.lower()
    if with_image:
        unc_p = "<|extra_203|>You are a helpful assistant. USER: <|IMAGE|> ASSISTANT: <|extra_100|>"
        tmpl = "<|extra_203|>You are a helpful assistant for %s task. USER: {{question}}<|IMAGE|> ASSISTANT: <|extra_100|>" % task_str
    else:
        unc_p = "<|extra_203|>You are a helpful assistant. USER:  ASSISTANT: <|extra_100|>"
        tmpl = "<|extra_203|>You are a helpful assistant for %s task. USER: {{question}} ASSISTANT: <|extra_100|>" % task_str
    return unc_p, tmpl


unc_prompt, template = build_unc_and_template(task_type, use_image)

sampling_params = dict(
    use_cache=True,
    text_top_k=1024,
    text_top_p=0.9,
    text_temperature=1.0,
    image_top_k=5120,
    image_top_p=1.0,
    image_temperature=1.0,
    top_k=131072,
    top_p=1.0,
    temperature=1.0,
    num_beams_per_group=1,
    num_beam_groups=1,
    diversity_penalty=0.0,
    max_new_tokens=max_new_tokens,
    guidance_scale=1.0,
    use_differential_sampling=True,
)

sampling_params["do_sample"] = sampling_params["num_beam_groups"] <= 1
sampling_params["num_beams"] = sampling_params["num_beams_per_group"] * sampling_params["num_beam_groups"]


special_tokens = dict(
    BOS="<|extra_203|>",
    EOS="<|extra_204|>",
    PAD="<|endoftext|>",
    EOL="<|extra_200|>",
    EOF="<|extra_201|>",
    TMS="<|extra_202|>",
    IMG="<|image token|>",
    BOI="<|image start|>",
    EOI="<|image end|>",
    BSS="<|extra_100|>",
    ESS="<|extra_101|>",
    BOG="<|extra_60|>",
    EOG="<|extra_61|>",
    BOC="<|extra_50|>",
    EOC="<|extra_51|>",
)

seed = {int(seed)}

prompts = [{prompt_literal}]
"""


class Emu35Client(BaseGenerationClient):
    def __init__(self, repo_dir: Optional[str] = None) -> None:
        default_repo_dir = Path(__file__).resolve().parents[2] / "third_party" / "Emu3.5"
        repo_dir = repo_dir or os.environ.get("EMU35_REPO_DIR") or str(default_repo_dir)
        self.repo_dir = Path(repo_dir).expanduser().resolve()
        self.inference_script = self.repo_dir / "inference.py"
        self.inference_vllm_script = self.repo_dir / "inference_vllm.py"
        if not self.inference_script.exists():
            raise FileNotFoundError(
                f"Emu3.5 inference script not found: {self.inference_script}. "
                "Please clone Emu3.5 into third_party/Emu3.5 or set EMU35_REPO_DIR."
            )

    def _run_inference(
        self,
        config_path: Path,
        timeout_s: int,
        python_bin: Optional[str],
        gpu_id: Optional[int],
        cuda_visible_devices: Optional[str],
        use_vllm: bool,
    ) -> None:
        py = python_bin or os.environ.get("EMU35_PYTHON_BIN") or sys.executable
        script = self.inference_vllm_script if use_vllm else self.inference_script
        if not script.exists():
            raise FileNotFoundError(f"Emu3.5 inference script not found: {script}")
        cmd = [py, str(script), "--cfg", str(config_path)]
        env = os.environ.copy()
        if cuda_visible_devices is not None:
            env["CUDA_VISIBLE_DEVICES"] = str(cuda_visible_devices)
        elif gpu_id is not None:
            env["CUDA_VISIBLE_DEVICES"] = str(gpu_id)

        proc = subprocess.run(
            cmd,
            cwd=str(self.repo_dir),
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            timeout=int(timeout_s),
            env=env,
        )
        if proc.returncode != 0:
            stderr_tail = (proc.stderr or "")[-2000:]
            stdout_tail = (proc.stdout or "")[-1200:]
            raise RuntimeError(
                f"Emu3.5 inference failed (exit={proc.returncode}).\n"
                f"stdout_tail:\n{stdout_tail}\n"
                f"stderr_tail:\n{stderr_tail}"
            )

    def _load_first_image(self, proto_path: Path) -> Tuple[bytes, str]:
        if str(self.repo_dir) not in sys.path:
            sys.path.insert(0, str(self.repo_dir))
        from src.proto import emu_pb as emu_pb

        story = emu_pb.Story()
        story.ParseFromString(proto_path.read_bytes())

        def first_image_from_list(images) -> Optional[Tuple[bytes, int]]:
            for image_meta in images:
                image = image_meta.image
                if image and image.image_data:
                    return image.image_data, int(image.format)
            return None

        img = first_image_from_list(story.reference_images)
        if img:
            image_data, fmt = img
        else:
            image_data = None
            fmt = None
            for clip in story.clips:
                for segment in clip.segments:
                    found = first_image_from_list(segment.images)
                    if found:
                        image_data, fmt = found
                        break
                if image_data:
                    break

        if not image_data:
            raise RuntimeError(f"No images found in proto: {proto_path}")

        if fmt == emu_pb.ImageFormat.JPEG:
            ext = ".jpg"
        elif fmt == emu_pb.ImageFormat.PNG:
            ext = ".png"
        elif fmt == emu_pb.ImageFormat.WEBP:
            ext = ".webp"
        elif fmt == emu_pb.ImageFormat.BMP:
            ext = ".bmp"
        else:
            ext = ".png"
        return image_data, ext

    def image_generation(
        self,
        prompt: str,
        model_path: Optional[str] = None,
        vq_path: Optional[str] = None,
        tokenizer_path: Optional[str] = None,
        vq_type: str = "ibq",
        task_type: str = "t2i",
        use_image: bool = False,
        aspect_ratio: str = "default",
        hf_device: str = "auto",
        vq_device: str = "cuda:0",
        classifier_free_guidance: float = 5.0,
        max_new_tokens: int = 5120,
        image_area: int = 1048576,
        image_cfg_scale: float = 1.0,
        seed: int = 6666,
        image_ext: Optional[str] = ".png",
        size: Optional[str] = None,
        timeout_s: int = 7200,
        python_bin: Optional[str] = None,
        gpu_id: Optional[int] = None,
        cuda_visible_devices: Optional[str] = None,
        use_vllm: bool = False,
        max_retries: int = 1,
        retry_backoff_s: float = 2.0,
        **kwargs,
    ) -> GenerationArtifact:
        del kwargs
        model_path = model_path or os.environ.get("EMU35_MODEL_PATH") or "BAAI/Emu3.5-Image"
        vq_path = vq_path or os.environ.get("EMU35_VQ_PATH") or "BAAI/Emu3.5-VisionTokenizer"
        tokenizer_path = tokenizer_path or os.environ.get("EMU35_TOKENIZER_PATH") or "./src/tokenizer_emu3_ibq"
        vq_type = vq_type or os.environ.get("EMU35_VQ_TYPE") or "ibq"
        hf_device = hf_device or os.environ.get("EMU35_HF_DEVICE") or "auto"
        vq_device = vq_device or os.environ.get("EMU35_VQ_DEVICE") or "cuda:0"
        aspect_ratio = aspect_ratio or os.environ.get("EMU35_ASPECT_RATIO") or "default"
        if size:
            width, height = _parse_size(size)
            image_area = int(width * height)
            aspect_ratio = _aspect_ratio_for_size(width, height)

        tmp_dir = Path(tempfile.mkdtemp(prefix="emu35_gen_"))
        cfg_name = f"emu3p5_tmp_{uuid.uuid4().hex}"
        cfg_rel_path = Path("configs") / f"{cfg_name}.py"
        cfg_path = self.repo_dir / cfg_rel_path
        save_path = tmp_dir / "outputs"
        proto_dir = save_path / "proto"
        try:
            cfg_content = _build_config_content(
                prompt=prompt,
                save_path=str(save_path),
                model_path=model_path,
                vq_path=vq_path,
                tokenizer_path=tokenizer_path,
                vq_type=vq_type,
                task_type=task_type,
                use_image=use_image,
                aspect_ratio=aspect_ratio,
                hf_device=hf_device,
                vq_device=vq_device,
                classifier_free_guidance=classifier_free_guidance,
                max_new_tokens=max_new_tokens,
                image_area=image_area,
                image_cfg_scale=image_cfg_scale,
                seed=seed,
            )
            cfg_path.write_text(cfg_content, encoding="utf-8")

            last_err: Optional[Exception] = None
            for attempt in range(1, max_retries + 2):
                try:
                    self._run_inference(
                        config_path=cfg_rel_path,
                        timeout_s=timeout_s,
                        python_bin=python_bin,
                        gpu_id=gpu_id,
                        cuda_visible_devices=cuda_visible_devices,
                        use_vllm=use_vllm,
                    )
                    break
                except Exception as e:
                    last_err = e
                    if attempt > max_retries:
                        raise
                    sleep_s = min(30.0, retry_backoff_s * (2 ** (attempt - 1)))
                    sleep_s += random.uniform(0, 0.5)
                    time.sleep(sleep_s)

            if not proto_dir.exists():
                raise RuntimeError(f"Emu3.5 output missing proto directory: {proto_dir}")
            pb_candidates = sorted(proto_dir.glob("*.pb"))
            if not pb_candidates:
                raise RuntimeError(f"Emu3.5 produced no .pb files under {proto_dir}")
            proto_path = pb_candidates[0]

            image_data, src_ext = self._load_first_image(proto_path)
            requested_ext = _normalize_image_ext(image_ext) or src_ext
            if requested_ext == src_ext:
                return GenerationArtifact(data=image_data, extension=src_ext)

            try:
                from PIL import Image as PILImage
            except Exception as e:
                raise ImportError(
                    "Pillow is required to convert image formats. Install it with: pip install pillow"
                ) from e

            image = PILImage.open(io.BytesIO(image_data))
            buf = io.BytesIO()
            image.save(buf, format=_format_for_ext(requested_ext))
            return GenerationArtifact(data=buf.getvalue(), extension=requested_ext)
        finally:
            try:
                if cfg_path.exists():
                    cfg_path.unlink()
            finally:
                shutil.rmtree(tmp_dir, ignore_errors=True)

    def video_generation(self, prompt: str, **kwargs) -> GenerationArtifact:
        del prompt, kwargs
        raise NotImplementedError("Emu35Client only supports image_generation().")
