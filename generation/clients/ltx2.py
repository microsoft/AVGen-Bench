import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Optional

from .base import BaseGenerationClient, GenerationArtifact


def _parse_size(size: str) -> tuple[int, int]:
    normalized = str(size).strip().lower().replace("*", "x")
    parts = [p.strip() for p in normalized.split("x") if p.strip()]
    if len(parts) != 2:
        raise ValueError(f"Invalid size '{size}', expected format WIDTHxHEIGHT")
    width = int(parts[0])
    height = int(parts[1])
    return width, height


class Ltx2Client(BaseGenerationClient):
    """
    Local LTX-2 integration using vendored ltx-core + ltx-pipelines sources.

    Current default path uses the official DistilledPipeline:
      python -m ltx_pipelines.distilled ...
    """

    def __init__(self) -> None:
        default_repo_dir = Path(__file__).resolve().parents[2] / "third_party" / "LTX-2"
        repo_dir = os.environ.get("LTX2_REPO_DIR", str(default_repo_dir))
        self.repo_dir = Path(repo_dir).expanduser().resolve()
        self.ltx_core_src = self.repo_dir / "packages" / "ltx-core" / "src"
        self.ltx_pipelines_src = self.repo_dir / "packages" / "ltx-pipelines" / "src"
        self.distilled_module = self.ltx_pipelines_src / "ltx_pipelines" / "distilled.py"
        if not self.distilled_module.exists():
            raise FileNotFoundError(
                f"LTX-2 distilled pipeline not found: {self.distilled_module}. "
                "Please vendor LTX-2 into third_party/LTX-2 or set LTX2_REPO_DIR."
            )

    def _resolve_path(
        self,
        explicit: Optional[str],
        env_name: str,
        default_relative: Optional[str] = None,
    ) -> Optional[str]:
        if explicit:
            return str(Path(explicit).expanduser().resolve())
        env_value = os.environ.get(env_name)
        if env_value:
            return str(Path(env_value).expanduser().resolve())
        models_dir = os.environ.get("LTX2_MODELS_DIR")
        if models_dir and default_relative:
            return str((Path(models_dir).expanduser().resolve() / default_relative).resolve())
        return None

    def _pythonpath(self) -> str:
        paths = [str(self.ltx_core_src), str(self.ltx_pipelines_src)]
        existing = os.environ.get("PYTHONPATH")
        if existing:
            paths.append(existing)
        return os.pathsep.join(paths)

    def video_generation(
        self,
        prompt: str,
        pipeline: str = "distilled",
        distilled_checkpoint_path: Optional[str] = None,
        spatial_upsampler_path: Optional[str] = None,
        gemma_root: Optional[str] = None,
        size: str = "1280x704",
        seed: int = 100,
        num_frames: int = 241,
        frame_rate: float = 24.0,
        quantization: Optional[str] = None,
        enhance_prompt: bool = False,
        timeout_s: int = 7200,
        python_bin: Optional[str] = None,
        gpu_id: Optional[int] = None,
        cuda_visible_devices: Optional[str] = None,
        **kwargs,
    ) -> GenerationArtifact:
        if pipeline != "distilled":
            raise ValueError("Only pipeline='distilled' is currently supported for provider=ltx2.")
        del kwargs

        ckpt = self._resolve_path(
            distilled_checkpoint_path,
            "LTX2_DISTILLED_CHECKPOINT_PATH",
            "ltx-2.3-22b-distilled.safetensors",
        )
        upsampler = self._resolve_path(
            spatial_upsampler_path,
            "LTX2_SPATIAL_UPSAMPLER_PATH",
            "ltx-2.3-spatial-upscaler-x2-1.0.safetensors",
        )
        gemma = self._resolve_path(
            gemma_root,
            "LTX2_GEMMA_ROOT",
            "gemma-3-12b-it-qat-q4_0-unquantized",
        )

        missing = []
        if not ckpt:
            missing.append("distilled checkpoint")
        if not upsampler:
            missing.append("spatial upsampler")
        if not gemma:
            missing.append("gemma root")
        if missing:
            raise ValueError(
                "Missing LTX-2 paths: %s. Provide CLI args or env vars "
                "(LTX2_DISTILLED_CHECKPOINT_PATH, LTX2_SPATIAL_UPSAMPLER_PATH, LTX2_GEMMA_ROOT, optional LTX2_MODELS_DIR)."
                % ", ".join(missing)
            )

        width, height = _parse_size(size)
        if width % 64 != 0 or height % 64 != 0:
            raise ValueError(
                f"LTX-2 distilled two-stage pipeline requires width and height divisible by 64, got {width}x{height}."
            )

        tmp_dir = Path(tempfile.mkdtemp(prefix="ltx2_gen_"))
        try:
            output_path = tmp_dir / "ltx2_output.mp4"
            py = python_bin or os.environ.get("LTX2_PYTHON_BIN") or sys.executable
            cmd = [
                py,
                "-m",
                "ltx_pipelines.distilled",
                "--distilled-checkpoint-path",
                ckpt,
                "--spatial-upsampler-path",
                upsampler,
                "--gemma-root",
                gemma,
                "--prompt",
                prompt,
                "--output-path",
                str(output_path),
                "--seed",
                str(seed),
                "--height",
                str(height),
                "--width",
                str(width),
                "--num-frames",
                str(num_frames),
                "--frame-rate",
                str(frame_rate),
            ]
            if quantization:
                cmd.extend(["--quantization", quantization])
            if enhance_prompt:
                cmd.append("--enhance-prompt")

            env = os.environ.copy()
            env["PYTHONPATH"] = self._pythonpath()
            if cuda_visible_devices:
                env["CUDA_VISIBLE_DEVICES"] = str(cuda_visible_devices)
            elif gpu_id is not None:
                env["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
            proc = subprocess.run(
                cmd,
                cwd=str(self.repo_dir),
                env=env,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                timeout=int(timeout_s),
            )
            if proc.returncode != 0:
                raise RuntimeError(
                    "LTX-2 inference failed (exit=%s).\nstdout_tail:\n%s\nstderr_tail:\n%s"
                    % (proc.returncode, (proc.stdout or "")[-1200:], (proc.stderr or "")[-2000:])
                )
            if not output_path.exists() or output_path.stat().st_size < 1024:
                raise RuntimeError(f"LTX-2 did not produce a valid output file: {output_path}")
            return GenerationArtifact(data=output_path.read_bytes(), extension=".mp4")
        finally:
            shutil.rmtree(tmp_dir, ignore_errors=True)
