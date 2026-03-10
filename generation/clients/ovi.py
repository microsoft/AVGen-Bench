import json
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any, Dict, List, Optional

from .base import BaseGenerationClient, GenerationArtifact


def _parse_size_to_h_w(size: str) -> List[int]:
    s = str(size).strip().lower().replace("*", "x")
    parts = [p.strip() for p in s.split("x") if p.strip()]
    if len(parts) != 2:
        raise ValueError(f"Invalid size '{size}', expected format like 1280x720")
    width = int(parts[0])
    height = int(parts[1])
    return [height, width]


class OviClient(BaseGenerationClient):
    """
    Local Ovi integration by invoking Ovi's official inference script:
      python inference.py --config-file <temp_config.yaml>

    Requirements:
    - Ovi repository available locally (default: /tmp/Ovi, override via OVI_REPO_DIR).
    - Ovi checkpoints downloaded (default: <repo>/ckpts, override via OVI_CKPT_DIR).
    """

    def __init__(self) -> None:
        default_repo_dir = Path(__file__).resolve().parents[2] / "third_party" / "Ovi"
        repo_dir = os.environ.get("OVI_REPO_DIR", str(default_repo_dir))
        self.repo_dir = Path(repo_dir).expanduser().resolve()
        self.inference_script = self.repo_dir / "inference.py"
        if not self.inference_script.exists():
            raise FileNotFoundError(
                f"Ovi inference script not found: {self.inference_script}. "
                "Please vendor Ovi into third_party/Ovi or set OVI_REPO_DIR."
            )

    def _build_config(
        self,
        prompt: str,
        output_dir: Path,
        ckpt_dir: Optional[str],
        model_name: str,
        mode: str,
        size: str,
        sample_steps: int,
        solver_name: str,
        shift: float,
        seed: int,
        audio_guidance_scale: float,
        video_guidance_scale: float,
        slg_layer: int,
        sp_size: int,
        cpu_offload: bool,
        fp8: bool,
        video_negative_prompt: str,
        audio_negative_prompt: str,
    ) -> Dict[str, Any]:
        resolved_ckpt_dir = (
            Path(ckpt_dir).expanduser().resolve()
            if ckpt_dir
            else Path(os.environ.get("OVI_CKPT_DIR", str(self.repo_dir / "ckpts"))).expanduser().resolve()
        )
        return {
            "ckpt_dir": str(resolved_ckpt_dir),
            "output_dir": str(output_dir),
            "sample_steps": int(sample_steps),
            "solver_name": solver_name,
            "model_name": model_name,
            "shift": float(shift),
            "sp_size": int(sp_size),
            "audio_guidance_scale": float(audio_guidance_scale),
            "video_guidance_scale": float(video_guidance_scale),
            "mode": mode,
            "fp8": bool(fp8),
            "cpu_offload": bool(cpu_offload),
            "seed": int(seed),
            "video_negative_prompt": video_negative_prompt,
            "audio_negative_prompt": audio_negative_prompt,
            "video_frame_height_width": _parse_size_to_h_w(size),
            "text_prompt": prompt,
            "slg_layer": int(slg_layer),
            "each_example_n_times": 1,
        }

    def _run_inference(
        self,
        config_path: Path,
        timeout_s: int,
        python_bin: Optional[str],
        torchrun_nproc: int,
        gpu_id: Optional[str],
        cuda_visible_devices: Optional[str],
    ) -> None:
        py = python_bin or os.environ.get("OVI_PYTHON_BIN") or sys.executable
        if torchrun_nproc and int(torchrun_nproc) > 1:
            cmd = [
                "torchrun",
                "--nnodes",
                "1",
                "--nproc_per_node",
                str(int(torchrun_nproc)),
                str(self.inference_script),
                "--config-file",
                str(config_path),
            ]
        else:
            cmd = [py, str(self.inference_script), "--config-file", str(config_path)]

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
                f"Ovi inference failed (exit={proc.returncode}).\n"
                f"stdout_tail:\n{stdout_tail}\n"
                f"stderr_tail:\n{stderr_tail}"
            )

    @staticmethod
    def _pick_latest_mp4(output_dir: Path) -> Path:
        mp4s = sorted(output_dir.rglob("*.mp4"), key=lambda p: p.stat().st_mtime, reverse=True)
        if not mp4s:
            raise RuntimeError(f"Ovi produced no .mp4 under {output_dir}")
        return mp4s[0]

    def video_generation(
        self,
        prompt: str,
        ckpt_dir: Optional[str] = None,
        model_name: str = "960x960_10s",
        mode: str = "t2v",
        size: str = "1280x720",
        sample_steps: int = 50,
        solver_name: str = "unipc",
        shift: float = 5.0,
        seed: int = 100,
        audio_guidance_scale: float = 3.0,
        video_guidance_scale: float = 4.0,
        slg_layer: int = 11,
        sp_size: int = 1,
        cpu_offload: bool = False,
        fp8: bool = False,
        video_negative_prompt: str = "jitter, bad hands, blur, distortion",
        audio_negative_prompt: str = "robotic, muffled, echo, distorted",
        timeout_s: int = 7200,
        python_bin: Optional[str] = None,
        torchrun_nproc: int = 1,
        gpu_id: Optional[str] = None,
        cuda_visible_devices: Optional[str] = None,
        **kwargs,
    ) -> GenerationArtifact:
        if "gpu_id" in kwargs and gpu_id is None:
            gpu_id = kwargs.pop("gpu_id")
        if "cuda_visible_devices" in kwargs and cuda_visible_devices is None:
            cuda_visible_devices = kwargs.pop("cuda_visible_devices")
        tmp_dir = Path(tempfile.mkdtemp(prefix="ovi_gen_"))
        try:
            output_dir = tmp_dir / "outputs"
            output_dir.mkdir(parents=True, exist_ok=True)
            config = self._build_config(
                prompt=prompt,
                output_dir=output_dir,
                ckpt_dir=ckpt_dir,
                model_name=model_name,
                mode=mode,
                size=size,
                sample_steps=sample_steps,
                solver_name=solver_name,
                shift=shift,
                seed=seed,
                audio_guidance_scale=audio_guidance_scale,
                video_guidance_scale=video_guidance_scale,
                slg_layer=slg_layer,
                sp_size=sp_size,
                cpu_offload=cpu_offload,
                fp8=fp8,
                video_negative_prompt=video_negative_prompt,
                audio_negative_prompt=audio_negative_prompt,
            )

            config_path = tmp_dir / "inference_ovi_config.yaml"
            # YAML parser accepts JSON; avoids adding extra dependency.
            config_path.write_text(json.dumps(config, ensure_ascii=False, indent=2), encoding="utf-8")

            self._run_inference(
                config_path=config_path,
                timeout_s=timeout_s,
                python_bin=python_bin,
                torchrun_nproc=torchrun_nproc,
                gpu_id=gpu_id,
                cuda_visible_devices=cuda_visible_devices,
            )
            out_mp4 = self._pick_latest_mp4(output_dir)
            data = out_mp4.read_bytes()
            if len(data) < 1024:
                raise RuntimeError(f"Ovi output file is too small: {out_mp4}")
            return GenerationArtifact(data=data, extension=".mp4")
        finally:
            shutil.rmtree(tmp_dir, ignore_errors=True)
