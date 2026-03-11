import errno
import os
import pty
import select
import socket
import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from typing import Optional

from .base import BaseGenerationClient, GenerationArtifact


class MovaClient(BaseGenerationClient):
    """
    Local MOVA integration by invoking MOVA's official inference script:
      torchrun --nproc_per_node=<cp_size> scripts/inference_single.py ...

    Requirements:
    - MOVA repo available locally (default: third_party/MOVA, override via MOVA_REPO_DIR).
    - ckpt_path points to downloaded model weights (e.g., OpenMOSS-Team/MOVA-360p or MOVA-720p).
    - ref_path is a local image path (first frame).
    """

    def __init__(self, repo_dir: Optional[str] = None) -> None:
        default_repo_dir = Path(__file__).resolve().parents[2] / "third_party" / "MOVA"
        repo_dir = repo_dir or os.environ.get("MOVA_REPO_DIR") or str(default_repo_dir)
        self.repo_dir = Path(repo_dir).expanduser().resolve()
        self.inference_script = self.repo_dir / "scripts" / "inference_single.py"
        if not self.inference_script.exists():
            raise FileNotFoundError(
                f"MOVA inference script not found: {self.inference_script}. "
                "Please clone MOVA into third_party/MOVA or set MOVA_REPO_DIR."
            )

    def video_generation(
        self,
        prompt: str,
        ckpt_path: str,
        ref_path: str,
        output_ext: str = ".mp4",
        negative_prompt: Optional[str] = None,
        num_frames: int = 193,
        fps: float = 24.0,
        height: int = 720,
        width: int = 1280,
        seed: int = 42,
        num_inference_steps: int = 50,
        cfg_scale: float = 5.0,
        sigma_shift: float = 5.0,
        cp_size: int = 1,
        attn_type: str = "fa",
        offload: str = "none",
        offload_to_disk_path: Optional[str] = None,
        remove_video_dit: bool = False,
        timeout_s: int = 7200,
        python_bin: Optional[str] = None,
        torchrun_bin: Optional[str] = None,
        gpu_id: Optional[int] = None,
        cuda_visible_devices: Optional[str] = None,
        **kwargs,
    ) -> GenerationArtifact:
        del kwargs

        if not ckpt_path:
            raise ValueError("mova ckpt_path is required.")
        if not ref_path:
            raise ValueError("mova ref_path is required (first frame image path).")

        ref_p = Path(ref_path).expanduser()
        if not ref_p.exists():
            raise FileNotFoundError(f"ref_path not found: {ref_p}")

        tmp_dir = Path(tempfile.mkdtemp(prefix="mova_gen_"))
        try:
            out_path = tmp_dir / ("mova_output" + (output_ext if output_ext.startswith(".") else "." + output_ext))

            cp = int(cp_size) if cp_size else 1
            if cp < 1:
                raise ValueError(f"cp_size must be >= 1, got {cp_size}")

            script_args = [
                str(self.inference_script),
                "--ckpt_path",
                str(Path(ckpt_path).expanduser()),
                "--cp_size",
                str(cp),
                "--height",
                str(int(height)),
                "--width",
                str(int(width)),
                "--prompt",
                prompt,
                "--ref_path",
                str(ref_p),
                "--output_path",
                str(out_path),
                "--seed",
                str(int(seed)),
                "--num_frames",
                str(int(num_frames)),
                "--fps",
                str(float(fps)),
                "--num_inference_steps",
                str(int(num_inference_steps)),
                "--cfg_scale",
                str(float(cfg_scale)),
                "--sigma_shift",
                str(float(sigma_shift)),
                "--attn_type",
                str(attn_type),
                "--offload",
                str(offload),
            ]
            if negative_prompt:
                script_args.extend(["--negative_prompt", negative_prompt])
            if offload_to_disk_path:
                script_args.extend(["--offload_to_disk_path", str(offload_to_disk_path)])
            if remove_video_dit:
                script_args.append("--remove_video_dit")

            py = python_bin or os.environ.get("MOVA_PYTHON_BIN") or sys.executable
            if cp == 1:
                cmd = [py, *script_args]
            else:
                # Use torchrun for multi-process context parallel only.
                torchrun = torchrun_bin or os.environ.get("MOVA_TORCHRUN_BIN") or "torchrun"
                cmd = [
                    torchrun,
                    "--nnodes",
                    "1",
                    "--nproc_per_node",
                    str(cp),
                ]
                if torchrun == "python" or torchrun.endswith("python") or torchrun.endswith("python3"):
                    cmd.extend(["-m", "torch.distributed.run", "--nproc_per_node", str(cp)])
                cmd.extend(script_args)

            env = os.environ.copy()
            master_addr, master_port = self._reserve_master_endpoint()
            env.setdefault("MASTER_ADDR", master_addr)
            env["MASTER_PORT"] = str(master_port)
            if cp == 1:
                env["RANK"] = "0"
                env["WORLD_SIZE"] = "1"
                env["LOCAL_RANK"] = "0"
            # For single-GPU inference, allow selecting a single GPU.
            # For multi-process (cp_size>1), users should set CUDA_VISIBLE_DEVICES explicitly.
            if cuda_visible_devices is not None:
                env["CUDA_VISIBLE_DEVICES"] = str(cuda_visible_devices)
            elif gpu_id is not None and cp == 1:
                env["CUDA_VISIBLE_DEVICES"] = str(gpu_id)

            returncode, output_tail = self._run_command_with_streaming(
                cmd=cmd,
                env=env,
                timeout_s=int(timeout_s),
            )
            if returncode != 0:
                raise RuntimeError(
                    f"MOVA inference failed (exit={returncode}).\n"
                    f"output_tail:\n{output_tail[-3000:]}"
                )

            if not out_path.exists() or out_path.stat().st_size < 1024:
                raise RuntimeError(f"MOVA did not produce a valid output file: {out_path}")
            return GenerationArtifact(data=out_path.read_bytes(), extension=out_path.suffix)
        finally:
            shutil.rmtree(tmp_dir, ignore_errors=True)

    def _reserve_master_endpoint(self) -> tuple[str, int]:
        """
        Pick an available localhost port for torch.distributed rendezvous.
        This avoids EADDRINUSE when multiple MOVA jobs start concurrently.
        """
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
            sock.bind(("127.0.0.1", 0))
            sock.listen(1)
            addr, port = sock.getsockname()
            return str(addr), int(port)

    def _run_command_with_streaming(
        self,
        cmd: list[str],
        env: dict[str, str],
    timeout_s: int,
    ) -> tuple[int, str]:
        """
        Stream subprocess output through a PTY so tqdm-style carriage-return updates
        behave like a real terminal instead of block-buffered pipes.
        """
        master_fd, slave_fd = pty.openpty()
        proc = subprocess.Popen(
            cmd,
            cwd=str(self.repo_dir),
            stdin=subprocess.DEVNULL,
            stdout=slave_fd,
            stderr=slave_fd,
            env=env,
            close_fds=True,
        )
        os.close(slave_fd)

        deadline = time.time() + int(timeout_s)
        tail = bytearray()
        max_tail_bytes = 64 * 1024

        try:
            while True:
                if time.time() > deadline:
                    proc.kill()
                    raise TimeoutError(f"MOVA inference timed out after {timeout_s}s")

                ready, _, _ = select.select([master_fd], [], [], 0.2)
                if not ready:
                    if proc.poll() is not None:
                        return proc.returncode, tail.decode("utf-8", errors="replace")
                    continue

                try:
                    chunk = os.read(master_fd, 4096)
                except OSError as e:
                    # PTY returns EIO on EOF on many Unix systems.
                    if e.errno != errno.EIO:
                        raise
                    chunk = b""

                if chunk:
                    try:
                        sys.stderr.buffer.write(chunk)
                        sys.stderr.buffer.flush()
                    except Exception:
                        pass
                    tail.extend(chunk)
                    if len(tail) > max_tail_bytes:
                        del tail[:-max_tail_bytes]

                if proc.poll() is not None and not chunk:
                    return proc.returncode, tail.decode("utf-8", errors="replace")
        finally:
            try:
                os.close(master_fd)
            except Exception:
                pass
