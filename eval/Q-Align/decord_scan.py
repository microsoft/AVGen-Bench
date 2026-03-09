import os
import re
import sys
from pathlib import Path

MMCO = re.compile(r"mmco:\s*unref short failure", re.I)

def iter_mp4(root: str):
    rootp = Path(root)
    for p in rootp.rglob("*.mp4"):
        yield str(p)

def capture_fd2(func):
    """
    Capture anything written to OS-level stderr (fd=2) during func().
    Returns: (ok: bool, captured_text: str, exc: Exception|None)
    """
    import tempfile

    # Make a temp file to store fd=2 output
    tmp = tempfile.NamedTemporaryFile(delete=False)
    tmp_path = tmp.name
    tmp.close()

    old_fd2 = os.dup(2)
    exc = None
    ok = True
    try:
        with open(tmp_path, "wb") as f:
            os.dup2(f.fileno(), 2)  # redirect fd=2 -> temp file
            try:
                func()
            except Exception as e:
                ok = False
                exc = e
    finally:
        os.dup2(old_fd2, 2)  # restore stderr
        os.close(old_fd2)

    try:
        data = Path(tmp_path).read_bytes()
    finally:
        try:
            os.unlink(tmp_path)
        except OSError:
            pass

    # Best-effort decode
    text = data.decode("utf-8", errors="replace")
    return ok, text, exc

def make_loader(video_file: str):
    def _run():
        from decord import VideoReader
        vr = VideoReader(video_file)

        fps = vr.get_avg_fps()
        if not fps or fps <= 0:
            # force a read to trigger decoder messages
            _ = vr[0]
            return

        # mirror your logic (1 fps sampling)
        n = len(vr)
        frame_indices = [int(fps * i) for i in range(int(n / fps))]
        if not frame_indices:
            _ = vr[0]
            return
        _ = vr.get_batch(frame_indices).asnumpy()
    return _run

def main():
    root = sys.argv[1] if len(sys.argv) > 1 else "/path/to/video_generation/veo3.1_fast"
    out_path = sys.argv[2] if len(sys.argv) > 2 else "mmco_bad_decord.txt"

    bad = []
    scanned = 0

    for f in iter_mp4(root):
        scanned += 1
        if scanned % 50 == 0:
            print(f"scanned {scanned} ...", file=sys.stderr)

        ok, err, exc = capture_fd2(make_loader(f))

        # decord/ffmpeg messages sometimes go to stderr even on success
        if MMCO.search(err):
            bad.append((f, "MMCO", err.strip().splitlines()[-1] if err.strip() else ""))
            print(f"MMCO_BAD: {f}")

        # Optional: also record videos that crash/fail during direct loading.
        # if not ok:
        #     bad.append((f, "EXCEPTION", str(exc)))
        #     print(f"EXC_BAD: {f}  ({exc})")

    with open(out_path, "w", encoding="utf-8") as w:
        for f, tag, msg in bad:
            w.write(f"{tag}\t{f}\t{msg}\n")

    print(f"done. scanned={scanned}, mmco={len(bad)}. wrote {out_path}")

if __name__ == "__main__":
    main()
