"""Frame grid, WAV and ffmpeg helpers."""
import io, math, shutil, subprocess, tempfile, wave
from pathlib import Path
from typing import Optional


def align_frames(n: int) -> int:
    """sd.cpp aligns H3 frame counts upward onto the 17k+5 grid (min 5)."""
    n = max(5, int(n))
    return 17 * math.ceil((n - 5) / 17) + 5


def floor_aligned(n: int) -> int:
    """Largest 17k+5 value <= n (min 5)."""
    n = max(5, int(n))
    return 17 * ((n - 5) // 17) + 5


def clamp_side(width: int, height: int, max_side: int):
    """Keeps aspect ratio, multiples of 32. Returns (w, h, note|None)."""
    width, height = max(32, width // 32 * 32), max(32, height // 32 * 32)
    side = max(width, height)
    if side <= max_side:
        return width, height, None
    s = max_side / side
    w, h = max(32, int(width * s) // 32 * 32), max(32, int(height * s) // 32 * 32)
    return w, h, f"resolution clamped to {w}x{h}"


def wav_seconds(b: bytes) -> Optional[float]:
    try:
        with wave.open(io.BytesIO(b)) as w:
            return w.getnframes() / w.getframerate()
    except Exception:
        return None


def wav_truncate(b: bytes, seconds: float) -> bytes:
    with wave.open(io.BytesIO(b)) as r:
        n = min(r.getnframes(), int(seconds * r.getframerate()))
        params, data = r.getparams(), r.readframes(n)
    out = io.BytesIO()
    with wave.open(out, "wb") as w:
        w.setparams(params); w.writeframes(data)
    return out.getvalue()


def ffmpeg_exe() -> Optional[str]:
    exe = shutil.which("ffmpeg")
    if exe:
        return exe
    try:
        import imageio_ffmpeg
        return imageio_ffmpeg.get_ffmpeg_exe()
    except Exception:
        return None


def to_wav_bytes(data: bytes, ext: str) -> bytes:
    """sd.cpp only reads WAV. Convert anything else (mp3, m4a, flac...) with ffmpeg."""
    if ext.lower() == ".wav" and wav_seconds(data) is not None:
        return data
    ff = ffmpeg_exe()
    if not ff:
        raise RuntimeError(f"audio is {ext}, sd.cpp needs WAV and ffmpeg is unavailable (pip install imageio-ffmpeg)")
    with tempfile.TemporaryDirectory() as d:
        src, dst = Path(d) / f"in{ext or '.bin'}", Path(d) / "out.wav"
        src.write_bytes(data)
        subprocess.run([ff, "-y", "-loglevel", "error", "-i", str(src), "-acodec", "pcm_s16le", str(dst)], check=True)
        return dst.read_bytes()


def to_mp4(path: Path) -> Path:
    """Normalize whatever container sd-cli wrote (mp4/avi/webm...) to mp4 if ffmpeg exists."""
    if path.suffix.lower() == ".mp4":
        return path
    ff = ffmpeg_exe()
    if not ff:
        return path
    dst = path.with_suffix(".mp4")
    subprocess.run([ff, "-y", "-loglevel", "error", "-i", str(path), "-c:v", "libx264", "-pix_fmt", "yuv420p",
                    "-c:a", "aac", str(dst)], check=True)
    return dst
