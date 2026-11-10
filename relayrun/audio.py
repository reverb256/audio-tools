"""Audio load, measure, and PCM helpers. ffmpeg is used for EBU R128; numpy elsewhere."""

from __future__ import annotations

import hashlib
import math
import os
import re
import subprocess
from pathlib import Path

import numpy as np
import soundfile as sf

from relayrun.errors import RelayError

SILENCE_DB = -60.0
SILENCE_LIMIT_S = 0.5
# Sample peak at or above this is "at or near full scale" (-0.1 dBFS).
CLIP_DBFS = -0.1
CLIP_LINEAR = 10 ** (CLIP_DBFS / 20.0)


def ffmpeg_bin() -> str:
    return os.environ.get("RELAY_FFMPEG") or "ffmpeg"


def run_ffmpeg(args: list[str], *, what: str) -> subprocess.CompletedProcess:
    cmd = [ffmpeg_bin(), "-nostdin", "-hide_banner", *args]
    try:
        proc = subprocess.run(cmd, capture_output=True, text=True)
    except FileNotFoundError as exc:
        raise RelayError(
            f"ffmpeg not found ({ffmpeg_bin()}). Install ffmpeg or set RELAY_FFMPEG."
        ) from exc
    if proc.returncode != 0:
        tail = (proc.stderr or proc.stdout or "").strip().splitlines()[-12:]
        detail = "\n".join(tail) if tail else f"exit {proc.returncode}"
        raise RelayError(f"{what} failed:\n{detail}")
    return proc


def load_audio(path: Path) -> tuple[np.ndarray, int]:
    """Read a file as float64, shape (frames, channels)."""
    path = Path(path)
    if not path.is_file():
        raise RelayError(f"audio file not found: {path}", exit_code=2)
    try:
        data, sr = sf.read(str(path), dtype="float64", always_2d=True)
    except Exception as exc:
        raise RelayError(f"could not read {path.name}: {exc}") from exc
    if data.size == 0 or sr <= 0:
        raise RelayError(f"{path.name} has no audio samples")
    return data, int(sr)


def to_stereo(pcm: np.ndarray) -> np.ndarray:
    if pcm.shape[1] == 1:
        return np.repeat(pcm, 2, axis=1)
    if pcm.shape[1] >= 2:
        return pcm[:, :2].copy()
    raise RelayError("audio has no channels")


def peak_dbfs(pcm: np.ndarray) -> float:
    peak = float(np.max(np.abs(pcm))) if pcm.size else 0.0
    if peak <= 0.0:
        return -120.0
    return float(20.0 * math.log10(peak))


def is_clipped(pcm: np.ndarray) -> bool:
    if pcm.size == 0:
        return False
    return bool(np.max(np.abs(pcm)) >= CLIP_LINEAR)


def _frame_rms(pcm: np.ndarray, sr: int, frame_s: float = 0.05) -> tuple[np.ndarray, int]:
    frame = max(1, int(round(frame_s * sr)))
    n = pcm.shape[0]
    usable = (n // frame) * frame
    if usable == 0:
        mono = pcm.astype(np.float64)
        rms = float(np.sqrt(np.mean(mono ** 2))) if mono.size else 0.0
        return np.array([rms]), n
    frames = pcm[:usable].reshape(-1, frame, pcm.shape[1])
    rms = np.sqrt(np.mean(frames.astype(np.float64) ** 2, axis=(1, 2)))
    return rms, frame


def silence_regions(pcm: np.ndarray, sr: int, thresh_db: float = SILENCE_DB) -> dict:
    """Leading, trailing, and longest internal silence, in seconds.

    A frame is silent when its RMS is below ``thresh_db``. The opening run is
    leading, the closing run is trailing, and any other run is internal.
    """
    duration = pcm.shape[0] / float(sr) if sr else 0.0
    rms, frame = _frame_rms(pcm, sr)
    thresh = 10 ** (thresh_db / 20.0)
    quiet = rms < thresh
    runs: list[tuple[int, int]] = []
    i = 0
    n = len(quiet)
    while i < n:
        if not quiet[i]:
            i += 1
            continue
        j = i + 1
        while j < n and quiet[j]:
            j += 1
        runs.append((i, j))
        i = j

    def seconds(a: int, b: int) -> float:
        return (b - a) * frame / float(sr)

    leading = 0.0
    trailing = 0.0
    internal = 0.0
    for a, b in runs:
        length = seconds(a, b)
        if a == 0:
            leading = length
        elif b == n:
            trailing = length
        else:
            internal = max(internal, length)
    return {
        "leading_s": round(leading, 4),
        "trailing_s": round(trailing, 4),
        "internal_s": round(internal, 4),
        "duration_s": round(duration, 4),
    }


def trim_edges(pcm: np.ndarray, sr: int, thresh_db: float = SILENCE_DB,
               pad_s: float = 0.25) -> tuple[np.ndarray, int, int]:
    """Drop leading and trailing silence below ``thresh_db``, keeping ``pad_s``.

    Interior samples are not modified. Returns ``(trimmed, start, end)`` where
    ``start`` and ``end`` are frame indexes into ``pcm`` (``end`` is exclusive).
    """
    n = pcm.shape[0]
    if n == 0:
        return pcm, 0, 0
    rms, frame = _frame_rms(pcm, sr, frame_s=0.01)
    thresh = 10 ** (thresh_db / 20.0)
    loud = rms >= thresh
    if not np.any(loud):
        raise RelayError("audio is silent below -60 dBFS; nothing to master")
    first = int(np.argmax(loud))
    last = int(len(loud) - 1 - np.argmax(loud[::-1]))
    pad = int(round(pad_s * sr))
    start = max(0, first * frame - pad)
    if last >= len(loud) - 1:
        end = n
    else:
        end = min(n, (last + 1) * frame + pad)
    if end <= start:
        raise RelayError("trim removed the whole file")
    return pcm[start:end].copy(), start, end


def integrated_lufs(pcm: np.ndarray, sr: int) -> float:
    """BS.1770 integrated loudness via pyloudnorm. ``-inf`` if the block is silent."""
    import pyloudnorm as pyln

    data = np.asarray(pcm, dtype=np.float64)
    if data.ndim == 2 and data.shape[1] == 1:
        data = data[:, 0]
    meter = pyln.Meter(int(sr))
    try:
        value = float(meter.integrated_loudness(data))
    except Exception as exc:
        raise RelayError(f"integrated loudness failed: {exc}") from exc
    return value


def true_peak_dbtp(pcm: np.ndarray, sr: int, oversample: int = 4) -> float:
    """Oversampled true peak (polyphase upsample), in dBTP."""
    from scipy.signal import resample_poly

    if pcm.size == 0:
        return -120.0
    peak = 0.0
    for ch in range(pcm.shape[1]):
        up = resample_poly(pcm[:, ch].astype(np.float64), int(oversample), 1)
        peak = max(peak, float(np.max(np.abs(up))) if up.size else 0.0)
    if peak <= 0.0:
        return -120.0
    return float(20.0 * math.log10(peak))


def parse_ebur128(stderr: str) -> tuple[float, float]:
    """Return ``(integrated_lufs, true_peak_dbtp)`` from ffmpeg ebur128 stderr."""
    idx = stderr.rfind("Summary:")
    block = stderr[idx:] if idx >= 0 else stderr
    i_vals = re.findall(r"^\s*I:\s+(-?\d+(?:\.\d+)?)\s+LUFS", block, re.M)
    p_vals = re.findall(r"Peak:\s+(-?\d+(?:\.\d+)?)\s+dBFS", block)
    if not i_vals or not p_vals:
        raise RelayError("could not parse ffmpeg ebur128 loudness or true peak")
    return float(i_vals[-1]), float(p_vals[-1])


def measure_ebur128(path: Path) -> tuple[float, float]:
    proc = run_ffmpeg(
        ["-i", str(path), "-af", "ebur128=peak=true", "-f", "null", "-"],
        what=f"ebur128 measure of {Path(path).name}",
    )
    return parse_ebur128(proc.stderr or "")


def file_md5(path: Path) -> str:
    digest = hashlib.md5()
    with Path(path).open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def pcm_md5(pcm: np.ndarray) -> str:
    """MD5 of a contiguous PCM buffer (the sample payload, not a WAV header)."""
    arr = np.ascontiguousarray(pcm)
    return hashlib.md5(arr.tobytes()).hexdigest()


def read_pcm24(path: Path) -> tuple[np.ndarray, int]:
    data, sr = sf.read(str(path), dtype="int32", always_2d=True)
    return data, int(sr)


def write_pcm24(path: Path, pcm: np.ndarray, sr: int) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    sf.write(str(path), np.ascontiguousarray(pcm), int(sr), subtype="PCM_24")


def probe_format(path: Path) -> dict:
    info = sf.info(str(path))
    return {
        "sample_rate": int(info.samplerate),
        "channels": int(info.channels),
        "subtype": info.subtype,
        "frames": int(info.frames),
        "duration_s": float(info.duration),
    }
