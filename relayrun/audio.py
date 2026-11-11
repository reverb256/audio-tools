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
# RMS window for silence. One sample, including a zero crossing, cannot fill it.
SILENCE_FRAME_S = 0.02
# Non-silent audio shorter than this cannot bound an internal dropout.
# A click or an isolated sample stays with the surrounding edge silence.
SILENCE_CONTEXT_S = 0.05
# Sample peak at or above this is a near-full-scale warning, not clipping.
PEAK_WARN_DBFS = -0.1
PEAK_WARN_LINEAR = 10 ** (PEAK_WARN_DBFS / 20.0)
# Same threshold under the previous names. The value is unchanged.
CLIP_DBFS = PEAK_WARN_DBFS
CLIP_LINEAR = PEAK_WARN_LINEAR
# Flat-top evidence must be longer than the same-code crest of a full-scale
# sinusoid at this frequency for the file's integer step. Float formats have
# no integer crest; they use the floor below.
FLAT_TOP_REF_HZ = 20.0
FLAT_TOP_FLOOR = 3
# 4x polyphase peak on a low-frequency sine moves by only a few thousandths of
# a dB. A larger gap is reported as an intersample overshoot. It is not clipping
# and it is not the release true-peak gate.
INTERSAMPLE_DISTINCT_DB = 0.05
QA_OVERSAMPLE = 4

_PCM_STEP = {
    "PCM_U8": 1.0 / 128.0,
    "PCM_16": 1.0 / 32768.0,
    "PCM_24": 1.0 / 8388608.0,
    "PCM_32": 1.0 / 2147483648.0,
}


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


def quantization_step(subtype: str) -> float:
    """Full-scale step of integer PCM after soundfile's float decode.

    Float formats return 0: a flat-top is exact sample equality, and there is
    no multi-sample quantised crest to exclude.
    """
    return _PCM_STEP.get(subtype or "", 0.0)


def flat_top_min_samples(sample_rate: int, subtype: str) -> int:
    """Minimum equal-sample run that is longer than a 20 Hz full-scale crest.

    For integer PCM the crest of an unclipped sinusoid at ``FLAT_TOP_REF_HZ``
    can sit on one code for several samples. A run must be longer than that
    dwell (and at least ``FLAT_TOP_FLOOR`` samples) before it is a flat-top.
    Float and unknown subtypes use the floor only.
    """
    step = quantization_step(subtype)
    if step <= 0.0 or sample_rate <= 0:
        return FLAT_TOP_FLOOR
    dwell = sample_rate * math.sqrt(2.0 * step) / (math.pi * FLAT_TOP_REF_HZ)
    return max(FLAT_TOP_FLOOR, math.ceil(dwell) + 1)


def _longest_equal_run(channel: np.ndarray, level: float) -> tuple[int, float | None]:
    """Longest run of identical samples whose absolute value is >= ``level``."""
    x = np.ascontiguousarray(channel, dtype=np.float64)
    n = int(x.size)
    if n == 0:
        return 0, None
    hot = np.abs(x) >= level
    if not np.any(hot):
        return 0, None
    boundary = np.empty(n, dtype=bool)
    boundary[0] = True
    boundary[1:] = ~((x[1:] == x[:-1]) & hot[1:] & hot[:-1])
    starts = np.flatnonzero(boundary)
    ends = np.append(starts[1:], n)
    lengths = ends - starts
    hot_start = hot[starts]
    if not np.any(hot_start):
        return 0, None
    scored = np.where(hot_start, lengths, 0)
    idx = int(np.argmax(scored))
    best = int(scored[idx])
    if best <= 0:
        return 0, None
    return best, float(x[int(starts[idx])])


def clipping_report(pcm: np.ndarray, sr: int, subtype: str = "") -> dict:
    """Sample-peak warning, flat-top clipping, and 4x oversampled peak.

    Units: sample peak is dBFS re 1.0. The oversampled peak is dBTP from a
    4x polyphase upsample (the same factor the release gate compares, but QA
    does not apply the -1.0 dBTP limit). Clipping is a flat-top on one
    channel: at least ``flat_top_min_samples`` consecutive decoded samples
    with the same value, each at or above -0.1 dBFS. A single full-scale
    sample, a smooth crest, and an intersample overshoot are not clipping.
    """
    data = np.asarray(pcm, dtype=np.float64)
    if data.ndim == 1:
        data = data[:, None]
    channels = int(data.shape[1]) if data.ndim == 2 else 0
    subtype = subtype or ""
    min_run = flat_top_min_samples(int(sr), subtype)
    best_len = 0
    best_value = None
    best_channel = None
    for ch in range(channels):
        length, value = _longest_equal_run(data[:, ch], PEAK_WARN_LINEAR)
        if length > best_len:
            best_len = length
            best_value = value
            best_channel = ch
    sample_peak = peak_dbfs(data)
    oversampled = true_peak_dbtp(data, int(sr), oversample=QA_OVERSAMPLE) if sr else -120.0
    overshoot = float(oversampled - sample_peak)
    value_db = None
    if best_value is not None and abs(best_value) > 0.0:
        value_db = float(20.0 * math.log10(abs(best_value)))
    duration_s = (best_len / float(sr)) if sr and best_len else 0.0
    return {
        "sample_peak_dbfs": round(sample_peak, 3),
        "near_full_scale": bool(np.isfinite(sample_peak) and sample_peak >= PEAK_WARN_DBFS),
        "clipped": bool(best_len >= min_run and best_len > 0),
        "flat_top_samples": int(best_len),
        "flat_top_min_samples": int(min_run),
        "flat_top_channel": best_channel,
        "flat_top_value_dbfs": None if value_db is None else round(value_db, 3),
        "flat_top_duration_s": round(duration_s, 6),
        "channels": channels,
        "sample_rate": int(sr),
        "subtype": subtype,
        "quantization_step": quantization_step(subtype),
        "crest_reference_hz": FLAT_TOP_REF_HZ,
        "oversample": QA_OVERSAMPLE,
        "oversampled_peak_dbtp": round(float(oversampled), 3),
        "intersample_overshoot_db": round(overshoot, 3),
        "intersample_distinct": bool(overshoot > INTERSAMPLE_DISTINCT_DB),
    }


def is_clipped(pcm: np.ndarray, sr: int | None = None, subtype: str | None = None) -> bool:
    """True only when ``clipping_report`` finds a flat-top.

    A sample peak at or above -0.1 dBFS is not enough. ``subtype`` selects the
    integer crest guard; unknown and float formats use a 3-sample floor.
    """
    if pcm.size == 0:
        return False
    return bool(clipping_report(pcm, int(sr or 0), subtype or "")["clipped"])


def _frame_spans(pcm: np.ndarray, sr: int, frame_s: float) -> tuple[list[float], list[tuple[int, int]]]:
    """Max-channel RMS and ``[start, end)`` for each frame, including a short tail.

    A frame is one channel's energy. The returned level is the loudest channel,
    so a silent channel does not mark the frame silent while another channel
    still has signal. The tail is measured on its own and is not padded with
    zeros.
    """
    frame = max(1, int(round(frame_s * sr)))
    n = int(pcm.shape[0])
    levels: list[float] = []
    spans: list[tuple[int, int]] = []
    full = (n // frame) * frame
    if full:
        blocks = np.asarray(pcm[:full], dtype=np.float64).reshape(-1, frame, pcm.shape[1])
        per = np.sqrt(np.mean(blocks * blocks, axis=1))
        loudest = np.max(per, axis=1)
        for i, value in enumerate(loudest):
            levels.append(float(value))
            spans.append((i * frame, (i + 1) * frame))
    if full < n:
        tail = np.asarray(pcm[full:], dtype=np.float64)
        per = np.sqrt(np.mean(tail * tail, axis=0))
        levels.append(float(np.max(per)) if per.size else 0.0)
        spans.append((full, n))
    return levels, spans


def _silence_runs(flags: list[bool], spans: list[tuple[int, int]]) -> list[dict]:
    runs: list[dict] = []
    i = 0
    n = len(flags)
    while i < n:
        j = i + 1
        while j < n and flags[j] == flags[i]:
            j += 1
        runs.append({
            "silent": bool(flags[i]),
            "start": spans[i][0],
            "end": spans[j - 1][1],
        })
        i = j
    return runs


def _edge_silence(runs: list[dict], n: int, sr: int, context_s: float
                  ) -> tuple[float, float, float, tuple[float, float] | None]:
    """Split silence into leading, trailing, and internal seconds.

    Internal silence is the longest gap between two non-silent runs that are
    each at least ``context_s``. Anything before the first such run is leading,
    and anything after the last is trailing. A sub-context blip does not split
    those regions and does not count as a boundary. With no context-length
    audio, a mostly silent file is all leading silence (a click is not an
    internal dropout). A short, mostly-audio file reports only its edge
    silence.
    """
    anchors = [
        run for run in runs
        if not run["silent"] and (run["end"] - run["start"]) / float(sr) >= context_s
    ]
    if anchors:
        leading = anchors[0]["start"] / float(sr)
        trailing = (n - anchors[-1]["end"]) / float(sr)
        internal = 0.0
        span = None
        for left, right in zip(anchors, anchors[1:]):
            gap = (right["start"] - left["end"]) / float(sr)
            if gap > internal:
                internal = gap
                span = (left["end"] / float(sr), right["start"] / float(sr))
        return leading, trailing, internal, span

    nonsilent = sum((run["end"] - run["start"]) for run in runs if not run["silent"])
    if nonsilent == 0 or nonsilent * 2 < n:
        return (n / float(sr) if n else 0.0), 0.0, 0.0, None
    leading = 0.0
    trailing = 0.0
    if runs and runs[0]["silent"]:
        leading = (runs[0]["end"] - runs[0]["start"]) / float(sr)
    if runs and runs[-1]["silent"] and runs[-1]["start"] != 0:
        trailing = (runs[-1]["end"] - runs[-1]["start"]) / float(sr)
    return leading, trailing, 0.0, None


def silence_regions(pcm: np.ndarray, sr: int, thresh_db: float = SILENCE_DB,
                    frame_s: float = SILENCE_FRAME_S,
                    context_s: float = SILENCE_CONTEXT_S) -> dict:
    """Leading, trailing, and longest internal silence, in seconds.

    A frame is silent when every channel's RMS is strictly below ``thresh_db``
    (the stored level is the loudest channel). Leading silence touches the
    start and trailing silence touches the end. Internal silence is a gap
    between two stretches of non-silent audio that each last at least
    ``context_s``. Durations are seconds of audio, not a count of near-zero
    samples.
    """
    data = np.asarray(pcm, dtype=np.float64)
    if data.ndim == 1:
        data = data[:, None]
    n = int(data.shape[0]) if data.ndim == 2 else 0
    duration = n / float(sr) if sr else 0.0
    empty = {
        "leading_s": 0.0,
        "trailing_s": 0.0,
        "internal_s": 0.0,
        "duration_s": round(duration, 4),
        "internal_start_s": None,
        "internal_end_s": None,
        "threshold_dbfs": float(thresh_db),
        "frame_s": float(frame_s),
        "context_s": float(context_s),
        "channel_rule": "loudest-channel RMS",
    }
    if n == 0 or sr <= 0:
        return empty
    levels, spans = _frame_spans(data, int(sr), frame_s)
    thresh = 10 ** (float(thresh_db) / 20.0)
    flags = [level < thresh for level in levels]
    runs = _silence_runs(flags, spans)
    leading, trailing, internal, span = _edge_silence(runs, n, int(sr), float(context_s))
    return {
        "leading_s": round(leading, 4),
        "trailing_s": round(trailing, 4),
        "internal_s": round(internal, 4),
        "duration_s": round(duration, 4),
        "internal_start_s": None if span is None else round(span[0], 4),
        "internal_end_s": None if span is None else round(span[1], 4),
        "threshold_dbfs": float(thresh_db),
        "frame_s": float(frame_s),
        "context_s": float(context_s),
        "channel_rule": "loudest-channel RMS",
    }


def _frame_rms(pcm: np.ndarray, sr: int, frame_s: float = 0.05) -> tuple[np.ndarray, int]:
    """Mean-square frames for edge trim. The short tail is left on the file.

    Silence classification does not use this. Trim keeps its previous 10 ms
    grid so a master does not move its edit points.
    """
    frame = max(1, int(round(frame_s * sr)))
    n = pcm.shape[0]
    usable = (n // frame) * frame
    if usable == 0:
        mono = np.asarray(pcm, dtype=np.float64)
        rms = float(np.sqrt(np.mean(mono ** 2))) if mono.size else 0.0
        return np.array([rms]), n
    frames = np.asarray(pcm[:usable], dtype=np.float64).reshape(-1, frame, pcm.shape[1])
    rms = np.sqrt(np.mean(frames ** 2, axis=(1, 2)))
    return rms, frame


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
