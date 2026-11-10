"""Side A release master: -14 LUFS, true peak <= -1.0 dBTP, 44.1 kHz 24-bit.

Chain, per file:

1. Trim leading and trailing silence below -60 dBFS, leaving a 0.25 s pad.
   Interior audio, including interior silence, is not trimmed and is not faded.
2. Two-pass EBU R128 loudnorm, linear mode, integrated -14 LUFS, true-peak
   target -1.5 dBTP (headroom for the resampler).
3. When the input is not already 44.1 kHz, soxr resample to 44.1 kHz with
   triangular dither. Output is stereo 24-bit WAV.
4. Re-measure integrated loudness (ffmpeg ebur128) and true peak. The true
   peak used for the gate is the higher of ffmpeg's oversampled ebur128 peak
   and a 4x polyphase oversampled peak.
5. One corrective gain plus a 4x-oversampled limiter if the result is outside
   the release window. A second miss fails the command.

Release window: integrated loudness -15..-13 LUFS, true peak <= -1.0 dBTP,
leading and trailing silence each under 0.5 s, 44.1 kHz, 24-bit, stereo.
"""

from __future__ import annotations

import json
import math
import re
import shutil
import tempfile
from pathlib import Path

import soundfile as sf

from relayrun.audio import (
    SILENCE_LIMIT_S,
    ffmpeg_bin,
    file_md5,
    load_audio,
    measure_ebur128,
    probe_format,
    run_ffmpeg,
    silence_regions,
    to_stereo,
    trim_edges,
    true_peak_dbtp,
)
from relayrun.errors import RelayError

TARGET_LUFS = -14.0
LUFS_TOLERANCE = 1.0
TRUE_PEAK_MAX = -1.0
LOUDNORM_TP = -1.5
LIMITER_CEILING_DB = -1.2
OUTPUT_SR = 44100


def lufs_window() -> tuple[float, float]:
    return TARGET_LUFS - LUFS_TOLERANCE, TARGET_LUFS + LUFS_TOLERANCE


def judge(lufs: float, true_peak: float, leading_s: float, trailing_s: float,
          sample_rate: int, subtype: str, channels: int,
          target_lufs: float = TARGET_LUFS, tolerance: float = LUFS_TOLERANCE,
          true_peak_max: float = TRUE_PEAK_MAX) -> tuple[bool, list[str]]:
    """Return whether a finished master meets the release window."""
    notes: list[str] = []
    low, high = target_lufs - tolerance, target_lufs + tolerance
    if sample_rate != OUTPUT_SR:
        notes.append(f"sample rate {sample_rate} Hz, expected {OUTPUT_SR}")
    if channels != 2:
        notes.append(f"{channels} channels, expected stereo")
    if subtype != "PCM_24":
        notes.append(f"subtype {subtype}, expected PCM_24")
    if not math.isfinite(lufs) or not low <= lufs <= high:
        shown = "unmeasurable" if not math.isfinite(lufs) else f"{lufs:.2f}"
        notes.append(f"integrated loudness {shown} LUFS outside {low:.1f}..{high:.1f}")
    if not math.isfinite(true_peak) or true_peak > true_peak_max:
        shown = "unmeasurable" if not math.isfinite(true_peak) else f"{true_peak:.2f}"
        notes.append(f"true peak {shown} dBTP exceeds {true_peak_max:.1f}")
    if leading_s >= SILENCE_LIMIT_S:
        notes.append(f"leading silence {leading_s:.3f}s")
    if trailing_s >= SILENCE_LIMIT_S:
        notes.append(f"trailing silence {trailing_s:.3f}s")
    return (not notes), notes


def _finite(value: str, label: str) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError) as exc:
        raise RelayError(f"loudnorm {label} was {value!r}") from exc
    if not math.isfinite(number):
        raise RelayError(f"loudnorm {label} was not finite ({value}); the input may be silent")
    return number


def _loudnorm_measure(path: Path, target_lufs: float) -> dict:
    proc = run_ffmpeg(
        ["-i", str(path), "-af",
         f"loudnorm=I={target_lufs}:TP={LOUDNORM_TP}:LRA=11:print_format=json",
         "-f", "null", "-"],
        what=f"loudnorm measure of {path.name}",
    )
    matches = list(re.finditer(r"\{[^{}]*\}", proc.stderr or "", re.S))
    if not matches:
        raise RelayError(f"loudnorm measure of {path.name} returned no JSON")
    try:
        data = json.loads(matches[-1].group(0))
    except json.JSONDecodeError as exc:
        raise RelayError(f"loudnorm measure of {path.name} returned invalid JSON") from exc
    for key in ("input_i", "input_tp", "input_lra", "input_thresh", "target_offset"):
        data[key] = _finite(data.get(key), key)
    return data


def _apply_loudnorm(src: Path, dst: Path, measured: dict, target_lufs: float, resample: bool) -> str:
    loudnorm = (
        f"loudnorm=I={target_lufs}:TP={LOUDNORM_TP}:LRA=11:"
        f"measured_I={measured['input_i']}:measured_TP={measured['input_tp']}:"
        f"measured_LRA={measured['input_lra']}:measured_thresh={measured['input_thresh']}:"
        f"offset={measured['target_offset']}:linear=true:print_format=json"
    )
    if resample:
        af = loudnorm + ",aresample=44100:resampler=soxr:precision=28:dither_method=triangular"
    else:
        af = loudnorm
    proc = run_ffmpeg(
        ["-y", "-i", str(src), "-af", af, "-c:a", "pcm_s24le", "-ar", str(OUTPUT_SR), "-ac", "2", str(dst)],
        what=f"loudnorm apply to {src.name}",
    )
    mode = "unknown"
    matches = list(re.finditer(r"\{[^{}]*\}", proc.stderr or "", re.S))
    if matches:
        try:
            mode = str(json.loads(matches[-1].group(0)).get("normalization_type") or mode)
        except json.JSONDecodeError:
            mode = "unknown"
    return mode


def _corrective(src: Path, dst: Path, gain_db: float) -> None:
    limit = 10 ** (LIMITER_CEILING_DB / 20.0)
    af = (
        f"volume={gain_db:.3f}dB,"
        f"aresample={OUTPUT_SR * 4}:resampler=soxr,"
        f"alimiter=limit={limit:.6f}:level=false:attack=1:release=50:asc=1,"
        f"aresample={OUTPUT_SR}:resampler=soxr:precision=28:dither_method=triangular"
    )
    run_ffmpeg(
        ["-y", "-i", str(src), "-af", af, "-c:a", "pcm_s24le", "-ar", str(OUTPUT_SR), "-ac", "2", str(dst)],
        what="corrective limiter",
    )


def measure_master(path: Path) -> dict:
    """Independent measurement of a finished WAV. True peak is oversampled."""
    ebur_i, ebur_tp = measure_ebur128(path)
    pcm, sr = load_audio(path)
    oversampled = true_peak_dbtp(pcm, sr, oversample=4)
    true_peak = max(ebur_tp, oversampled)
    regions = silence_regions(pcm, sr)
    info = probe_format(path)
    return {
        "lufs_i": ebur_i,
        "ebur128_true_peak_dbtp": ebur_tp,
        "oversampled_true_peak_dbtp": oversampled,
        "true_peak_dbtp": true_peak,
        "leading_silence_s": regions["leading_s"],
        "trailing_silence_s": regions["trailing_s"],
        "sample_rate": info["sample_rate"],
        "channels": info["channels"],
        "subtype": info["subtype"],
        "duration_s": info["duration_s"],
        "frames": info["frames"],
    }


def master_file(src: Path, dst: Path, *, target_lufs: float = TARGET_LUFS) -> dict:
    """Master ``src`` to ``dst`` (24-bit 44.1 kHz WAV) and return the receipt.

    The destination is replaced only after the file is complete. The receipt
    records trim indexes into the decoded input so a later split can map
    cue-sheet positions onto this master.
    """
    src, dst = Path(src), Path(dst)
    pcm, sr = load_audio(src)
    stereo = to_stereo(pcm)
    trimmed, trim_start, trim_end = trim_edges(stereo, sr)
    dst.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="relay-master-") as tmp:
        tmpdir = Path(tmp)
        trimmed_path = tmpdir / "trimmed.wav"
        sf.write(str(trimmed_path), trimmed, sr, subtype="FLOAT")
        measured = _loudnorm_measure(trimmed_path, target_lufs)
        staged = tmpdir / "master.wav"
        mode = _apply_loudnorm(trimmed_path, staged, measured, target_lufs, resample=(sr != OUTPUT_SR))
        measured_out = measure_master(staged)
        ok, notes = judge(
            measured_out["lufs_i"], measured_out["true_peak_dbtp"],
            measured_out["leading_silence_s"], measured_out["trailing_silence_s"],
            measured_out["sample_rate"], measured_out["subtype"], measured_out["channels"],
            target_lufs=target_lufs,
        )
        corrective = False
        if not ok:
            gain = target_lufs - measured_out["lufs_i"]
            if not math.isfinite(gain):
                gain = 0.0
            corrected = tmpdir / "corrected.wav"
            _corrective(staged, corrected, gain)
            corrected.replace(staged)
            corrective = True
            mode = mode if mode != "unknown" else "corrective"
            measured_out = measure_master(staged)
            ok, notes = judge(
                measured_out["lufs_i"], measured_out["true_peak_dbtp"],
                measured_out["leading_silence_s"], measured_out["trailing_silence_s"],
                measured_out["sample_rate"], measured_out["subtype"], measured_out["channels"],
                target_lufs=target_lufs,
            )
        # Publish the WAV before the receipt so a reader never sees a receipt
        # for a missing file. Verification still happens on the published bytes.
        published = dst
        if published.exists():
            published.unlink()
        shutil.move(str(staged), str(published))
    digest = file_md5(published)
    report = {
        "file": published.name,
        "duration_s": round(measured_out["duration_s"], 4),
        "lufs_i": round(measured_out["lufs_i"], 2),
        "true_peak_dbtp": round(measured_out["true_peak_dbtp"], 2),
        "ebur128_true_peak_dbtp": round(measured_out["ebur128_true_peak_dbtp"], 2),
        "oversampled_true_peak_dbtp": round(measured_out["oversampled_true_peak_dbtp"], 2),
        "lead_silence_s": measured_out["leading_silence_s"],
        "tail_silence_s": measured_out["trailing_silence_s"],
        "md5": digest,
        "loudnorm_mode": mode,
        "corrective_pass": corrective,
        "sample_rate": measured_out["sample_rate"],
        "bit_depth": 24,
        "channels": measured_out["channels"],
        "input_frames": int(stereo.shape[0]),
        "input_sample_rate": int(sr),
        "trim_start_sample": int(trim_start),
        "trim_end_sample": int(trim_end),
        "output_frames": int(measured_out["frames"]),
        "target_lufs": target_lufs,
        "verdict": "PASS" if ok else "FAIL",
        "notes": notes,
        "ffmpeg": ffmpeg_bin(),
    }
    _write_receipt(published, report)
    return report


def _write_receipt(wav: Path, report: dict) -> None:
    from mixgen.util import atomic_write_json

    atomic_write_json(Path(str(wav) + ".json"), report)


def master_failure_message(report: dict) -> str:
    window_lo, window_hi = lufs_window()
    detail = "; ".join(report.get("notes") or []) or "out of tolerance"
    return (
        f"master FAIL {report.get('file')}: {detail} "
        f"(want {window_lo:.0f}..{window_hi:.0f} LUFS, true peak <= {TRUE_PEAK_MAX:.1f} dBTP, "
        f"{OUTPUT_SR} Hz 24-bit stereo)"
    )


def resolve_take(renders: Path, album: str | None, stem: str, take: int) -> Path:
    root = Path(renders)
    folder = root / album / stem if album else root / stem
    if not folder.is_dir() and (root / stem).is_dir():
        folder = root / stem
    for ext in ("flac", "wav", "aiff", "aif"):
        path = folder / f"take-{take}.{ext}"
        if path.is_file():
            return path
    raise RelayError(
        f"no take-{take} for {stem} under {folder} (expected take-{take}.flac)",
        exit_code=2,
    )


def iter_side_a(designs: Path) -> list[tuple[str, dict]]:
    """``(stem, design)`` for every ``NN-slug.json``, in track order."""
    rows = []
    for path in sorted(designs.glob("[0-9][0-9]-*.json")):
        data = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(data, dict):
            raise RelayError(f"{path.name}: expected an object")
        side = str(data.get("side") or "A").upper()
        if side not in ("A", ""):
            continue
        rows.append((path.stem, data))
    rows.sort(key=lambda item: int(item[1].get("track") or item[0][:2]))
    return rows
