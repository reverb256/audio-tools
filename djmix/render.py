"""Stretch, align, automate and sum the tracks; write 48k/24-bit; loudnorm."""
from __future__ import annotations

import json
import os
import re
import shutil
import subprocess
import tempfile
from pathlib import Path

import numpy as np
import soundfile as sf
from scipy import signal

from djmix.plan import Placement, Plan

SPLIT_HZ = 150.0  # bass-swap crossover


# ---------------------------------------------------------------- time stretch
def rubberband_bin() -> str | None:
    return os.environ.get("DJMIX_RUBBERBAND") or shutil.which("rubberband")


_RB_FLAGS = ["-q", "-3", "--pitch-hq"]   # R3 ("finer") engine: best quality for full mixes
_LAT_CACHE: dict = {}


def _rb_run(exe: str, flags: list[str], seg, sr, src_pts, dst_pts) -> np.ndarray:
    with tempfile.TemporaryDirectory(prefix="djmix-") as td:
        fin, fout, fmap = (Path(td) / n for n in ("in.wav", "out.wav", "map.txt"))
        sf.write(fin, seg, sr, subtype="FLOAT")
        fmap.write_text("".join(f"{int(s)} {int(d)}\n" for s, d in zip(src_pts, dst_pts)))
        n_out = int(dst_pts[-1])
        subprocess.run([exe, *flags, "-D", f"{n_out / sr:.6f}", "-M", str(fmap), str(fin), str(fout)],
                       check=True, capture_output=True, text=True)
        out, _ = sf.read(fout, dtype="float32", always_2d=True)
    return _fix_len(out, n_out)


def _click_times(y: np.ndarray, sr: int) -> np.ndarray:
    env = np.abs(y[:, 0])
    k = max(1, int(0.0005 * sr))
    env = np.convolve(env, np.ones(k) / k, mode="same")
    peaks, _ = signal.find_peaks(env, height=0.3 * env.max(), distance=int(0.2 * sr))
    return peaks / sr


def _rb_latency(exe: str, flags: list[str], sr: int, ratio: float) -> int:
    """rubberband's R3 engine leaves a small constant delay (~8-9 ms at 48 kHz) even with
    a time map. Measure it once per (flags, ratio) on a click probe and compensate."""
    key = (exe, tuple(flags), sr, round(ratio, 3))
    if key in _LAT_CACHE:
        return _LAT_CACHE[key]
    per = 0.5
    n_clicks = 12
    y = np.zeros((int((n_clicks + 1) * per * sr), 2), dtype=np.float32)
    t = np.arange(int(0.004 * sr)) / sr
    burst = (np.sin(2 * np.pi * 2000 * t) * np.exp(-t / 0.001)).astype(np.float32)
    for k in range(1, n_clicks + 1):
        s = int(k * per * sr)
        y[s:s + len(burst)] += burst[:, None]
    src = np.round(np.arange(n_clicks + 2) * per * sr).astype(np.int64)
    src[-1] = len(y)
    dst = np.round(src * ratio).astype(np.int64)
    ref = _click_times(y, sr)
    try:
        out = _rb_run(exe, flags, y, sr, src, dst)
        got = _click_times(out, sr)
        exp = ref * ratio
        d = [g - exp[np.argmin(np.abs(exp - g))] for g in got]
        lat = int(round(float(np.median(d)) * sr)) if len(d) >= n_clicks // 2 else 0
    except Exception:
        lat = 0
    _LAT_CACHE[key] = lat
    return lat


def _stretch_rubberband(seg: np.ndarray, sr: int, src_pts: np.ndarray, dst_pts: np.ndarray, exe: str) -> np.ndarray:
    """One rubberband call with a key-frame time map (handles constant and ramped tempo)."""
    flags = list(_RB_FLAGS)
    try:
        out = _rb_run(exe, flags, seg, sr, src_pts, dst_pts)
    except subprocess.CalledProcessError:  # rubberband < 3 has no R3 engine
        flags = ["-q"]
        out = _rb_run(exe, flags, seg, sr, src_pts, dst_pts)
    ratio = float(dst_pts[-1] - dst_pts[0]) / float(max(1, src_pts[-1] - src_pts[0]))
    lat = _rb_latency(exe, flags, sr, ratio)
    if lat > 0:
        out = np.concatenate([out[lat:], np.zeros((lat, out.shape[1]), dtype=out.dtype)])
    elif lat < 0:
        out = np.concatenate([np.zeros((-lat, out.shape[1]), dtype=out.dtype), out[:lat]])
    return out


def _stretch_librosa(seg: np.ndarray, sr: int, src_pts: np.ndarray, dst_pts: np.ndarray) -> np.ndarray:
    """Fallback phase vocoder, piecewise between key frames with equal ratios merged."""
    import librosa
    pieces = []
    ratios = np.diff(dst_pts) / np.diff(src_pts)
    i = 0
    while i < len(ratios):
        j = i
        while j + 1 < len(ratios) and abs(ratios[j + 1] - ratios[i]) < 1e-6:
            j += 1
        s0, s1 = int(src_pts[i]), int(src_pts[j + 1])
        n_out = int(dst_pts[j + 1]) - int(dst_pts[i])
        chunk = seg[s0:s1]
        if abs(ratios[i] - 1.0) < 1e-6:
            y = chunk
        else:
            y = np.stack([librosa.effects.time_stretch(chunk[:, c], rate=1.0 / ratios[i], n_fft=1024)
                          for c in range(chunk.shape[1])], axis=1)
        pieces.append(_fix_len(y.astype(np.float32), n_out))
        i = j + 1
    return np.concatenate(pieces, axis=0)


def _fix_len(y: np.ndarray, n: int) -> np.ndarray:
    if len(y) >= n:
        return y[:n]
    return np.pad(y, ((0, n - len(y)), (0, 0)))


def stretch(seg, sr, src_pts, dst_pts, engine: str = "auto") -> np.ndarray:
    exe = rubberband_bin()
    if engine == "rubberband" or (engine == "auto" and exe):
        if not exe:
            raise RuntimeError("rubberband CLI not found (install rubberband-cli)")
        return _stretch_rubberband(seg, sr, src_pts, dst_pts, exe)
    return _stretch_librosa(seg, sr, src_pts, dst_pts)


# ---------------------------------------------------------------- automation
def _ramp(n: int, a: int, b: int, shape: str = "lin") -> np.ndarray:
    """0 before a, 1 after b, ramp between (sample indices)."""
    x = np.clip((np.arange(n) - a) / max(1, b - a), 0.0, 1.0)
    return np.sin(x * np.pi / 2) if shape == "pow" else x


def _split(y: np.ndarray, sr: int):
    """Zero-phase LR-style split so the band sum stays time-aligned (no beat drift)."""
    sos = signal.butter(4, SPLIT_HZ, "lowpass", fs=sr, output="sos")
    low = signal.sosfiltfilt(sos, y, axis=0).astype(np.float32)
    return low, y - low


def _sweep(y: np.ndarray, sr: int, f0: float, f1: float, kind: str, block: int = 1024) -> np.ndarray:
    """Time-varying 2nd-order filter, log-swept f0->f1 across y (block-wise, state carried)."""
    out = np.empty_like(y)
    n = len(y)
    zi = np.zeros((1, 2, y.shape[1]))
    for s in range(0, n, block):
        e = min(n, s + block)
        f = f0 * (f1 / f0) ** (((s + e) / 2) / n)
        f = min(f, sr * 0.45)
        sos = signal.butter(2, f, kind, fs=sr, output="sos")
        out[s:e], zi = signal.sosfilt(sos, y[s:e], axis=0, zi=zi)
    return out


def apply_transition(out_seg: np.ndarray, in_seg: np.ndarray, sr: int, kind: str, beat_len: int):
    """Shape the overlapping parts of the outgoing and incoming tracks (same length)."""
    n = len(out_seg)
    half = n // 2
    if kind == "long_blend":
        g_in = _ramp(n, 0, n, "pow")
        g_out = _ramp(n, 0, n, "pow")[::-1]
        return out_seg * g_out[:, None], in_seg * g_in[:, None]
    if kind == "eq_bass_swap":
        o_lo, o_hi = _split(out_seg, sr)
        i_lo, i_hi = _split(in_seg, sr)
        swap = _ramp(n, half - beat_len // 2, half + beat_len // 2)  # one-beat bass swap on the phrase mid-point
        in_hi_g = _ramp(n, 0, half, "pow")
        out_hi_g = _ramp(n, half, n, "pow")[::-1]
        o = o_lo * (1 - swap)[:, None] + o_hi * out_hi_g[:, None]
        i = i_lo * swap[:, None] + i_hi * in_hi_g[:, None]
        return o, i
    if kind == "filter_sweep":
        o = _sweep(out_seg, sr, 30.0, 3000.0, "highpass")
        i = _sweep(in_seg, sr, 250.0, 20000.0, "lowpass")
        g_in = _ramp(n, 0, half, "pow")
        g_out = _ramp(n, half, n, "pow")[::-1]
        return o * g_out[:, None], i * g_in[:, None]
    raise ValueError(kind)


# ---------------------------------------------------------------- render
def _load(path: Path, sr: int) -> np.ndarray:
    y, file_sr = sf.read(str(path), dtype="float32", always_2d=True)
    if y.shape[1] == 1:
        y = np.repeat(y, 2, axis=1)
    y = y[:, :2]
    if file_sr != sr:
        import librosa
        y = librosa.resample(y.T, orig_sr=file_sr, target_sr=sr, res_type="soxr_hq").T.astype(np.float32)
    return y


def render_layers(plan: Plan, engine: str = "auto", log=print, keep_layers: bool = False) -> tuple[np.ndarray, list[np.ndarray]]:
    """Return (mix, per-track layers placed on the output timeline). Layers are for tests/stems."""
    sr = plan.cue.sample_rate
    bpb = plan.cue.beats_per_bar
    total = int(round(plan.duration * sr))
    mix = np.zeros((total, 2), dtype=np.float32)
    layers: list[np.ndarray] = []
    shaped: dict[int, tuple[int, np.ndarray]] = {}
    for p in plan.placements:
        log(f"  stretching [{p.index}] {p.track.title} -> {p.track.target_bpm:g} BPM")
        y = _load(p.track.file, sr) * np.float32(10 ** (p.track.gain_db / 20))
        src_beats = p.src_bar0 + np.arange(p.bars * bpb + 1) * p.src_period
        src_pts = np.round(src_beats * sr).astype(np.int64)
        out_t = plan.out_beat_times(p)
        o0 = int(round(out_t[0] * sr))
        dst_pts = np.round(out_t * sr).astype(np.int64) - o0
        s0 = src_pts[0]
        seg = y[s0:src_pts[-1]]
        seg = _fix_len(seg, int(src_pts[-1] - s0))
        st = stretch(seg, sr, src_pts - s0, dst_pts, engine)
        # 2 ms de-click at the very edges
        e = int(0.002 * sr)
        st[:e] *= np.linspace(0, 1, e, dtype=np.float32)[:, None]
        st[-e:] *= np.linspace(1, 0, e, dtype=np.float32)[:, None]
        shaped[p.index] = (o0, st)
    # transitions
    for k in range(1, len(plan.placements)):
        cur = plan.placements[k]
        if cur.overlap_in == 0:
            continue
        prev = plan.placements[k - 1]
        po0, pst = shaped[prev.index]
        co0, cst = shaped[cur.index]
        n = len(pst) - (co0 - po0)
        n = min(n, len(cst))
        beat_len = int(round(60.0 / cur.track.target_bpm * sr))
        o, i = apply_transition(pst[-n:].copy(), cst[:n].copy(), sr, cur.track.transition, beat_len)
        pst[-n:] = o
        cst[:n] = i
    for p in plan.placements:
        o0, st = shaped[p.index]
        end = min(total, o0 + len(st))
        mix[o0:end] += st[:end - o0]
        layer = np.zeros((total, 2), dtype=np.float32) if keep_layers else None
        if layer is not None:
            layer[o0:end] = st[:end - o0]
            layers.append(layer)
    return mix, layers


# ---------------------------------------------------------------- loudness
def _ffmpeg() -> str:
    return os.environ.get("DJMIX_FFMPEG") or ("/usr/bin/ffmpeg" if Path("/usr/bin/ffmpeg").exists() else "ffmpeg")


def _measure(ff: str, path: Path, I: float, TP: float, LRA: float = 11.0) -> dict:
    r = subprocess.run([ff, "-hide_banner", "-nostats", "-i", str(path), "-af",
                        f"loudnorm=I={I}:TP={TP}:LRA={LRA}:print_format=json", "-f", "null", "-"],
                       capture_output=True, text=True, check=True)
    return json.loads(re.findall(r"\{[^{}]*\}", r.stderr, re.S)[-1])


def _apply(ff: str, src: Path, dst: Path, sr: int, gain_db: float, ceiling_db: float):
    lim = 10 ** (ceiling_db / 20)
    # limiter runs 4x oversampled so it catches inter-sample (true) peaks
    af = (f"volume={gain_db:.3f}dB,aresample={sr * 4}:resampler=soxr,"
          f"alimiter=limit={lim:.6f}:level=false:attack=0.1:release=60:asc=1,aresample={sr}:resampler=soxr")
    subprocess.run([ff, "-hide_banner", "-nostats", "-y", "-i", str(src), "-af", af,
                    "-ar", str(sr), "-c:a", "pcm_s24le", str(dst)], capture_output=True, text=True, check=True)


def loudnorm_2pass(src: Path, dst: Path, sr: int, I: float, TP: float, LRA: float = 11.0) -> dict:
    """Two-pass EBU R128 to a loud club target.

    Pass 1 measures with ffmpeg loudnorm (EBU R128 / BS.1770). Pass 2 applies the
    exact gain plus a 4x-oversampled true-peak limiter at TP - 0.3 dB, re-measures,
    and trims the gain (max 3 rounds) until integrated loudness is within 0.3 LU.
    ffmpeg's own loudnorm cannot reach -9 LUFS linearly on dynamic material and its
    dynamic fallback pumps, so it is used to measure, not to apply.
    """
    ff = _ffmpeg()
    m = _measure(ff, src, I, TP, LRA)
    gain = I - float(m["input_i"])
    after = m
    ceiling = TP - 0.3
    for _ in range(4):
        _apply(ff, src, dst, sr, gain, ceiling)
        after = _measure(ff, dst, I, TP, LRA)
        err = I - float(after["input_i"])
        over = float(after["input_tp"]) - TP
        if abs(err) <= 0.3 and over <= 0:
            break
        if over > 0:              # limiter missed inter-sample overs: pull the ceiling down
            ceiling -= over + 0.1
        gain += err
    over = float(after["input_tp"]) - TP
    if over > 0:  # last resort: keep the true-peak promise, give up a little loudness
        tmp = dst.with_suffix(".tp.wav")
        subprocess.run([ff, "-hide_banner", "-nostats", "-y", "-i", str(dst), "-af", f"volume={-(over + 0.05):.3f}dB",
                        "-c:a", "pcm_s24le", str(tmp)], capture_output=True, text=True, check=True)
        os.replace(tmp, dst)
        after = _measure(ff, dst, I, TP, LRA)
    return {"pass1": m, "mode": "gain+truepeak-limiter", "gain_db": round(gain, 2), "ceiling_db": round(ceiling, 2),
            "integrated_lufs": float(after["input_i"]), "true_peak_dbtp": float(after["input_tp"]),
            "lra": float(after["input_lra"])}


def render(plan: Plan, engine: str = "auto", log=print) -> dict:
    cue = plan.cue
    cue.output.parent.mkdir(parents=True, exist_ok=True)
    mix, _ = render_layers(plan, engine, log)
    peak = float(np.max(np.abs(mix))) if len(mix) else 0.0
    if peak > 0.99:  # headroom for the float->file step; loudnorm sets final level
        mix *= np.float32(0.99 / peak)
    report = {"output": str(cue.output), "duration_s": round(plan.duration, 3), "sample_rate": cue.sample_rate}
    if cue.loudnorm:
        with tempfile.TemporaryDirectory(prefix="djmix-ln-", dir=str(cue.output.parent)) as td:
            pre = Path(td) / "premaster.wav"
            sf.write(pre, mix, cue.sample_rate, subtype="FLOAT")
            del mix
            log(f"  loudnorm 2-pass -> {cue.target_lufs} LUFS / {cue.target_tp} dBTP")
            report["loudness"] = loudnorm_2pass(pre, cue.output, cue.sample_rate, cue.target_lufs, cue.target_tp)
    else:
        sf.write(cue.output, mix, cue.sample_rate, subtype="PCM_24")
    from djmix.plan import tracklist
    cue.tracklist.write_text(tracklist(plan))
    report["tracklist"] = str(cue.tracklist)
    return report
