"""Per-take QA: tempo, key, length, drums-only intro/outro, clipping and silence.

Each take gets sub-scores in 0..1, a weighted total, and the best take is picked
per track. Reports are written per track plus a mix-wide summary.
"""
from __future__ import annotations

import time
from pathlib import Path

import numpy as np

from mixgen.plan import Track, camelot_of
from mixgen.util import atomic_write_json, atomic_write_text

ANALYSIS_SR = 22050
DRUMS_ONLY_HARMONIC_RATIO = 0.35   # harmonic/percussive energy below this == drums-only bar
REQUIRED_DRUMS_BARS = 16

WEIGHTS = {"tempo": 0.30, "key": 0.25, "length": 0.15, "drums": 0.20, "tech": 0.10}


# ------------------------------------------------------------------ helpers
def norm_key(name: str) -> str:
    """``Am`` / ``A minor`` / ``Amin`` -> ``A``; ``F# major`` -> ``F#``."""
    s = str(name).strip().lower().replace("minor", "min").replace("major", "maj")
    if s.endswith("min"):
        s = s[:-3]
    elif s.endswith("maj"):
        s = s[:-3]
    elif s.endswith("m"):
        s = s[:-1]
    s = s.strip().capitalize()
    return s.replace("b", "#") if s.endswith("b") and len(s) == 2 else s


def key_to_camelot(name: str, mode: str | None = None) -> str:
    """Convert a key name (``Am``, ``A``, ``A minor``) to Camelot."""
    if mode is None:
        low = str(name).strip().lower()
        if "min" in low or (low.endswith("m") and not low.endswith("maj")):
            mode = "min"
        elif "maj" in low:
            mode = "maj"
        else:
            mode = "min"
    return camelot_of(norm_key(name), mode)


def fold_tempo(detected: float, target: float) -> tuple[float, float]:
    """Fold half/double-time detections onto the target; return (folded, abs err)."""
    best, err = float(detected), abs(float(detected) - target)
    for mult in (0.25, 0.5, 2.0, 4.0):
        cand = float(detected) * mult
        if abs(cand - target) < err:
            best, err = cand, abs(cand - target)
    return best, err


def detect_tempo(y: np.ndarray, sr: int, target: float) -> float:
    """Rough tempo estimate refined with djmix's onset comb fit (+-3 %)."""
    import librosa

    from djmix.analysis import _comb_fit

    est, _ = librosa.beat.beat_track(y=y, sr=sr, start_bpm=target, tightness=100)
    est = float(np.atleast_1d(est)[0])
    folded, _ = fold_tempo(est, target)
    bpm, _phase = _comb_fit(y, sr, folded)
    return float(bpm)


def _bar_energies(y: np.ndarray, sr: int, bpm: float, beats_per_bar: int = 4,
                  h_spec=None, p_spec=None):
    """Per-bar harmonic vs percussive RMS using HPSS (pass spectra to reuse them)."""
    import librosa

    bar_dur = beats_per_bar * 60.0 / bpm
    hop = 512
    if h_spec is None or p_spec is None:
        S = np.abs(librosa.stft(y, n_fft=1024, hop_length=hop))
        h_spec, p_spec = librosa.decompose.hpss(S, margin=2.0)
    h = librosa.feature.rms(S=h_spec, frame_length=1024)[0]
    p = librosa.feature.rms(S=p_spec, frame_length=1024)[0]

    n = min(len(h), len(p))
    if n == 0:
        return np.zeros(0), np.zeros(0), bar_dur
    times = librosa.frames_to_time(np.arange(n), sr=sr, hop_length=hop)
    n_bars = int(times[-1] / bar_dur) + 1
    idx = np.minimum((times / bar_dur).astype(int), n_bars - 1)
    harm = np.zeros(n_bars)
    perc = np.zeros(n_bars)
    cnt = np.zeros(n_bars)
    np.add.at(harm, idx, h[:n])
    np.add.at(perc, idx, p[:n])
    np.add.at(cnt, idx, 1.0)
    cnt[cnt == 0] = 1.0
    return harm / cnt, perc / cnt, bar_dur


def drums_sections(y: np.ndarray, sr: int, bpm: float, beats_per_bar: int = 4,
                   h_spec=None, p_spec=None) -> tuple[int, int]:
    """Leading/trailing run of drums-only bars (harmonic energy low vs percussive)."""
    harm, perc, _ = _bar_energies(y, sr, bpm, beats_per_bar, h_spec, p_spec)
    if len(harm) == 0:
        return 0, 0
    drums_only = (perc > 0) & (harm < DRUMS_ONLY_HARMONIC_RATIO * perc)
    intro = 0
    for v in drums_only:
        if not v:
            break
        intro += 1
    outro = 0
    for v in drums_only[::-1]:
        if not v:
            break
        outro += 1
    return intro, outro


def clipping_stats(y: np.ndarray) -> dict:
    a = np.abs(y)
    peak = float(a.max()) if len(a) else 0.0
    near = a >= 0.997
    frac = float(near.mean()) if len(a) else 0.0
    edges = np.diff(near.astype(np.int8))
    runs = int(np.count_nonzero(edges == 1) + (1 if len(near) and near[0] else 0))
    clipped = bool(peak >= 0.99 and runs >= 3 and frac > 1e-5)
    return {"peak": peak, "peak_dbfs": float(20 * np.log10(peak)) if peak > 0 else -120.0,
            "clip_fraction": frac, "clip_runs": runs, "clipped": clipped}


def silence_stats(y: np.ndarray, sr: int) -> dict:
    hop = max(1, int(0.01 * sr))
    n = (len(y) // hop) * hop
    if n == 0:
        return {"longest_silence_s": 0.0, "silent": True}
    frames = y[:n].reshape(-1, hop)
    rms = np.sqrt(np.mean(frames.astype(np.float64) ** 2, axis=1))
    quiet = rms < 10 ** (-60 / 20)
    longest = 0
    run = 0
    for q in quiet:
        run = run + 1 if q else 0
        longest = max(longest, run)
    longest_s = longest * hop / sr
    overall = float(np.sqrt(np.mean(y.astype(np.float64) ** 2))) if len(y) else 0.0
    return {"longest_silence_s": longest_s, "silent": bool(longest_s > 2.0 or overall < 1e-5)}


# ------------------------------------------------------------------ scoring
def analyze_take(path: Path, track: Track, take_idx: int = 0,
                 analysis_sr: int = ANALYSIS_SR) -> dict:
    import librosa

    y, sr = librosa.load(str(path), sr=analysis_sr, mono=True)
    duration = len(y) / sr if sr else 0.0
    target_bpm = float(track.bpm)
    det = detect_tempo(y, sr, target_bpm)
    folded, bpm_err = fold_tempo(det, target_bpm)
    tol = max(1.0, 0.01 * target_bpm)
    bpm_score = 1.0 if bpm_err <= tol else max(0.0, 1.0 - (bpm_err - tol) / (2.0 * tol))

    # key from the harmonic component only (pad/harmony, not kick noise); reuse
    # the same HPSS spectra for the drums-only intro/outro check below.
    S = np.abs(librosa.stft(y, n_fft=1024, hop_length=512))
    H, P = librosa.decompose.hpss(S, margin=2.0)
    y_h = librosa.istft(H, hop_length=512, length=len(y))
    # drop the kick fundamental before chroma: a decaying low sine is "harmonic"
    # to HPSS but must not vote on the key.
    from scipy import signal as _signal

    sos = _signal.butter(4, 120.0, "highpass", fs=sr, output="sos")
    y_h = _signal.sosfiltfilt(sos, y_h).astype(np.float32)

    intro_bars, outro_bars = drums_sections(y, sr, target_bpm, h_spec=H, p_spec=P)

    # Key comes from the harmonic body: the drums-only intro/outro carry no key
    # information and would otherwise swamp chroma with percussion bleed.
    bar_n = int(round(4 * 60.0 / target_bpm * sr))
    body = y_h[intro_bars * bar_n: len(y_h) - outro_bars * bar_n]
    if len(body) < 4 * bar_n:
        body = y_h

    from djmix.analysis import detect_key

    key_name, camelot = detect_key(body, sr)

    key_score = 1.0 if camelot == track.key else 0.0
    if key_score == 0.0 and camelot:
        num_t, let_t = int(track.key[:-1]), track.key[-1]
        try:
            num_d, let_d = int(camelot[:-1]), camelot[-1]
        except ValueError:
            num_d, let_d = -99, "?"
        if (num_d == num_t) or (let_d == let_t and abs(num_d - num_t) == 1):
            key_score = 0.4

    length_err = abs(duration - float(track.duration))
    tol_len = max(2.0, 0.02 * track.duration)
    length_score = 1.0 if length_err <= tol_len else max(0.0, 1.0 - (length_err - tol_len) / (0.10 * track.duration))

    drums_score = 0.5 * min(1.0, intro_bars / REQUIRED_DRUMS_BARS) + \
        0.5 * min(1.0, outro_bars / REQUIRED_DRUMS_BARS)

    # clipping/silence must be measured at the native sample rate: a 22.05 kHz
    # resample softens flat-topped peaks below any sensible clip threshold.
    import soundfile as sf

    y_native, native_sr = sf.read(str(path), dtype="float32", always_2d=True)
    # left channel, not a channel average: averaging a clipped L with a slightly
    # quieter R hides flat-topped peaks.
    mono = y_native[:, 0] if y_native.ndim == 2 else y_native
    clip = clipping_stats(mono)
    sil = silence_stats(mono, native_sr or sr)
    tech_score = 1.0 - (0.5 if clip["clipped"] else 0.0) - (0.5 if sil["silent"] else 0.0)
    tech_score = max(0.0, tech_score)

    total = (WEIGHTS["tempo"] * bpm_score + WEIGHTS["key"] * key_score
             + WEIGHTS["length"] * length_score + WEIGHTS["drums"] * drums_score
             + WEIGHTS["tech"] * tech_score)

    notes = []
    if bpm_score < 1.0:
        notes.append(f"tempo {det:.2f} BPM (folded {folded:.2f}) vs {target_bpm:g}")
    if key_score < 1.0:
        notes.append(f"key {key_name} ({camelot or '?'}) vs {track.key}")
    if length_score < 1.0:
        notes.append(f"length {duration:.1f}s vs {track.duration:.1f}s")
    if intro_bars < REQUIRED_DRUMS_BARS:
        notes.append(f"drums-only intro {intro_bars} bars < {REQUIRED_DRUMS_BARS}")
    if outro_bars < REQUIRED_DRUMS_BARS:
        notes.append(f"drums-only outro {outro_bars} bars < {REQUIRED_DRUMS_BARS}")
    if clip["clipped"]:
        notes.append(f"clipping: peak {clip['peak_dbfs']:.2f} dBFS, {clip['clip_runs']} runs")
    if sil["silent"]:
        notes.append(f"silence: {sil['longest_silence_s']:.1f}s")

    return {
        "take": take_idx,
        "file": str(path),
        "duration_s": round(duration, 3),
        "bpm_detected": round(det, 2),
        "bpm_folded": round(folded, 2),
        "bpm_target": target_bpm,
        "bpm_err": round(bpm_err, 2),
        "key_detected": key_name,
        "camelot_detected": camelot,
        "camelot_target": track.key,
        "drums_intro_bars": intro_bars,
        "drums_outro_bars": outro_bars,
        "peak_dbfs": round(clip["peak_dbfs"], 2),
        "clip_fraction": round(clip["clip_fraction"], 6),
        "clip_runs": clip["clip_runs"],
        "clipped": clip["clipped"],
        "longest_silence_s": round(sil["longest_silence_s"], 2),
        "silent": sil["silent"],
        "scores": {
            "tempo": round(bpm_score, 4), "key": round(key_score, 4),
            "length": round(length_score, 4), "drums": round(drums_score, 4),
            "tech": round(tech_score, 4),
        },
        "score": round(float(total), 4),
        "notes": notes,
    }


def qa_track(track: Track, take_paths: list[Path], out_dir: Path) -> dict:
    """Analyze all takes, pick the best, write report.json + report.md."""
    takes = [analyze_take(p, track, i + 1) for i, p in enumerate(take_paths)]
    best = max(takes, key=lambda t: (t["score"], -t["take"]))
    report = {
        "track": {"index": track.index, "title": track.title, "slug": track.slug,
                  "bpm": track.bpm, "key": track.key, "duration_s": track.duration,
                  "takes": track.takes},
        "takes": takes,
        "pick": f"take-{best['take']}",
        "pick_file": best["file"],
        "pick_score": best["score"],
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }
    out_dir.mkdir(parents=True, exist_ok=True)
    atomic_write_json(out_dir / "report.json", report)
    atomic_write_text(out_dir / "report.md", render_md(report))
    return report


def render_md(report: dict) -> str:
    t = report["track"]
    lines = [f"# QA — {t['title']}", "",
             f"- target: {t['bpm']:g} BPM, {t['key']}, {t['duration_s']:.0f}s, {t['takes']} takes",
             f"- pick: **{report['pick']}** (score {report['pick_score']:.3f})", "",
             "| take | bpm | key | dur (s) | drums in/out | peak dBFS | clipped | score | notes |",
             "|---|---|---|---|---|---|---|---|---|"]
    for k in report["takes"]:
        notes = "; ".join(k["notes"]) or "ok"
        lines.append(
            f"| {k['take']} | {k['bpm_detected']:.2f} | {k['camelot_detected'] or '?'} "
            f"| {k['duration_s']:.1f} | {k['drums_intro_bars']}/{k['drums_outro_bars']} "
            f"| {k['peak_dbfs']:.2f} | {'yes' if k['clipped'] else 'no'} "
            f"| {k['score']:.3f} | {notes} |")
    return "\n".join(lines) + "\n"
