"""BPM, beat grid, downbeat and key detection (librosa; madmom/keyfinder if present)."""
from __future__ import annotations

import shutil
import subprocess
from dataclasses import dataclass

import numpy as np

KEYS = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]
# Krumhansl-Kessler key profiles
_MAJOR = np.array([6.35, 2.23, 3.48, 2.33, 4.38, 4.09, 2.52, 5.19, 2.39, 3.66, 2.29, 2.88])
_MINOR = np.array([6.33, 2.68, 3.52, 5.38, 2.60, 3.53, 2.54, 4.75, 3.98, 2.69, 3.34, 3.17])
# Camelot wheel, handy for DJ key compatibility
_CAMELOT_MAJ = {"B": "1B", "F#": "2B", "C#": "3B", "G#": "4B", "D#": "5B", "A#": "6B",
                "F": "7B", "C": "8B", "G": "9B", "D": "10B", "A": "11B", "E": "12B"}
_CAMELOT_MIN = {"G#": "1A", "D#": "2A", "A#": "3A", "F": "4A", "C": "5A", "G": "6A",
                "D": "7A", "A": "8A", "E": "9A", "B": "10A", "F#": "11A", "C#": "12A"}


@dataclass
class Analysis:
    bpm: float
    beats: np.ndarray        # seconds, a regular grid fitted to the detected beats
    first_downbeat: float    # seconds
    key: str
    camelot: str
    duration: float
    method: str


def _comb_fit(y, sr, est_bpm: float, fixed: bool = False, hop: int = 64):
    """Fine tempo + phase: the regular grid that collects the most onset energy.

    Searches +-3 % around the estimate in 0.01 BPM steps (exact for grid-rendered
    tracks); the phase is resolved to one onset hop (~3 ms at 22.05 kHz).
    """
    import librosa
    env = librosa.onset.onset_strength(y=y, sr=sr, hop_length=hop, aggregate=np.median)
    env = env - np.median(env)
    env[env < 0] = 0
    fps = sr / hop
    n = len(env)
    cands = np.array([est_bpm]) if fixed else np.round(np.arange(est_bpm * 0.97, est_bpm * 1.03, 0.01), 2)
    best = (-1.0, est_bpm, 0.0)
    for b in cands:
        P = 60.0 / b * fps                       # frames per beat (fractional)
        k = np.arange(int((n - 1) / P))
        if len(k) < 4:
            continue
        phases = np.arange(int(np.ceil(P)))
        idx = np.round(phases[:, None] + k[None, :] * P).astype(int)
        idx[idx >= n] = n - 1
        sc = env[idx].sum(axis=1)
        j = int(np.argmax(sc))
        if sc[j] > best[0]:
            best = (float(sc[j]), float(b), float(phases[j]))
    _, b, ph = best
    # onset_strength peaks sit ~1 hop after the true attack; librosa centres frames
    return b, max(0.0, ph * hop / sr)


def _refine_phase(y, sr, phase: float, period: float, win: float = 0.015) -> float:
    """Sample-accurate grid phase: median offset of the steepest energy rise near each beat."""
    from scipy import signal
    sos = signal.butter(2, 30.0, "highpass", fs=sr, output="sos")
    e = signal.sosfiltfilt(sos, y) ** 2
    k = max(1, int(0.001 * sr))
    e = np.convolve(e, np.ones(k) / k, mode="same")
    rise = np.diff(e, prepend=e[:1])
    w = int(win * sr)
    offs = []
    for t in np.arange(phase, len(y) / sr - win, period):
        c = int(round(t * sr))
        lo, hi = max(0, c - w), min(len(rise), c + w)
        seg = rise[lo:hi]
        if seg.size and seg.max() > 0:
            offs.append((lo + int(np.argmax(seg)) - c) / sr)
    if len(offs) < 4:
        return phase
    return max(0.0, phase + float(np.median(offs)))


def _downbeat_offset(y, sr, beats, beats_per_bar):
    """Pick the beat phase (0..bpb-1) with the strongest low-frequency onsets."""
    import librosa
    S = np.abs(librosa.stft(y, n_fft=2048, hop_length=512))
    freqs = librosa.fft_frequencies(sr=sr, n_fft=2048)
    low = S[freqs < 150].sum(axis=0)
    full = S.sum(axis=0)
    flux = np.maximum(0, np.diff(low, prepend=low[:1])) + 0.25 * np.maximum(0, np.diff(full, prepend=full[:1]))
    frames = librosa.time_to_frames(beats, sr=sr, hop_length=512)
    frames = frames[frames < len(flux)]
    strength = flux[frames]
    scores = [strength[k::beats_per_bar].mean() if len(strength[k::beats_per_bar]) else 0 for k in range(beats_per_bar)]
    return int(np.argmax(scores))


def detect_key(y, sr) -> tuple[str, str]:
    import librosa
    chroma = librosa.feature.chroma_cqt(y=y, sr=sr).mean(axis=1)
    best = (-2.0, "C", "maj")
    for i in range(12):
        for prof, mode in ((_MAJOR, "maj"), (_MINOR, "min")):
            r = np.corrcoef(np.roll(prof, i), chroma)[0, 1]
            if np.isfinite(r) and r > best[0]:
                best = (r, KEYS[i], mode)
    _, tonic, mode = best
    cam = (_CAMELOT_MAJ if mode == "maj" else _CAMELOT_MIN)[tonic]
    return f"{tonic}{'m' if mode == 'min' else ''}", cam


def _keyfinder(path) -> str | None:
    exe = shutil.which("keyfinder-cli")
    if not exe:
        return None
    try:
        out = subprocess.run([exe, str(path)], capture_output=True, text=True, timeout=300)
        return out.stdout.strip() or None
    except Exception:
        return None


def analyze(path, bpm: float | None = None, first_downbeat: float | None = None,
            beats_per_bar: int = 4, analysis_sr: int = 22050) -> Analysis:
    import librosa
    y, sr = librosa.load(str(path), sr=analysis_sr, mono=True)
    duration = len(y) / sr
    method = "librosa"
    beat_times = None
    try:  # madmom is optional and often fails to build on new Pythons
        from madmom.features.beats import DBNBeatTrackingProcessor, RNNBeatProcessor  # type: ignore
        act = RNNBeatProcessor()(str(path))
        beat_times = DBNBeatTrackingProcessor(fps=100)(act)
        method = "madmom"
    except Exception:
        pass
    if beat_times is None or len(beat_times) < 4:
        start_bpm = bpm or 125.0
        _, frames = librosa.beat.beat_track(y=y, sr=sr, start_bpm=start_bpm, tightness=400, units="frames")
        beat_times = librosa.frames_to_time(frames, sr=sr)
    if bpm is None:
        est = 60.0 / float(np.median(np.diff(beat_times))) if len(beat_times) > 3 else 125.0
        # house/trance live 100..160: fold half/double-time detections
        while est < 100:
            est *= 2
        while est > 160:
            est /= 2
    else:
        est = float(bpm)
    bpm_v, phase = _comb_fit(y, sr, est, fixed=bpm is not None)
    period = 60.0 / bpm_v
    phase = _refine_phase(y, sr, phase, period)
    beats = np.arange(phase, duration, period)
    if first_downbeat is None:
        k = _downbeat_offset(y, sr, beats, beats_per_bar)
        first_downbeat = float(beats[k]) if len(beats) > k else 0.0
    key = _keyfinder(path)
    if key:
        camelot = ""
        method += "+keyfinder"
    else:
        key, camelot = detect_key(y, sr)
    return Analysis(bpm=bpm_v, beats=beats, first_downbeat=float(first_downbeat),
                    key=key, camelot=camelot, duration=duration, method=method)
