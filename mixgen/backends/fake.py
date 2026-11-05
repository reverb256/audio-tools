"""Synthetic WAV backend: drums + a tonal pad in the target key.

Used by tests and ``--dry-run``-adjacent workflows where no GPU/ACE-Step is
available. The layout mirrors the prompts the real backend receives: a
drums-only intro and outro of 16 bars, pad + drums in the body.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np

from mixgen.backends.base import GenBackend
from mixgen.plan import Track, camelot_to_tonic

_PITCH = {"C": 0, "C#": 1, "D": 2, "D#": 3, "E": 4, "F": 5, "F#": 6,
          "G": 7, "G#": 8, "A": 9, "A#": 10, "B": 11}


def _freq(pc: int, octave: int = 3) -> float:
    midi = 12 * (octave + 1) + pc
    return 440.0 * 2 ** ((midi - 69) / 12)


def synthesize(path: Path, *, bpm: float, camelot: str, duration: float,
               sample_rate: int = 48000, seed: int = 0, intro_bars: int = 16,
               outro_bars: int = 16, pad: bool = True, clipped: bool = False,
               drums: bool = True) -> Path:
    """Write one synthetic take. All knobs are explicit so tests can build bad takes."""
    rng = np.random.default_rng(seed)
    n = int(round(duration * sample_rate))
    y = np.zeros(n, dtype=np.float32)
    beat = 60.0 / bpm
    bar = beat * 4
    n_bars = int(duration / bar)
    intro_bars = max(0, min(intro_bars, n_bars // 2))
    outro_bars = max(0, min(outro_bars, n_bars // 2))

    t = np.arange(n) / sample_rate

    # --- drums (kick on every beat, accented downbeats)
    if drums:
        kick_len = int(0.18 * sample_rate)
        kt = np.arange(kick_len) / sample_rate
        kick = (np.sin(2 * np.pi * 58.0 * kt) * np.exp(-kt / 0.045)).astype(np.float32)
        k = 0
        while k * beat < duration:
            start = int(round(k * beat * sample_rate))
            amp = 0.62 if k % 4 == 0 else 0.46
            amp *= 1.0 + 0.03 * rng.standard_normal()
            end = min(n, start + kick_len)
            y[start:end] += amp * kick[: end - start]
            k += 1
        # offbeat tick: keeps the percussive band busy without drowning the pad
        hat_len = int(0.02 * sample_rate)
        ht = np.arange(hat_len) / sample_rate
        hat = (rng.standard_normal(hat_len) * np.exp(-ht / 0.004)).astype(np.float32)
        k = 0
        while (k + 0.5) * beat < duration:
            start = int(round((k + 0.5) * beat * sample_rate))
            end = min(n, start + hat_len)
            y[start:end] += 0.05 * hat[: end - start]
            k += 1

    # --- pad: tonic triad, only in the body (drums-only intro/outro stay tonal-free)
    if pad:
        tonic, mode = camelot_to_tonic(camelot)
        third = 3 if mode == "min" else 4
        freqs = [_freq(_PITCH[tonic]), _freq(_PITCH[tonic] + third),
                 _freq(_PITCH[tonic] + 7, octave=2)]
        body_from = int(round(intro_bars * bar * sample_rate))
        body_to = int(round((n_bars - outro_bars) * bar * sample_rate))
        body_to = min(body_to, n)
        if body_to > body_from:
            seg_n = body_to - body_from
            ts = np.arange(seg_n) / sample_rate
            env = 0.5 + 0.5 * np.sin(2 * np.pi * 0.07 * ts + rng.random() * 2 * np.pi)
            pad_sig = np.zeros(seg_n, dtype=np.float32)
            for amp, f in zip((0.08, 0.065, 0.05), freqs):
                pad_sig += (amp * (1.0 + 0.05 * rng.standard_normal())) * np.sin(
                    2 * np.pi * f * ts).astype(np.float32)
            pad_sig *= env.astype(np.float32)
            y[body_from:body_to] += pad_sig

    peak = float(np.max(np.abs(y))) or 1.0
    if clipped:
        y = np.clip(y * (2.5 / peak), -1.0, 1.0)   # flat tops at +/-1: loud, clipped take
    else:
        y *= 0.86 / peak

    path.parent.mkdir(parents=True, exist_ok=True)
    import soundfile as sf

    sf.write(path, np.stack([y, y * 0.97], axis=1).astype(np.float32), sample_rate,
             subtype="PCM_24")
    return path


class FakeBackend(GenBackend):
    name = "fake"

    def __init__(self, sample_rate: int = 48000):
        self.sample_rate = sample_rate

    def generate(self, track: Track, take_idx: int, out_path: Path) -> Path:
        seed = (track.seed or 0) + take_idx * 7919 + sum(ord(c) for c in track.slug)
        return synthesize(
            out_path, bpm=track.bpm, camelot=track.key, duration=track.duration,
            sample_rate=self.sample_rate, seed=seed,
            # take 1 gets a long outro, later takes vary slightly
            outro_bars=16 if take_idx == 1 else 20,
        )
