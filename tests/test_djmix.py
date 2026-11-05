"""Two synthetic click tracks (122 / 126 BPM) blended at 124 BPM: every click of both
tracks must land within 10 ms of the shared output beat grid through the blend."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest
import soundfile as sf
from scipy import signal

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from djmix.analysis import analyze  # noqa: E402
from djmix.cuesheet import load  # noqa: E402
from djmix.plan import build_plan, describe  # noqa: E402
from djmix.render import render, render_layers  # noqa: E402

SR = 48000
TOL = 0.010


def click_track(path: Path, bpm: float, bars: int, freq: float, lead_in: float):
    period = 60.0 / bpm
    n = int((lead_in + bars * 4 * period + 1.0) * SR)
    y = np.zeros(n, dtype=np.float32)
    t = np.arange(int(0.012 * SR)) / SR
    for k in range(bars * 4):
        s = int(round((lead_in + k * period) * SR))
        amp = 1.0 if k % 4 == 0 else 0.6       # accented downbeats
        y[s:s + len(t)] += (amp * np.sin(2 * np.pi * freq * t) * np.exp(-t / 0.003)).astype(np.float32)
    sf.write(path, np.stack([y, y], 1) * 0.8, SR, subtype="PCM_24")


def onsets(y: np.ndarray, freq: float) -> np.ndarray:
    """Click times (s) in one band: envelope peaks of a zero-phase band-pass."""
    sos = signal.butter(4, [freq / 1.3, freq * 1.3], "bandpass", fs=SR, output="sos")
    env = np.abs(signal.hilbert(signal.sosfiltfilt(sos, y)))
    thr = 0.2 * env.max()
    peaks, _ = signal.find_peaks(env, height=thr, distance=int(0.2 * SR))
    return peaks / SR


@pytest.fixture()
def mix_case(tmp_path):
    click_track(tmp_path / "a.wav", 122, 64, 1000.0, 0.25)
    click_track(tmp_path / "b.wav", 126, 64, 4000.0, 0.40)
    (tmp_path / "cue.yaml").write_text(
        "phrase_bars: 16\noutput: out/mix.wav\nloudnorm: false\n"
        "tracks:\n"
        "  - {file: a.wav, title: A, target_bpm: 124, play_bars: 48}\n"
        "  - {file: b.wav, title: B, target_bpm: 124, play_bars: 48, mix_in_bars: 16, transition: long_blend}\n")
    return tmp_path


def test_analysis_detects_tempo_and_downbeat(mix_case):
    a = analyze(mix_case / "a.wav")
    b = analyze(mix_case / "b.wav")
    assert abs(a.bpm - 122) < 0.2 and abs(b.bpm - 126) < 0.2
    assert abs(a.first_downbeat - 0.25) < TOL and abs(b.first_downbeat - 0.40) < TOL


@pytest.mark.parametrize("engine", ["librosa", "rubberband"])
def test_blend_beats_within_10ms(mix_case, engine):
    from djmix.render import rubberband_bin
    if engine == "rubberband" and not rubberband_bin():
        pytest.skip("rubberband CLI not installed")
    cue = load(mix_case / "cue.yaml", root=mix_case)
    plan = build_plan(cue, [analyze(t.file) for t in cue.tracks])
    mix, layers = render_layers(plan, engine, log=lambda m: None, keep_layers=True)
    grid = plan.beat_times
    blend0 = grid[plan.placements[1].out_bar0 * 4]
    blend1 = grid[(plan.placements[1].out_bar0 + 16) * 4]
    assert abs((blend1 - blend0) - 16 * 4 * 60 / 124) < 1e-6

    def check(sig, freq, lo, hi):
        hits = onsets(sig, freq)
        hits = hits[(hits >= lo - 0.02) & (hits < hi - 0.05)]
        expected = grid[(grid >= lo - 1e-6) & (grid < hi - 0.05)]
        assert len(hits) >= len(expected) - 1, (freq, len(hits), len(expected))
        err = np.array([np.min(np.abs(expected - h)) for h in hits])
        assert err.max() < TOL, f"{engine} {freq} Hz: worst beat error {err.max() * 1000:.1f} ms"
        return err

    # through the blend, in the actual mix (both click bands present at once)
    m = mix[:, 0]
    ea = check(m, 1000.0, blend0, blend1 - 0.6 * (blend1 - blend0))  # outgoing still audible
    eb = check(m, 4000.0, blend0 + 0.4 * (blend1 - blend0), blend1)  # incoming audible
    # and per layer across the whole overlap (gain-independent)
    L = blend1 - blend0
    check(layers[0][:, 0], 1000.0, blend0, blend0 + 0.75 * L)
    check(layers[1][:, 0], 4000.0, blend0 + 0.25 * L, blend1)
    assert ea.max() < TOL and eb.max() < TOL


def test_ramp_and_bass_swap_plan(mix_case):
    (mix_case / "ramp.yaml").write_text(
        "phrase_bars: 16\noutput: out/ramp.wav\nloudnorm: false\n"
        "tracks:\n"
        "  - {file: a.wav, target_bpm: 122, play_bars: 48}\n"
        "  - {file: b.wav, target_bpm: 126, play_bars: 48, mix_in_bars: 16, transition: eq_bass_swap, ramp: true}\n")
    cue = load(mix_case / "ramp.yaml", root=mix_case)
    plan = build_plan(cue, [analyze(t.file) for t in cue.tracks])
    ov = plan.beat_bpm[32 * 4:48 * 4]
    assert ov[0] < 122.1 and ov[-1] > 125.9 and np.all(np.diff(ov) > 0)
    assert "eq_bass_swap" in describe(plan)
    mix, layers = render_layers(plan, "librosa", log=lambda m: None, keep_layers=True)
    g = plan.beat_times
    lo, hi = g[32 * 4], g[48 * 4]
    hits = onsets(layers[1][:, 0], 4000.0)
    hits = hits[(hits >= lo - 0.02) & (hits < hi - 0.05)]
    err = np.array([np.min(np.abs(g - h)) for h in hits])
    assert len(hits) > 50 and err.max() < TOL


def test_full_render_writes_48k_24bit_and_tracklist(mix_case):
    (mix_case / "full.yaml").write_text(
        "phrase_bars: 16\noutput: out/full.wav\ntarget_lufs: -9\ntarget_tp: -1\n"
        "tracks:\n"
        "  - {file: a.wav, title: A, target_bpm: 124, play_bars: 32}\n"
        "  - {file: b.wav, title: B, target_bpm: 124, play_bars: 32, mix_in_bars: 16, transition: filter_sweep}\n")
    cue = load(mix_case / "full.yaml", root=mix_case)
    plan = build_plan(cue, [analyze(t.file) for t in cue.tracks])
    rep = render(plan, "librosa", log=lambda m: None)
    info = sf.info(rep["output"])
    assert info.samplerate == 48000 and info.subtype == "PCM_24" and info.channels == 2
    ld = rep["loudness"]
    assert ld["true_peak_dbtp"] <= -0.95          # -1 dBTP promise holds even on pathological clicks
    assert ld["integrated_lufs"] <= -8.5
    tl = Path(rep["tracklist"]).read_text().splitlines()
    assert tl[0].startswith("00:00  A") and tl[1].startswith("00:31  B")
