"""Synthetic QA semantics. No catalogue audio and no release files.

These tests separate a near-full-scale sample-peak warning from a flat-top,
and an internal dropout from edge silence, zero crossings, and quiet passages.
Release loudness and true-peak limits are asserted unchanged and are not
applied by rewriting the source.
"""

from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import soundfile as sf

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

from relayrun.audio import (  # noqa: E402
    FLAT_TOP_FLOOR,
    FLAT_TOP_REF_HZ,
    PEAK_WARN_DBFS,
    QA_OVERSAMPLE,
    SILENCE_CONTEXT_S,
    SILENCE_DB,
    SILENCE_FRAME_S,
    SILENCE_LIMIT_S,
    clipping_report,
    flat_top_min_samples,
    silence_regions,
)
from relayrun.master import (  # noqa: E402
    LUFS_TOLERANCE,
    OUTPUT_SR,
    TARGET_LUFS,
    TRUE_PEAK_MAX,
    judge,
)
from relayrun.qa import LOUDNESS_MAX, LOUDNESS_MIN, check_take, qa_thresholds  # noqa: E402


def _md5(path: Path) -> str:
    return hashlib.md5(path.read_bytes()).hexdigest()


def _write(path: Path, pcm: np.ndarray, sr: int, subtype: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    sf.write(str(path), pcm, sr, subtype=subtype)


def _sine(n: int, sr: int, freq: float, amp: float, phase: float = 0.0) -> np.ndarray:
    t = np.arange(n) / float(sr)
    return amp * np.sin(2 * np.pi * freq * t + phase)


def _frame(sr: int) -> int:
    return max(1, int(round(SILENCE_FRAME_S * sr)))


def _stereo(mono: np.ndarray, right: np.ndarray | None = None) -> np.ndarray:
    other = mono if right is None else right
    return np.stack([mono, other], axis=1)


def test_release_gate_constants_are_unchanged():
    thresholds = qa_thresholds()
    assert TARGET_LUFS == -14.0
    assert LUFS_TOLERANCE == 1.0
    assert TRUE_PEAK_MAX == -1.0
    assert OUTPUT_SR == 44100
    assert SILENCE_DB == -60.0
    assert SILENCE_LIMIT_S == 0.5
    assert PEAK_WARN_DBFS == -0.1
    assert LOUDNESS_MIN == -40.0
    assert LOUDNESS_MAX == -8.0
    assert thresholds["release_target_lufs"] == -14.0
    assert thresholds["release_lufs_tolerance"] == 1.0
    assert thresholds["release_true_peak_max_dbtp"] == -1.0
    assert thresholds["raw_loudness_lufs"] == [-40.0, -8.0]
    assert thresholds["raw_loudness_is_release_target"] is False
    assert thresholds["peak_warning_fails_take"] is False
    assert thresholds["clipping_fails_take"] is True
    assert thresholds["intersample_fails_take"] is False
    assert thresholds["normalization_assessed"] is False
    assert thresholds["audio_rewritten"] is False
    assert thresholds["qa_oversample"] == QA_OVERSAMPLE == 4
    assert thresholds["silence_comparison"].startswith("fail when")

    assert judge(-14.0, -1.5, 0.0, 0.0, 44100, "PCM_24", 2)[0]
    assert judge(-15.0, -1.0, 0.49, 0.49, 44100, "PCM_24", 2)[0]
    assert judge(-13.0, -1.0, 0.0, 0.0, 44100, "PCM_24", 2)[0]
    # Exactly -1.0 dBTP is still inside the gate. A hair above it is not.
    assert judge(-14.0, -1.0, 0.0, 0.0, 44100, "PCM_24", 2)[0]
    hot_ok, hot_notes = judge(-14.0, -0.999, 0.0, 0.0, 44100, "PCM_24", 2)
    assert not hot_ok
    assert any("dBTP" in note and "-1.0" in note for note in hot_notes)
    loud_ok, loud_notes = judge(-12.99, -1.5, 0.0, 0.0, 44100, "PCM_24", 2)
    assert not loud_ok
    assert any("LUFS" in note and "-13.0" in note for note in loud_notes)
    quiet_ok, quiet_notes = judge(-15.01, -1.5, 0.0, 0.0, 44100, "PCM_24", 2)
    assert not quiet_ok
    assert any("LUFS" in note for note in quiet_notes)
    # Release edge silence still fails at 0.5 s. QA's own comparison stays ">".
    assert not judge(-14.0, -1.5, 0.5, 0.0, 44100, "PCM_24", 2)[0]
    assert not judge(-14.0, -1.5, 0.0, 0.5, 44100, "PCM_24", 2)[0]


def test_smooth_crest_and_single_sample_are_not_flat_tops():
    sr = 44100
    n = sr
    crest = _sine(n, sr, 440.0, 0.99)
    report = clipping_report(_stereo(crest), sr, "FLOAT")
    assert report["sample_peak_dbfs"] >= PEAK_WARN_DBFS
    assert report["near_full_scale"] is True
    assert report["clipped"] is False
    assert report["flat_top_samples"] < FLAT_TOP_FLOOR
    assert report["oversample"] == 4

    single = _sine(n, sr, 440.0, 0.15)
    single[n // 2] = 0.995
    report = clipping_report(_stereo(single), sr, "FLOAT")
    assert report["near_full_scale"] is True
    assert report["flat_top_samples"] == 1
    assert report["clipped"] is False
    assert report["intersample_distinct"] is False


def test_intersample_overshoot_is_not_sample_clipping():
    sr = 44100
    t = np.arange(sr) / float(sr)
    steep = 0.99 * np.sin(2 * np.pi * 12000 * t)
    report = clipping_report(_stereo(steep), sr, "FLOAT")
    assert report["near_full_scale"] is True
    assert report["clipped"] is False
    assert report["flat_top_samples"] < FLAT_TOP_FLOOR
    assert report["intersample_distinct"] is True
    assert report["oversampled_peak_dbtp"] > report["sample_peak_dbfs"]
    assert report["oversampled_peak_dbtp"] > 0.0
    assert report["intersample_overshoot_db"] > 0.05

    nyquist = 0.99 * np.resize(np.array([1.0, -1.0]), sr)
    report = clipping_report(nyquist, sr, "FLOAT")
    assert report["clipped"] is False
    assert report["flat_top_samples"] == 1
    assert report["oversampled_peak_dbtp"] > 0.0
    assert report["intersample_distinct"] is True
    # The same oversampled peak still fails the release gate. QA does not.
    ok, notes = judge(-14.0, report["oversampled_peak_dbtp"], 0.0, 0.0, 44100, "PCM_24", 2)
    assert not ok
    assert any("true peak" in note for note in notes)


def test_float_and_pcm_flat_tops_follow_format_and_rate(tmp_path):
    sr = 44100
    bed = _sine(sr, sr, 440.0, 0.15)

    def plateau(count: float, value: float = 0.995) -> np.ndarray:
        x = bed.copy()
        start = sr // 2
        x[start:start + count] = value
        return x

    short = tmp_path / "float-short.wav"
    _write(short, _stereo(plateau(2)), sr, "FLOAT")
    row = check_take(short, 1.0)
    assert row["subtype"] == "FLOAT"
    assert row["flat_top_min_samples"] == FLAT_TOP_FLOOR
    assert row["flat_top_samples"] == 2
    assert row["clipped"] is False
    assert row["near_full_scale"] is True
    assert row["verdict"] == "PASS"
    assert row["warnings"]
    assert any("not clipping" in item for item in row["warnings"])
    assert not any("clipping:" in item for item in row["notes"])

    three = tmp_path / "float-flat.wav"
    _write(three, _stereo(plateau(3)), sr, "FLOAT")
    before = _md5(three)
    row = check_take(three, 1.0)
    assert _md5(three) == before
    assert row["clipped"] is True
    assert row["flat_top_samples"] >= 3
    assert row["verdict"] == "FAIL"
    assert any("flat-top" in item and "dBFS" in item for item in row["notes"])
    assert row["warnings"] == []

    right_only = bed.copy()
    right_only[sr // 2:sr // 2 + 8] = 1.0
    sided = tmp_path / "right.wav"
    _write(sided, _stereo(bed, right_only), sr, "FLOAT")
    row = check_take(sided, 1.0)
    assert row["channels"] == 2
    assert row["clipped"] is True
    assert row["flat_top_channel"] == 1

    need = flat_top_min_samples(sr, "PCM_16")
    assert need > FLAT_TOP_FLOOR
    assert flat_top_min_samples(sr, "PCM_24") == FLAT_TOP_FLOOR
    assert flat_top_min_samples(sr, "FLOAT") == FLAT_TOP_FLOOR
    below = tmp_path / "pcm16-below.wav"
    exact = tmp_path / "pcm16-exact.wav"
    _write(below, _stereo(plateau(need - 1, 1.0)), sr, "PCM_16")
    _write(exact, _stereo(plateau(need, 1.0)), sr, "PCM_16")
    under = check_take(below, 1.0)
    over = check_take(exact, 1.0)
    assert under["subtype"] == "PCM_16"
    assert under["flat_top_min_samples"] == need
    assert under["flat_top_samples"] == need - 1
    assert under["clipped"] is False
    assert under["verdict"] == "PASS"
    assert over["flat_top_samples"] >= need
    assert over["clipped"] is True
    assert over["verdict"] == "FAIL"

    pcm24 = tmp_path / "pcm24-flat.wav"
    _write(pcm24, _stereo(plateau(3, 0.999)), sr, "PCM_24")
    row = check_take(pcm24, 1.0)
    assert row["subtype"] == "PCM_24"
    assert row["flat_top_min_samples"] == 3
    assert row["clipped"] is True

    for rate in (22050, 96000):
        count = flat_top_min_samples(rate, "PCM_16")
        assert count >= FLAT_TOP_FLOOR
        tone = _sine(rate, rate, 440.0, 0.15)
        short_run = tone.copy()
        long_run = tone.copy()
        short_run[rate // 3:rate // 3 + count - 1] = 1.0
        long_run[rate // 3:rate // 3 + count] = 1.0
        _write(tmp_path / f"r{rate}-under.wav", short_run, rate, "PCM_16")
        _write(tmp_path / f"r{rate}-over.wav", long_run, rate, "PCM_16")
        under = check_take(tmp_path / f"r{rate}-under.wav", 1.0)
        over = check_take(tmp_path / f"r{rate}-over.wav", 1.0)
        assert under["sample_rate"] == rate
        assert under["channels"] == 1
        assert under["clipped"] is False
        assert over["clipped"] is True
        assert over["flat_top_min_samples"] == count


def test_unclipped_low_frequency_pcm16_crest_is_not_clipping(tmp_path):
    """A 16-bit crest at the warning line can repeat one code. That is not a flat-top."""
    sr = 44100
    n = sr * 2
    t = np.arange(n) / float(sr)
    level = 10 ** (PEAK_WARN_DBFS / 20.0)
    need = flat_top_min_samples(sr, "PCM_16")
    worst = 0
    for amp in (level, 32767 / 32768):
        for phase in (0.0, 0.3, 0.7, 1.1):
            x = amp * np.sin(2 * np.pi * FLAT_TOP_REF_HZ * t + phase)
            path = tmp_path / f"crest-{amp:.4f}-{phase:.1f}.wav"
            _write(path, x, sr, "PCM_16")
            pcm, read_sr = sf.read(str(path), dtype="float64", always_2d=True)
            report = clipping_report(pcm, read_sr, "PCM_16")
            worst = max(worst, report["flat_top_samples"])
            assert report["sample_rate"] == sr
            assert report["clipped"] is False, report
            assert report["flat_top_samples"] < need
            assert report["near_full_scale"] is True
    assert worst >= 3


def test_square_below_warning_level_is_not_clipping():
    sr = 44100
    n = sr // 2
    square = np.resize(np.array([0.5, 0.5, 0.5, 0.5, -0.5, -0.5, -0.5, -0.5]), n)
    # The plateau is real and long. It is below -0.1 dBFS, so it is not a candidate.
    run = 1
    longest = 1
    for left, right in zip(square[:-1], square[1:]):
        run = run + 1 if left == right else 1
        longest = max(longest, run)
    assert longest >= 4
    assert 20 * np.log10(0.5) < PEAK_WARN_DBFS
    report = clipping_report(_stereo(square), sr, "FLOAT")
    assert report["flat_top_samples"] == 0
    assert report["near_full_scale"] is False
    assert report["clipped"] is False


def test_near_full_scale_warning_passes_qa_and_fails_release_gate(tmp_path):
    sr = 44100
    n = sr
    x = _sine(n, sr, 440.0, 0.15)
    x[n // 2] = 10 ** (-0.05 / 20.0)
    path = tmp_path / "hot.wav"
    _write(path, _stereo(x), sr, "FLOAT")
    before = _md5(path)
    row = check_take(path, 1.0)
    assert _md5(path) == before
    assert list(path.parent.glob("*.wav")) == [path]
    assert row["clipped"] is False
    assert row["near_full_scale"] is True
    assert row["peak_dbfs"] >= PEAK_WARN_DBFS
    assert row["loudness_ok"] is True
    assert row["silence_ok"] is True
    assert row["duration_ok"] is True
    assert row["verdict"] == "PASS"
    assert row["notes"] == []
    assert any("near-full-scale" in item and "dBFS" in item for item in row["warnings"])
    assert row["oversampled_peak_dbtp"] > TRUE_PEAK_MAX
    ok, notes = judge(
        -14.0, row["oversampled_peak_dbtp"], 0.0, 0.0, 44100, "PCM_24", 2,
    )
    assert not ok
    assert any("dBTP" in note for note in notes)


def test_zero_crossings_sparse_zeros_and_quiet_passage_are_not_dropouts(tmp_path):
    sr = 44100
    frame = _frame(sr)
    body = 50 * frame
    loud = _sine(body, sr, 440.0, 0.15)
    quiet_n = 50 * frame
    quiet = _sine(quiet_n, sr, 440.0, 10 ** (-50.0 / 20.0))
    hole = np.concatenate([loud, quiet, loud])
    regions = silence_regions(_stereo(hole), sr)
    assert regions["internal_s"] == 0.0
    assert regions["leading_s"] == 0.0
    assert regions["trailing_s"] == 0.0
    assert regions["threshold_dbfs"] == -60.0
    assert regions["context_s"] == SILENCE_CONTEXT_S

    crossed = loud.copy()
    sign = np.sign(crossed)
    sign[sign == 0] = 1
    flips = np.flatnonzero(sign[1:] != sign[:-1])
    crossed[flips] = 0.0
    assert silence_regions(crossed, sr)["internal_s"] == 0.0

    sparse = loud.copy()
    sparse[::100] = 0.0
    assert np.count_nonzero(sparse == 0.0) > 10
    regions = silence_regions(_stereo(sparse), sr)
    assert regions["internal_s"] == 0.0
    path = tmp_path / "sparse.wav"
    _write(path, _stereo(sparse), sr, "FLOAT")
    row = check_take(path, len(sparse) / sr)
    blob = json.dumps(row).lower()
    assert "normalis" not in blob
    assert "normaliz" not in blob
    assert row["clipped"] is False
    assert row["internal_silence_s"] == 0.0
    assert row["verdict"] == "PASS"
    assert qa_thresholds()["normalization_assessed"] is False


def test_edge_silence_context_and_real_dropout(tmp_path):
    sr = 44100
    frame = _frame(sr)

    def blocks(*parts: tuple[int, float]) -> np.ndarray:
        chunks = []
        for frames, amp in parts:
            n = frames * frame
            if amp <= 0.0:
                chunks.append(np.zeros(n))
            else:
                chunks.append(_sine(n, sr, 330.0, amp))
        return np.concatenate(chunks)

    expected = blocks((10, 0.0), (50, 0.15), (10, 0.0))
    regions = silence_regions(expected, sr)
    assert regions["leading_s"] == 0.2
    assert regions["trailing_s"] == 0.2
    assert regions["internal_s"] == 0.0
    path = tmp_path / "edges-ok.wav"
    _write(path, expected, sr, "PCM_24")
    row = check_take(path, len(expected) / sr)
    assert row["silence_ok"] is True
    assert row["verdict"] == "PASS"
    assert not any("dropout" in note for note in row["notes"])

    long_edges = blocks((40, 0.0), (50, 0.15), (40, 0.0))
    regions = silence_regions(_stereo(long_edges), sr)
    assert regions["leading_s"] == 0.8
    assert regions["trailing_s"] == 0.8
    assert regions["internal_s"] == 0.0
    path = tmp_path / "edges-long.wav"
    _write(path, _stereo(long_edges), sr, "FLOAT")
    row = check_take(path, len(long_edges) / sr)
    assert row["silence_ok"] is False
    assert row["verdict"] == "FAIL"
    assert any(note.startswith("leading silence") and "dBFS" in note for note in row["notes"])
    assert any(note.startswith("trailing silence") for note in row["notes"])
    assert not any("dropout" in note for note in row["notes"])

    half = blocks((50, 0.15), (25, 0.0), (50, 0.15))
    regions = silence_regions(half, sr)
    assert regions["internal_s"] == 0.5
    path = tmp_path / "half.wav"
    _write(path, half, sr, "FLOAT")
    row = check_take(path, len(half) / sr)
    assert row["silence_ok"] is True
    assert not any("dropout" in note for note in row["notes"])

    over = blocks((50, 0.15), (26, 0.0), (50, 0.15))
    regions = silence_regions(over, sr)
    assert regions["internal_s"] > SILENCE_LIMIT_S
    assert regions["internal_start_s"] is not None
    path = tmp_path / "dropout.wav"
    _write(path, _stereo(over), sr, "PCM_24")
    row = check_take(path, len(over) / sr)
    assert row["silence_ok"] is False
    assert row["verdict"] == "FAIL"
    assert row["clipped"] is False
    note = next(item for item in row["notes"] if item.startswith("internal dropout"))
    assert "dBFS" in note and "0.020 s" in note and "0.05 s" in note

    low = blocks((50, 0.15), (40, 0.0), (50, 0.15))
    # Replace the digital hole with a -70 dBFS sine, which is still under -60.
    start = 50 * frame
    stop = start + 40 * frame
    low[start:stop] = _sine(stop - start, sr, 220.0, 10 ** (-70.0 / 20.0))
    regions = silence_regions(low, sr)
    assert regions["internal_s"] > SILENCE_LIMIT_S

    spike = np.concatenate([
        np.zeros(40 * frame),
        np.array([0.8]),
        np.zeros(40 * frame - 1),
        _sine(50 * frame, sr, 440.0, 0.15),
    ])
    regions = silence_regions(_stereo(spike), sr)
    assert regions["internal_s"] == 0.0
    assert regions["leading_s"] > SILENCE_LIMIT_S
    path = tmp_path / "spike.wav"
    _write(path, spike, sr, "FLOAT")
    row = check_take(path, len(spike) / sr)
    assert not any("dropout" in note for note in row["notes"])
    assert any(note.startswith("leading silence") for note in row["notes"])


def test_channel_and_rate_silence(tmp_path):
    def hole(sr: int, channels: int) -> np.ndarray:
        frame = _frame(sr)
        loud = _sine(50 * frame, sr, 500.0, 0.15)
        gap = np.zeros(40 * frame)
        mono = np.concatenate([loud, gap, loud])
        if channels == 1:
            return mono
        return _stereo(mono)

    for sr in (22050, 48000, 96000):
        regions = silence_regions(hole(sr, 2), sr)
        assert regions["internal_s"] > SILENCE_LIMIT_S, sr
        assert regions["leading_s"] == 0.0
        assert abs(regions["internal_s"] - 0.8) < 0.03

    sr = 48000
    pcm = hole(sr, 1)
    path = tmp_path / "mono.wav"
    _write(path, pcm, sr, "PCM_24")
    row = check_take(path, len(pcm) / sr)
    assert row["channels"] == 1
    assert row["sample_rate"] == sr
    assert row["internal_silence_s"] > 0.5
    assert row["verdict"] == "FAIL"
    assert row["clipped"] is False

    frame = _frame(44100)
    loud = _sine(80 * frame, 44100, 440.0, 0.15)
    right = loud.copy()
    right[30 * frame:70 * frame] = 0.0
    regions = silence_regions(_stereo(loud, right), 44100)
    assert regions["internal_s"] == 0.0
    both = loud.copy()
    both[30 * frame:70 * frame] = 0.0
    assert silence_regions(_stereo(both), 44100)["internal_s"] > SILENCE_LIMIT_S


def test_very_short_files_do_not_become_internal_dropouts(tmp_path):
    sr = 44100
    tone = tmp_path / "tiny.wav"
    _write(tone, _sine(int(0.01 * sr), sr, 880.0, 0.15), sr, "FLOAT")
    row = check_take(tone, 0.01)
    assert row["internal_silence_s"] == 0.0
    assert row["leading_silence_s"] <= SILENCE_LIMIT_S
    assert row["verdict"] == "FAIL"
    assert any("integrated loudness" in note and "LUFS" in note for note in row["notes"])
    assert not any("dropout" in note or "silence" in note for note in row["notes"])

    silent = tmp_path / "tiny-silent.wav"
    _write(silent, np.zeros(int(0.01 * sr)), sr, "FLOAT")
    row = check_take(silent, 0.01)
    assert row["internal_silence_s"] == 0.0
    assert row["leading_silence_s"] < SILENCE_LIMIT_S
    assert not any("dropout" in note for note in row["notes"])

    regions = silence_regions(np.zeros((16, 1)), 8000)
    assert regions["internal_s"] == 0.0
    assert regions["duration_s"] < SILENCE_LIMIT_S


def test_qa_warns_without_failing_and_keeps_report_semantics(tmp_path):
    from io import StringIO
    from contextlib import redirect_stderr, redirect_stdout

    from relayrun.cli import main

    sr = 44100
    designs = tmp_path / "designs"
    renders = tmp_path / "renders" / "synth"
    take = renders / "01-tone" / "take-1.wav"
    x = _sine(sr, sr, 440.0, 0.15)
    x[sr // 2] = 0.995
    _write(take, _stereo(x), sr, "FLOAT")
    designs.mkdir()
    (designs / "01-tone.json").write_text(
        json.dumps({"title": "Tone", "duration": 1.0}),
        encoding="utf-8",
    )
    report_path = tmp_path / "qa.json"
    out, err = StringIO(), StringIO()
    with redirect_stdout(out), redirect_stderr(err):
        code = main([
            "qa", "--designs", str(designs), "--renders", str(tmp_path / "renders"),
            "--album", "synth", "--report", str(report_path),
        ])
    assert code == 0
    assert "qa WARN" in err.getvalue()
    assert "not clipping" in err.getvalue()
    assert "qa FAIL" not in err.getvalue()
    report = json.loads(report_path.read_text(encoding="utf-8"))
    assert report["verdict"] == "PASS"
    assert report["failures"] == 0
    assert report["thresholds"]["peak_warning_dbfs"] == -0.1
    assert report["thresholds"]["release_true_peak_max_dbtp"] == -1.0
    assert report["thresholds"]["normalization_assessed"] is False
    row = report["rows"][0]
    assert row["clipped"] is False
    assert row["near_full_scale"] is True
    assert row["warnings"]
    out, err = StringIO(), StringIO()
    with redirect_stdout(out), redirect_stderr(err):
        code = main([
            "qa", "--designs", str(designs), "--renders", str(tmp_path / "renders"),
            "--album", "synth", "--format", "tsv", "--dry-run",
        ])
    assert code == 0
    text = out.getvalue()
    assert "near_full_scale" in text and "clipped" in text
    assert "not clipping" in err.getvalue()
    assert "qa FAIL" not in err.getvalue()
