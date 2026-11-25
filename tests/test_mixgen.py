"""mixgen tests: QA scoring, Camelot math, cue generation, plan validation,
resumability, and a dry run of the example plan.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from mixgen import cli  # noqa: E402
from mixgen.backends.fake import synthesize  # noqa: E402
from mixgen.cuegen import build_cue, chapters, timeline, total_duration  # noqa: E402
from mixgen.plan import (Job, PlanError, Track, camelot_of,  # noqa: E402
                         camelot_to_tonic, load_plan)
from mixgen.qa import analyze_take, key_to_camelot, qa_track  # noqa: E402
from mixgen.stages import run  # noqa: E402

REPO = Path(__file__).resolve().parent.parent
EXAMPLE = REPO / "examples" / "lightworld-relay.plan.yaml"


def write_plan(tmp_path: Path, name: str = "plan.yaml", **overrides) -> Path:
    data = {
        "title": "Test Mix",
        "slug": "test-mix",
        "output": "mixes/test-mix",
        "takes": 2,
        "backend": "fake",
        "tracks": [{
            "title": "Test Track",
            "slug": "test-track",
            "style": "deep house, 120 BPM, A minor. 16-bar drums-only intro",
            "bpm": 120,
            "key": "8A",
            "duration": "1:36",
            "instrumental": True,
            "takes": 2,
            "job": {"kind": "gpu", "host": "zephyr"},
            "play_bars": 32,
            "mix_in_bars": 0,
        }],
    }
    data.update(overrides)
    p = tmp_path / name
    p.write_text(yaml.safe_dump(data, sort_keys=False))
    return p


def make_track(**kw) -> Track:
    base = dict(index=0, title="Test Track", slug="test-track", style="style",
                bpm=120.0, key="8A", duration=96.0, instrumental=True, takes=4,
                job=Job("gpu", "zephyr"))
    base.update(kw)
    return Track(**base)


# ------------------------------------------------------------------ Camelot
def test_camelot_conversion_roundtrip():
    assert camelot_of("A", "min") == "8A"
    assert camelot_of("E", "maj") == "12B"
    assert camelot_to_tonic("8A") == ("A", "min")
    assert camelot_to_tonic("12B") == ("E", "maj")
    seen = set()
    for num in range(1, 13):
        for letter in ("A", "B"):
            cam = f"{num}{letter}"
            tonic, mode = camelot_to_tonic(cam)
            assert camelot_of(tonic, mode) == cam
            seen.add(cam)
    assert len(seen) == 24
    assert key_to_camelot("Am") == "8A"
    assert key_to_camelot("F# minor") == "11A"
    assert key_to_camelot("E major") == "12B"


# ------------------------------------------------------------------ QA scoring
def test_qa_prefers_correct_take(tmp_path):
    track = make_track()
    files = {
        "correct": synthesize(tmp_path / "correct.wav", bpm=120, camelot="8A",
                              duration=96.0, seed=3),
        "wrong": synthesize(tmp_path / "wrong.wav", bpm=138, camelot="8A",
                            duration=96.0, seed=3),
        "clipped": synthesize(tmp_path / "clipped.wav", bpm=120, camelot="8A",
                              duration=96.0, seed=3, clipped=True),
        "nointro": synthesize(tmp_path / "nointro.wav", bpm=120, camelot="8A",
                              duration=96.0, seed=3, intro_bars=0),
    }
    results = {name: analyze_take(path, track, i + 1) for i, (name, path) in enumerate(files.items())}
    good = results["correct"]
    assert good["bpm_detected"] == pytest.approx(120, abs=0.5)
    assert good["camelot_detected"] == "8A"
    assert good["drums_intro_bars"] >= 16 and good["drums_outro_bars"] >= 16
    assert not good["clipped"]
    for name in ("wrong", "clipped", "nointro"):
        assert good["score"] > results[name]["score"], (name, good["score"], results[name]["score"])
    assert results["wrong"]["scores"]["tempo"] == 0.0
    assert results["clipped"]["clipped"] is True
    assert results["nointro"]["drums_intro_bars"] == 0

    report = qa_track(track, list(files.values()), tmp_path / "qa")
    assert report["pick"] == "take-1"
    assert (tmp_path / "qa" / "report.json").exists()
    assert (tmp_path / "qa" / "report.md").exists()


# ------------------------------------------------------------------ cue math
def _cue_plan(tmp_path: Path):
    data = {
        "title": "Cue Test",
        "slug": "cue-test",
        "output": "mixes/cue-test",
        "phrase_bars": 16,
        "tracks": [
            {"title": "A", "style": "s", "bpm": 120, "key": "8A", "duration": "1:00",
             "instrumental": True, "takes": 1, "job": {"kind": "cpu", "host": "nexus"},
             "play_bars": 160, "mix_in_bars": 0, "ramp": False},
            {"title": "B", "style": "s", "bpm": 128, "key": "11A", "duration": "1:00",
             "instrumental": True, "takes": 1, "job": {"kind": "cpu", "host": "nexus"},
             "play_bars": 192, "mix_in_bars": 32, "transition": "long_blend", "ramp": True},
            {"title": "C", "style": "s", "bpm": 124, "key": "9A", "duration": "1:00",
             "instrumental": True, "takes": 1, "job": {"kind": "cpu", "host": "nexus"},
             "play_bars": 64, "mix_in_bars": 16, "transition": "long_blend", "ramp": False},
            {"title": "D", "style": "s", "bpm": 130, "key": "12A", "duration": "1:00",
             "instrumental": True, "takes": 1, "job": {"kind": "cpu", "host": "nexus"},
             "play_bars": 32, "mix_in_bars": 32, "transition": "cut", "ramp": False},
        ],
    }
    p = tmp_path / "cue.yaml"
    p.write_text(yaml.safe_dump(data, sort_keys=False))
    return load_plan(p, tmp_path)


def test_cue_bar_math_timestamps_and_chapters(tmp_path):
    plan = _cue_plan(tmp_path)
    slots = timeline(plan)
    # A: bars 0..160; B starts 32 bars before A ends; C 16 bars before B ends;
    # D is a cut, so it starts exactly when C ends.
    assert [(s.out_bar0, s.bars, s.overlap_in) for s in slots] == [
        (0, 160, 0), (128, 192, 32), (304, 64, 16), (368, 32, 0)]
    # A owns bars 0..128 at 120 BPM: 128 bars * 4 beats * 0.5 s = 256 s.
    assert slots[1].start_s == pytest.approx(256.0, abs=1e-6)
    # B's 32-bar overlap with A ramps 120 -> 128 over 128 beats, then holds 128
    # for 576 beats into C (C's own overlap does not ramp). 256 + ramp + 270 s.
    ramp = sum(60.0 / (120 + 8 * (k + 0.5) / 128) for k in range(128))
    assert slots[2].start_s == pytest.approx(256 + ramp + 576 * 60 / 128, abs=1e-6)
    assert slots[3].start_s == pytest.approx(slots[2].end_s, abs=1e-6)
    # C's first 16 bars are held at B's 128 BPM; the remaining 48 at 124 BPM.
    c_len = 16 * 4 * 60 / 128 + 48 * 4 * 60 / 124
    assert slots[2].end_s - slots[2].start_s == pytest.approx(c_len, abs=1e-6)
    assert total_duration(plan) == pytest.approx(slots[3].end_s)

    cue = build_cue(plan)
    assert cue["tracks"][0]["mix_in_bars"] == 0
    assert cue["tracks"][3]["mix_in_bars"] == 0          # forced 0 by 'cut'
    assert cue["tracks"][1]["file"] == "mixes/cue-test/mastered/b.wav"
    assert cue["output"] == "mixes/cue-test/cue-test.wav"

    text = chapters(plan)
    assert text.splitlines() == [
        "00:00 A", "04:16 B", "09:47 C", "11:50 D"]


# ------------------------------------------------------------------ validation
def _plan_dict(**job):
    return {
        "title": "T", "slug": "t", "output": "mixes/t", "backend": "fake",
        "tracks": [{"title": "X", "style": "s", "bpm": 120, "key": "8A",
                    "duration": "1:00", "instrumental": True, "takes": 1,
                    "job": {"kind": "gpu", "host": "zephyr", **job},
                    "play_bars": 16, "mix_in_bars": 0}],
    }


def test_plan_rejects_unknown_job_host(tmp_path):
    p = tmp_path / "bad.yaml"
    p.write_text(yaml.safe_dump(_plan_dict(host="forge"), sort_keys=False))
    with pytest.raises(PlanError, match="nexus.*zephyr|job.host"):
        load_plan(p, tmp_path)
    p.write_text(yaml.safe_dump(_plan_dict(kind="tpu"), sort_keys=False))
    with pytest.raises(PlanError, match="job.kind"):
        load_plan(p, tmp_path)
    p.write_text(yaml.safe_dump(_plan_dict(host="nexus"), sort_keys=False))
    with pytest.raises(PlanError, match="gpu@zephyr only"):
        load_plan(p, tmp_path)


def test_plan_rejects_path_escape(tmp_path):
    root = tmp_path / "music"
    root.mkdir()
    data = _plan_dict()
    data["output"] = "../../etc"
    p = tmp_path / "escape.yaml"
    p.write_text(yaml.safe_dump(data, sort_keys=False))
    with pytest.raises(PlanError, match="escapes root"):
        load_plan(p, root)
    data["output"] = "/etc/passwd"
    p.write_text(yaml.safe_dump(data, sort_keys=False))
    with pytest.raises(PlanError, match="escapes root"):
        load_plan(p, root)


def test_plan_loads_example_and_has_all_tracks():
    plan = load_plan(EXAMPLE, "/tmp/unused-root")
    assert len(plan.tracks) == 12
    assert plan.tracks[0].title == "Heliostat Dawn"
    assert plan.tracks[10].transition == "cut" and plan.tracks[10].mix_in_bars == 0
    assert all(t.job.kind == "gpu" and t.job.host == "zephyr" for t in plan.tracks)
    assert all(t.takes == 3 for t in plan.tracks)


# ------------------------------------------------------------------ resumability
def test_second_run_skips_existing_outputs(tmp_path):
    plan_path = write_plan(tmp_path, takes=2)
    plan = load_plan(plan_path, tmp_path)
    first = run(plan, ["gen", "qa"], backend_name="fake", log=lambda m: None)
    assert first["gen"] == {"generated": 2, "skipped": 0, "backend": "fake"}
    assert first["qa"]["analysed"] == 1
    report = tmp_path / "mixes/test-mix/qa/test-track/report.json"
    assert report.exists()
    mtime = report.stat().st_mtime_ns

    second = run(load_plan(plan_path, tmp_path), ["gen", "qa"], backend_name="fake",
                 log=lambda m: None)
    assert second["gen"] == {"generated": 0, "skipped": 2, "backend": "fake"}
    assert second["qa"]["analysed"] == 0 and second["qa"]["skipped"] == 1
    assert report.stat().st_mtime_ns == mtime


# ------------------------------------------------------------------ dry run
def test_dry_run_example_prints_all_stages(tmp_path, capsys):
    code = cli.main([str(EXAMPLE), "--root", str(tmp_path), "--dry-run"])
    out = capsys.readouterr().out
    assert code == 0
    for stage in ("gen:", "qa:", "master:", "mix:"):
        assert stage in out
    assert "Heliostat Dawn" in out and "take-1: RUN" in out
    assert "00:00  Heliostat Dawn" in out  # mix timeline / chapters preview
    assert not (tmp_path / "gen").exists()
    assert not (tmp_path / "mixes").exists()


# ------------------------------------------------------------------ stems
def test_stems_stage_is_opt_in(tmp_path):
    from mixgen.stages import run_stems

    plan = load_plan(write_plan(tmp_path, takes=2), tmp_path)
    assert plan.stems == {}
    with pytest.raises(PlanError, match="opt-in"):
        run_stems(plan, force=False, log=lambda _m: None)

    enabled = load_plan(write_plan(tmp_path, name="stems.yaml", takes=2,
                                   stems={"enabled": True, "model": "htdemucs"}), tmp_path)
    assert enabled.stems == {"enabled": True, "model": "htdemucs"}
    assert enabled.stems_rel(enabled.tracks[0]) == "test-mix/test-track-stems"


def test_plan_rejects_removed_acestep_backend(tmp_path):
    path = write_plan(tmp_path, name="ace.yaml", backend="acestep")
    with pytest.raises(PlanError, match="ACE-Step backend was removed"):
        load_plan(path, tmp_path)
    other = tmp_path / "ace-block.yaml"
    data = yaml.safe_load(path.read_text())
    data["backend"] = "fake"
    data["acestep"] = {"wrapper": "gpu-music-job"}
    other.write_text(yaml.safe_dump(data))
    with pytest.raises(PlanError, match="ACE-Step backend was removed"):
        load_plan(other, tmp_path)
