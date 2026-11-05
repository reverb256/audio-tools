"""mixgen tests: QA scoring, Camelot math, cue generation, plan validation,
resumability, acestep argv/poller, and a dry run of the example plan.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from mixgen import cli  # noqa: E402
from mixgen.backends.acestep import (AceStepBackend, AceStepConfig,  # noqa: E402
                                     job_fields, poll_for_take)
from mixgen.backends.acestep_flags import build_argv  # noqa: E402
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
            "job": {"kind": "gpu", "host": "sentry"},
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
                job=Job("gpu", "sentry"))
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
                    "job": {"kind": "gpu", "host": "sentry", **job},
                    "play_bars": 16, "mix_in_bars": 0}],
    }


def test_plan_rejects_unknown_job_host(tmp_path):
    p = tmp_path / "bad.yaml"
    p.write_text(yaml.safe_dump(_plan_dict(host="forge"), sort_keys=False))
    with pytest.raises(PlanError, match="nexus.*sentry|job.host"):
        load_plan(p, tmp_path)
    p.write_text(yaml.safe_dump(_plan_dict(kind="tpu"), sort_keys=False))
    with pytest.raises(PlanError, match="job.kind"):
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
    assert all(t.job.kind == "gpu" and t.job.host == "sentry" for t in plan.tracks)
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


# ------------------------------------------------------------------ acestep
def test_build_argv_field_mapping():
    argv = build_argv({
        "prompt": "deep house", "instrumental": True, "duration": 300,
        "bpm": "120", "key": "8A", "seed": 7, "takes": 1,
    })
    assert argv == ["gpu-music-job", "--prompt", "deep house", "--instrumental",
                    "--duration", "300", "--bpm", "120", "--key", "8A",
                    "--seed", "7", "--takes", "1"]
    with_lyrics = build_argv({"prompt": "p", "lyrics": "la la",
                              "instrumental": False, "duration": 10,
                              "bpm": "90", "key": "4A", "seed": 1, "takes": 1},
                             wrapper="job", extra=["--namespace", "music"])
    assert with_lyrics[:4] == ["job", "--prompt", "p", "--lyrics"]
    assert "--instrumental" not in with_lyrics
    assert with_lyrics[-2:] == ["--namespace", "music"]
    with pytest.raises(ValueError, match="unknown acestep job fields"):
        build_argv({"tempo": 120})


def test_poller_waits_for_fake_submitter(tmp_path):
    state = {"sleeps": 0}

    def sleep(_interval):
        state["sleeps"] += 1
        if state["sleeps"] == 2:
            (tmp_path / "take-2.wav").write_bytes(b"RIFF")

    found = poll_for_take(tmp_path, 2, poll_interval=0, sleep=sleep, clock=lambda: 0.0)
    assert found == tmp_path / "take-2.wav"
    assert state["sleeps"] == 2


def test_poller_honours_optional_timeout(tmp_path):
    now = {"t": 0.0}

    def clock():
        return now["t"]

    def sleep(interval):
        now["t"] += interval

    assert poll_for_take(tmp_path, 9, poll_interval=5, timeout=10,
                         sleep=sleep, clock=clock) is None


def test_acestep_generate_submits_and_collects(tmp_path):
    plan = load_plan(write_plan(tmp_path, takes=1), tmp_path)
    track = plan.tracks[0]
    take_dir = plan.track_gen_dir(track)
    take_dir.mkdir(parents=True)
    submitted: list[list[str]] = []

    def submit(argv):
        submitted.append(argv)
        (take_dir / "take-1.wav").write_bytes(b"RIFFfake")

    cfg = AceStepConfig(wrapper="gpu-music-job", poll_interval=0, timeout=5)
    backend = AceStepBackend(plan, cfg, submit=submit, sleep=lambda _s: None,
                             clock=lambda: 0.0)
    out = tmp_path / "collected.wav"
    backend.generate(track, 1, out)
    assert out.read_bytes() == b"RIFFfake"
    argv = submitted[0]
    assert argv[0] == "gpu-music-job"
    assert "--prompt" in argv and "--instrumental" in argv
    fields = job_fields(track, 1)
    assert fields["bpm"] == "120" and fields["key"] == "8A" and fields["takes"] == 1


def test_acestep_command_template_placeholders(tmp_path):
    plan = load_plan(write_plan(tmp_path, takes=1), tmp_path)
    track = plan.tracks[0]
    cfg = AceStepConfig(command="submit --text {prompt} --len {duration} --out {out}")
    backend = AceStepBackend(plan, cfg, submit=lambda argv: None)
    argv = backend.argv_for(track, 1, Path("/music/gen/x/y/take-1.wav"))
    assert argv[0] == "submit"
    assert argv[1] == "--text" and "deep house" in argv[2]
    assert "/music/gen/x/y/take-1.wav" in argv
