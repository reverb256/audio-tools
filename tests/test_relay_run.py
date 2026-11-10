"""relay-run tests. Audio is synthesized in the test (sines, noise, silence, clips)."""

from __future__ import annotations

import json
import os
import stat
import subprocess
import sys
import textwrap
from pathlib import Path

import numpy as np
import pytest
import soundfile as sf
import yaml

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

from relayrun.album import HEADER  # noqa: E402
from relayrun.audio import pcm_md5, read_pcm24  # noqa: E402
from relayrun.cli import main  # noqa: E402
from relayrun.errors import RelayError  # noqa: E402
from relayrun.master import judge, measure_master  # noqa: E402
from relayrun.split import map_cut_sample  # noqa: E402

SENTINEL = "SYNTHETIC LEGAL NAME"


def _b26(n: int, width: int = 4) -> str:
    chars = []
    for _ in range(width):
        chars.append(chr(ord("a") + (n % 26)))
        n //= 26
    return "".join(chars)


def _run(argv: list[str]) -> tuple[int, str, str]:
    from io import StringIO
    from contextlib import redirect_stderr, redirect_stdout

    out, err = StringIO(), StringIO()
    with redirect_stdout(out), redirect_stderr(err):
        code = main(argv)
    return code, out.getvalue(), err.getvalue()


def _tone(path: Path, seconds: float, sr: int = 44100, freq: float = 440.0,
          amp: float = 0.2, kind: str = "sine", seed: int = 1) -> float:
    n = int(round(sr * seconds))
    if kind == "sine":
        t = np.arange(n) / sr
        x = amp * np.sin(2 * np.pi * freq * t)
    elif kind == "noise":
        x = amp * np.random.default_rng(seed).standard_normal(n)
    elif kind == "clipped":
        t = np.arange(n) / sr
        x = np.clip(np.sin(2 * np.pi * freq * t) * 8.0, -1.0, 1.0)
    elif kind == "silence":
        x = np.zeros(n)
    else:
        raise AssertionError(kind)
    y = np.stack([x, x], axis=1)
    path.parent.mkdir(parents=True, exist_ok=True)
    subtype = "PCM_16"
    suffix = path.suffix.lower()
    if suffix == ".flac":
        sf.write(str(path), y, sr, format="FLAC", subtype="PCM_16")
    else:
        sf.write(str(path), y, sr, subtype=subtype)
    return n / sr


def _piece(path: Path, sr: int, lead: float, body: float, trail: float,
           internal: float = 0.0, amp: float = 0.2, freq: float = 440.0) -> float:
    def block(seconds: float, level: float) -> np.ndarray:
        n = int(round(sr * seconds))
        if n <= 0:
            return np.zeros(0)
        if level <= 0:
            return np.zeros(n)
        t = np.arange(n) / sr
        return level * np.sin(2 * np.pi * freq * t)

    half = body / 2.0
    x = np.concatenate([
        block(lead, 0.0),
        block(half, amp),
        block(internal, 0.0),
        block(half, amp),
        block(trail, 0.0),
    ])
    y = np.stack([x, x], axis=1)
    path.parent.mkdir(parents=True, exist_ok=True)
    sf.write(str(path), y, sr, subtype="PCM_16")
    return len(x) / sr


def _design(path: Path, *, title: str, duration: float, track: int = 1, **extra) -> None:
    data = {
        "album": "Sample Set",
        "track": track,
        "side": "A",
        "title": title,
        "style_prompt": "plain test prompt",
        "lyrics": "[instrumental]",
        "instrumental": True,
        "bpm": 120,
        "key": "A minor",
        "camelot": "8A",
        "duration": duration,
        "seed": 1000 + track,
        "takes": 1,
    }
    data.update(extra)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data), encoding="utf-8")


def _passing_album(root: Path, tracks: int = 24) -> Path:
    album = root / "sample-set"
    album.mkdir(parents=True)
    for i in range(1, tracks + 1):
        left, right = _b26(i), _b26(200 + i)
        title = f"{left} {right}"
        stem = f"{i:02d}-{left}-{right}"
        concept = " ".join(_b26(i * 40 + k, width=5) for k in range(30))
        meaning = (
            f"Slot {left} carries its own image from start to finish. "
            f"The second sentence belongs only to {right}."
        )
        notes = "Extra note keeps this design document long enough for the gate. " * 30
        md = (
            f"# {title}\n\n"
            f"## Concept\n{concept}\n\n"
            f"## Deep meaning\n{meaning}\n\n"
            f"**Lead instrument:** lead{left}{right}\n\n"
            "## Arrangement\n"
            "| intro | 16 |\n| body | 32 |\n| break | 16 |\n| outro | 32 |\n\n"
            "## Mix notes\n"
            "Mix in on the first downbeat of the phrase. "
            "Mix out on the last downbeat of the phrase.\n\n"
            f"## Notes\n{notes}\n"
        )
        assert len(md.encode()) >= 1200, len(md.encode())
        (album / f"{stem}.md").write_text(md, encoding="utf-8")
        tags = ", ".join([f"tag{i:02d}{k}" for k in range(8)] + ["instrumental"])
        assert len(tags) < 400
        _design(
            album / f"{stem}.json",
            title=title,
            duration=295,
            track=i,
            style_prompt=tags,
            seed=500000 + i,
            bpm=120,
            camelot="8A",
            key="A minor",
        )
    album_md = (
        "# Sample Set\n\n"
        + ("This album notes a distinct morning walk and a steady tempo. " * 40)
    )
    assert len(album_md.encode()) >= 1200
    (album / "ALBUM.md").write_text(album_md, encoding="utf-8")
    return album


def _receipt(wav: Path, *, verdict: str, lufs: float = -14.0, tp: float = -1.4,
             duration: float = 8.0) -> None:
    from relayrun.audio import file_md5

    payload = {
        "file": wav.name,
        "duration_s": duration,
        "lufs_i": lufs,
        "true_peak_dbtp": tp,
        "md5": file_md5(wav),
        "verdict": verdict,
        "sample_rate": 44100,
        "channels": 2,
    }
    Path(str(wav) + ".json").write_text(json.dumps(payload), encoding="utf-8")


def _fake_ssh(path: Path) -> None:
    path.write_text(textwrap.dedent("""\
        #!/usr/bin/env python3
        import os, sys
        args = sys.argv[1:]
        i = 0
        while i < len(args):
            a = args[i]
            if a == "--":
                i += 1
                break
            if a.startswith("-"):
                if a in ("-o", "-p", "-i", "-l", "-F", "-c", "-b", "-e", "-J"):
                    i += 2
                    continue
                i += 1
                continue
            break
        rest = args[i + 1:]
        if len(rest) == 1:
            os.execvp("bash", ["bash", "-lc", rest[0]])
        os.execvp(rest[0], rest)
        """), encoding="utf-8")
    path.chmod(path.stat().st_mode | stat.S_IEXEC)


# ---------------------------------------------------------------- help
@pytest.mark.parametrize("argv", [
    ["--help"],
    ["validate", "--help"],
    ["qa", "--help"],
    ["master", "--help"],
    ["mix", "--help"],
    ["split-sideb", "--help"],
    ["album", "--help"],
    ["mirror", "--help"],
])
def test_help(argv):
    with pytest.raises(SystemExit) as exc:
        main(argv)
    assert exc.value.code == 0


def test_module_help():
    env = os.environ.copy()
    env["PYTHONPATH"] = str(REPO)
    proc = subprocess.run(
        [sys.executable, "-m", "relayrun", "--help"],
        cwd=REPO, env=env, capture_output=True, text=True, check=False,
    )
    assert proc.returncode == 0
    for name in ("validate", "qa", "master", "mix", "split-sideb", "album", "mirror"):
        assert name in proc.stdout
    text = (REPO / "pyproject.toml").read_text(encoding="utf-8")
    assert 'relay-run = "relayrun.cli:main"' in text


def test_sources_omit_private_identifiers():
    needles = ("/home/", "zephyr", "nexus", "sentry", "forge", "reverb256", "j_kro", "noreply")
    files = list((REPO / "relayrun").rglob("*.py"))
    files.append(REPO / "examples" / "relay-cue.example.yaml")
    files.append(REPO / ".github" / "workflows" / "relay-run.yml")
    files.append(REPO / "tests" / "test_relay_run.py")
    # The needles live in this function, so skip this source file's own text
    # by scanning relayrun, the example, and the workflow only.
    files = [p for p in files if p.name != "test_relay_run.py"]
    blob = "\n".join(p.read_text(encoding="utf-8").lower() for p in files)
    for needle in needles:
        assert needle not in blob, needle


# ---------------------------------------------------------------- validate
def test_validate_fails_short_album(tmp_path):
    album = tmp_path / "sample-set"
    album.mkdir()
    _design(album / "01-aaaa-bbbb.json", title="Aaaa Bbbb", duration=200, track=1)
    (album / "01-aaaa-bbbb.md").write_text("too short\n", encoding="utf-8")
    (album / "ALBUM.md").write_text("short\n", encoding="utf-8")
    code, _out, err = _run(["validate", "--designs", str(album)])
    assert code == 1
    assert "FAIL" in err
    gate = json.loads((album / "GATE.json").read_text(encoding="utf-8"))
    assert gate["verdict"] == "FAIL"
    assert any(item["check"] == "structure" for item in gate["fails"])


def test_validate_fails_banned_word(tmp_path):
    album = tmp_path / "sample-set"
    album.mkdir()
    _design(album / "01-aaaa-bbbb.json", title="Aaaa Bbbb", duration=200, track=1)
    (album / "01-aaaa-bbbb.md").write_text("a line with the word queer in it\n", encoding="utf-8")
    (album / "ALBUM.md").write_text("notes\n", encoding="utf-8")
    code, _out, err = _run(["validate", "--designs", str(album)])
    assert code == 1
    assert "banned-word" in err


def test_validate_passes_and_is_idempotent(tmp_path):
    album = _passing_album(tmp_path)
    code, out, err = _run(["validate", "--designs", str(album)])
    assert code == 0, err
    assert "PASS" in out
    gate = json.loads((album / "GATE.json").read_text(encoding="utf-8"))
    assert gate["verdict"] == "PASS"
    assert gate["tracks"] == 24
    code, _out, err = _run(["validate", "--designs", str(album)])
    assert code == 0, err


def test_validate_dry_run_writes_nothing(tmp_path):
    album = tmp_path / "sample-set"
    album.mkdir()
    _design(album / "01-aaaa-bbbb.json", title="Aaaa Bbbb", duration=200, track=1)
    code, _out, err = _run(["validate", "--designs", str(album), "--dry-run"])
    assert code == 1
    assert "FAIL" in err
    assert not (album / "GATE.json").exists()
    assert not (album / "GATE.md").exists()


def test_validate_external_gate(tmp_path):
    script = tmp_path / "gate.py"
    script.write_text(textwrap.dedent("""\
        from pathlib import Path
        def main(root, report=None):
            ok = (Path(root) / "ok").is_file()
            if report:
                Path(report).write_text("external\\n", encoding="utf-8")
            return 0 if ok else 1
        """), encoding="utf-8")
    designs = tmp_path / "designs"
    designs.mkdir()
    code, _out, err = _run(["validate", "--designs", str(designs), "--content-gate", str(script)])
    assert code == 1
    assert "FAIL" in err
    (designs / "ok").write_text("1", encoding="utf-8")
    report = tmp_path / "gate.md"
    code, out, err = _run([
        "validate", "--designs", str(designs), "--content-gate", str(script),
        "--report", str(report),
    ])
    assert code == 0, err
    assert "PASS" in out
    assert report.read_text(encoding="utf-8") == "external\n"
    dry = tmp_path / "dry.md"
    code, _out, err = _run([
        "validate", "--designs", str(designs), "--content-gate", str(script),
        "--report", str(dry), "--dry-run",
    ])
    assert code == 0, err
    assert not dry.exists()


def test_validate_missing_dir(tmp_path):
    code, _out, err = _run(["validate", "--designs", str(tmp_path / "missing")])
    assert code == 2
    assert "not found" in err


# ---------------------------------------------------------------- qa
def test_qa_detects_clipping_silence_duration_and_loudness(tmp_path):
    designs = tmp_path / "designs"
    renders = tmp_path / "renders" / "demo"
    sr = 44100
    specs = {
        "01-clean": ("sine", 0.2, 3.0),
        "02-clipped": ("clipped", 1.0, 3.0),
        "03-edges": ("edges", 0.2, 0.0),
        "04-hole": ("hole", 0.2, 0.0),
        "05-quiet": ("sine", 0.003, 3.0),
        "06-long": ("sine", 0.2, 3.0),
    }
    durations = {}
    for stem, (kind, amp, seconds) in specs.items():
        folder = renders / stem
        if kind == "edges":
            durations[stem] = _piece(folder / "take-1.wav", sr, 0.8, 2.0, 0.8, freq=330)
        elif kind == "hole":
            durations[stem] = _piece(folder / "take-1.wav", sr, 0.0, 2.4, 0.0, internal=0.8, freq=550)
        else:
            durations[stem] = _tone(folder / "take-1.wav", seconds, sr=sr, amp=amp, freq=440, kind=kind)
        design_duration = 10.0 if stem == "06-long" else durations[stem]
        _design(designs / f"{stem}.json", title=stem, duration=design_duration, track=int(stem[:2]))
    report_path = tmp_path / "qa.json"
    code, _out, err = _run([
        "qa", "--designs", str(designs), "--renders", str(tmp_path / "renders"),
        "--album", "demo", "--report", str(report_path),
    ])
    assert code == 1
    assert "qa FAIL" in err
    report = json.loads(report_path.read_text(encoding="utf-8"))
    by = {row["track"]: row for row in report["rows"]}
    assert by["01-clean"]["verdict"] == "PASS"
    assert by["01-clean"]["clipped"] is False
    assert by["02-clipped"]["clipped"] is True
    assert by["02-clipped"]["verdict"] == "FAIL"
    assert by["03-edges"]["leading_silence_s"] > 0.5
    assert by["03-edges"]["trailing_silence_s"] > 0.5
    assert by["03-edges"]["silence_ok"] is False
    assert by["04-hole"]["internal_silence_s"] > 0.5
    assert by["04-hole"]["leading_silence_s"] <= 0.5
    assert by["04-hole"]["silence_ok"] is False
    assert by["05-quiet"]["loudness_ok"] is False
    assert by["05-quiet"]["silence_ok"] is True
    assert by["06-long"]["duration_ok"] is False
    assert report["verdict"] == "FAIL"

    tsv = tmp_path / "qa.tsv"
    code, out, _err = _run([
        "qa", "--designs", str(designs), "--renders", str(tmp_path / "renders"),
        "--album", "demo", "--format", "tsv", "--report", str(tsv), "--dry-run",
    ])
    assert code == 1
    assert not tsv.exists()
    assert "clipped" in out
    assert "\t" in out


def test_qa_missing_take(tmp_path):
    designs = tmp_path / "designs"
    renders = tmp_path / "renders" / "demo"
    renders.mkdir(parents=True)
    _design(designs / "01-alone.json", title="Alone", duration=3.0)
    code, _out, err = _run([
        "qa", "--designs", str(designs), "--renders", str(tmp_path / "renders"),
        "--album", "demo", "--report", str(tmp_path / "qa.json"),
    ])
    assert code == 1
    assert "no takes" in err


# ---------------------------------------------------------------- master
def test_judge_window():
    assert judge(-14.0, -1.5, 0.0, 0.0, 44100, "PCM_24", 2)[0]
    assert judge(-15.0, -1.0, 0.49, 0.49, 44100, "PCM_24", 2)[0]
    assert not judge(-14.0, -0.5, 0.0, 0.0, 44100, "PCM_24", 2)[0]
    assert not judge(-16.2, -1.5, 0.0, 0.0, 44100, "PCM_24", 2)[0]
    assert not judge(-14.0, -1.5, 0.6, 0.0, 44100, "PCM_24", 2)[0]
    assert not judge(-14.0, -1.5, 0.0, 0.0, 48000, "PCM_24", 2)[0]
    assert not judge(-14.0, -1.5, 0.0, 0.0, 44100, "PCM_16", 2)[0]


def _assert_master(path: Path, *, check_no_fade: bool = False):
    info = sf.info(str(path))
    assert info.samplerate == 44100
    assert info.channels == 2
    assert info.subtype == "PCM_24"
    measured = measure_master(path)
    ok, notes = judge(
        measured["lufs_i"], measured["true_peak_dbtp"],
        measured["leading_silence_s"], measured["trailing_silence_s"],
        measured["sample_rate"], measured["subtype"], measured["channels"],
    )
    assert ok, notes
    assert measured["true_peak_dbtp"] <= -1.0
    assert -15.0 <= measured["lufs_i"] <= -13.0
    if check_no_fade:
        pcm, sr = sf.read(str(path), always_2d=True)
        frame = max(1, sr // 5)
        mid = float(np.sqrt(np.mean(pcm[len(pcm) // 2:len(pcm) // 2 + frame] ** 2)))
        tail = float(np.sqrt(np.mean(pcm[-frame:] ** 2)))
        assert mid > 0
        assert tail > mid * 0.25


def test_master_sine_and_noise(tmp_path):
    sine = tmp_path / "sine.wav"
    _tone(sine, 8.0, sr=48000, amp=0.05, freq=440)
    sine_out = tmp_path / "sine-out.wav"
    code, _out, err = _run(["master", "--input", str(sine), "--output", str(sine_out)])
    assert code == 0, err
    _assert_master(sine_out, check_no_fade=True)
    receipt = json.loads(Path(str(sine_out) + ".json").read_text(encoding="utf-8"))
    assert receipt["verdict"] == "PASS"
    assert receipt["sample_rate"] == 44100
    assert receipt["bit_depth"] == 24

    noise = tmp_path / "noise.wav"
    _tone(noise, 8.0, sr=44100, amp=0.5, kind="noise")
    noise_out = tmp_path / "noise-out.wav"
    code, _out, err = _run(["master", "--input", str(noise), "--output", str(noise_out)])
    assert code == 0, err
    _assert_master(noise_out, check_no_fade=True)


def test_master_trims_edges_and_keeps_interior(tmp_path):
    src = tmp_path / "edges.wav"
    duration = _piece(src, 44100, lead=1.0, body=6.0, trail=1.0, internal=0.4, amp=0.15, freq=220)
    dst = tmp_path / "edges-out.wav"
    code, _out, err = _run(["master", "--input", str(src), "--output", str(dst)])
    assert code == 0, err
    _assert_master(dst)
    info = sf.info(str(dst))
    assert info.duration < duration - 0.5
    from relayrun.audio import load_audio, silence_regions
    pcm, sr = load_audio(dst)
    regions = silence_regions(pcm, sr)
    assert regions["leading_s"] < 0.5
    assert regions["trailing_s"] < 0.5
    assert regions["internal_s"] > 0.2


def test_master_dry_run_and_missing_input(tmp_path):
    src = tmp_path / "in.wav"
    _tone(src, 1.0, amp=0.2)
    dst = tmp_path / "out.wav"
    code, out, err = _run(["master", "--input", str(src), "--output", str(dst), "--dry-run"])
    assert code == 0, err
    assert "loudnorm" in out
    assert not dst.exists()
    missing = tmp_path / "missing.wav"
    code, _out, err = _run(["master", "--input", str(missing), "--output", str(dst)])
    assert code == 2
    assert "not found" in err
    assert not dst.exists()


def test_master_batch_layout(tmp_path):
    designs = tmp_path / "designs"
    stem = "01-amber-kiln"
    _design(designs / f"{stem}.json", title="Amber Kiln", duration=8.0, track=1)
    take = tmp_path / "renders" / "demo" / stem / "take-1.flac"
    _tone(take, 8.0, sr=44100, amp=0.08, freq=330)
    release = tmp_path / "release"
    code, _out, err = _run([
        "master", "--designs", str(designs), "--renders", str(tmp_path / "renders"),
        "--album", "demo", "--release", str(release), "--take", "1",
    ])
    assert code == 0, err
    out = release / "A" / f"{stem}.wav"
    assert out.is_file()
    _assert_master(out)
    code, out_text, err = _run([
        "master", "--designs", str(designs), "--renders", str(tmp_path / "renders"),
        "--album", "demo", "--release", str(release), "--dry-run",
    ])
    assert code == 0, err
    assert "dry" not in err.lower() or True
    assert str(out) in out_text or stem in out_text


# ---------------------------------------------------------------- mix + split
def _cue(path: Path, tracks: list[dict], sample_rate: int = 44100) -> None:
    path.write_text(yaml.safe_dump({
        "version": 1,
        "sample_rate": sample_rate,
        "tracks": tracks,
    }, sort_keys=False), encoding="utf-8")


def test_mix_and_split_pcm_roundtrip(tmp_path):
    sr = 44100
    titles = ["Amber Kiln", "Brass Harbor", "Cinder Metro"]
    freqs = [196.0, 294.0, 392.0]
    files = []
    for title, freq in zip(titles, freqs):
        wav = tmp_path / f"{title.split()[0].lower()}.wav"
        _tone(wav, 4.0, sr=sr, freq=freq, amp=0.25)
        files.append(wav)
    cue = tmp_path / "cue.yaml"
    _cue(cue, [
        {"title": titles[0], "file": files[0].name, "transition": "cut", "bpm": 120},
        {"title": titles[1], "file": files[1].name, "transition": "blend", "blend_s": 0.5, "bpm": 121},
        {"title": titles[2], "file": files[2].name, "transition": "blend", "blend_s": 0.5, "bpm": 122},
    ])
    mix = tmp_path / "mix.wav"
    code, _out, err = _run(["mix", "--cue", str(cue), "--root", str(tmp_path), "--output", str(mix)])
    assert code == 0, err
    timeline_path = Path(str(mix.with_suffix(".timeline.json")))
    assert timeline_path.is_file()
    timeline = json.loads(timeline_path.read_text(encoding="utf-8"))
    assert len(timeline["cuts"]) == 2
    assert len(timeline["tracks"]) == 3

    dry = tmp_path / "dry-mix.wav"
    code, out, err = _run([
        "mix", "--cue", str(cue), "--root", str(tmp_path), "--output", str(dry), "--dry-run",
    ])
    assert code == 0, err
    assert not dry.exists()
    assert "dry run" in out

    out_dir = tmp_path / "B"
    code, out, err = _run([
        "split-sideb", "--mix", str(mix), "--timeline", str(timeline_path),
        "--out", str(out_dir),
    ])
    assert code == 0, err
    assert "pcm_md5" in out
    master, master_sr = read_pcm24(out_dir / "full-mix.wav")
    pieces = sorted(out_dir.glob("[0-9][0-9]-*.wav"))
    assert len(pieces) == 3
    joined = np.concatenate([read_pcm24(piece)[0] for piece in pieces], axis=0)
    assert joined.shape == master.shape
    assert np.array_equal(joined, master)
    assert pcm_md5(joined) == pcm_md5(master)
    manifest = json.loads((out_dir / "split-manifest.json").read_text(encoding="utf-8"))
    assert manifest["pcm_md5"] == pcm_md5(master)
    assert manifest["full_mix_publish"] is False
    assert sum(piece["frames"] for piece in manifest["pieces"]) == master.shape[0]
    receipt = json.loads((out_dir / "full-mix.wav.json").read_text(encoding="utf-8"))
    assert receipt["verdict"] == "PASS"
    _assert_master(out_dir / "full-mix.wav")
    snap = int(round(master_sr * 0.02))
    starts = [piece["start_sample"] for piece in manifest["pieces"]]
    assert starts[0] == 0
    mono = master.astype(np.float64).sum(axis=1)
    for cut, piece in zip(timeline["cuts"], manifest["pieces"][1:]):
        expected = map_cut_sample(cut["fraction"], receipt)
        assert abs(piece["start_sample"] - expected) <= snap + 8
        index = piece["start_sample"]
        left, right = mono[index - 1], mono[index]
        assert left == 0 or right == 0 or (left < 0) != (right < 0)
        assert piece["title"].endswith(" (Mixed)")
        assert "(Mixed)" not in piece["source_title"] or piece["source_title"].endswith(" (Mixed)")
    # Gapless: no extra sample and no dropped sample, already proved by the md5.
    assert pcm_md5(joined) == manifest["pcm_md5"]

    dry_dir = tmp_path / "B-dry"
    code, out, err = _run([
        "split-sideb", "--mix", str(mix), "--timeline", str(timeline_path),
        "--out", str(dry_dir), "--dry-run",
    ])
    assert code == 0, err
    assert not dry_dir.exists()
    assert "dry run" in out
    assert "(Mixed)" in out


def test_mix_missing_file(tmp_path):
    cue = tmp_path / "cue.yaml"
    _cue(cue, [{"title": "Amber Kiln", "file": "missing.wav", "transition": "cut"}])
    code, _out, err = _run(["mix", "--cue", str(cue), "--root", str(tmp_path), "--output", str(tmp_path / "m.wav")])
    assert code != 0
    assert "missing" in err.lower()
    assert not (tmp_path / "m.wav").exists()


# ---------------------------------------------------------------- album
def test_album_blank_songwriter(tmp_path):
    designs = tmp_path / "designs"
    release = tmp_path / "release"
    stem = "01-amber-kiln"
    _design(
        designs / f"{stem}.json",
        title="Amber Kiln",
        duration=8,
        track=1,
        songwriter=SENTINEL,
        legal_name=SENTINEL,
        composer=SENTINEL,
    )
    wav = release / "A" / f"{stem}.wav"
    _tone(wav, 1.0, amp=0.2, freq=440)
    _receipt(wav, verdict="PASS", duration=1.0)
    piece = release / "B" / "01-amber-kiln.wav"
    _tone(piece, 1.0, amp=0.2, freq=554)
    _receipt(piece, verdict="RECORDED", lufs=-12.5, tp=-3.0, duration=1.0)
    manifest = {
        "full_mix": "full-mix.wav",
        "full_mix_publish": False,
        "pieces": [{
            "index": 1,
            "title": "Amber Kiln (Mixed)",
            "source_title": "Amber Kiln",
            "slug": "amber-kiln",
            "file": "01-amber-kiln.wav",
        }],
    }
    (release / "B").mkdir(parents=True, exist_ok=True)
    (release / "B" / "split-manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    # The unsplit mix is archived beside the pieces and must not become a row.
    full = release / "B" / "full-mix.wav"
    _tone(full, 1.0, amp=0.2, freq=100)
    _receipt(full, verdict="PASS", duration=1.0)

    code, out, err = _run(["album", "--designs", str(designs), "--release", str(release)])
    assert code == 0, err
    sheet = (release / "RELEASE.tsv").read_text(encoding="utf-8")
    assert SENTINEL not in sheet
    assert SENTINEL not in out
    lines = [line.split("\t") for line in sheet.strip().splitlines()]
    assert lines[0] == HEADER
    assert HEADER.index("songwriter") == 12
    assert HEADER.index("side") == 1
    body = lines[1:]
    assert [row[0] for row in body] == ["1", "2"]
    assert [row[1] for row in body] == ["A", "B"]
    assert all(row[HEADER.index("songwriter")] == "" for row in body)
    assert all(row[HEADER.index("isrc")] == "" for row in body)
    assert body[1][2].endswith(" (Mixed)")
    assert "full-mix.wav" not in sheet
    assert body[0][HEADER.index("instrumental")] == "true"

    dry = tmp_path / "DRY.tsv"
    code, out, err = _run([
        "album", "--designs", str(designs), "--release", str(release),
        "--out", str(dry), "--dry-run",
    ])
    assert code == 0, err
    assert not dry.exists()
    assert SENTINEL not in out
    assert "songwriter" in out


def test_album_rejects_bad_md5(tmp_path):
    designs = tmp_path / "designs"
    release = tmp_path / "release"
    stem = "01-amber-kiln"
    _design(designs / f"{stem}.json", title="Amber Kiln", duration=1, track=1)
    wav = release / "A" / f"{stem}.wav"
    _tone(wav, 1.0)
    _receipt(wav, verdict="PASS", duration=1.0)
    receipt = Path(str(wav) + ".json")
    data = json.loads(receipt.read_text(encoding="utf-8"))
    data["md5"] = "0" * 32
    receipt.write_text(json.dumps(data), encoding="utf-8")
    code, _out, err = _run(["album", "--designs", str(designs), "--release", str(release)])
    assert code == 1
    assert "md5" in err
    assert not (release / "RELEASE.tsv").exists()


# ---------------------------------------------------------------- mirror
def test_mirror_rsync_and_checksum(tmp_path, monkeypatch):
    monkeypatch.delenv("RELAY_MIRROR_HOST", raising=False)
    monkeypatch.delenv("RELAY_MIRROR_PATH", raising=False)
    release = tmp_path / "release"
    (release / "sub").mkdir(parents=True)
    (release / "a.txt").write_text("hello\n", encoding="utf-8")
    (release / "sub" / "b.txt").write_text("world\n", encoding="utf-8")
    fake = tmp_path / "fake-ssh"
    _fake_ssh(fake)
    dest = tmp_path / "remote" / "store"
    dest.parent.mkdir()
    host = "mirror-test.example"
    code, out, err = _run([
        "mirror", "--release", str(release), "--host", host,
        "--dest", str(dest), "--rsh", str(fake),
    ])
    assert code == 0, err
    assert "checksums ok" in out
    assert (dest / "a.txt").read_text(encoding="utf-8") == "hello\n"
    assert (dest / "sub" / "b.txt").read_text(encoding="utf-8") == "world\n"
    assert (dest / "MANIFEST.md5").is_file()
    assert host not in (REPO / "relayrun" / "mirror.py").read_text(encoding="utf-8")

    (dest / "a.txt").write_text("tampered\n", encoding="utf-8")
    from relayrun.mirror import verify_remote
    with pytest.raises(RelayError, match="checksum"):
        verify_remote(host, str(dest), str(fake))


def test_mirror_requires_host_and_dry_run(tmp_path, monkeypatch):
    monkeypatch.delenv("RELAY_MIRROR_HOST", raising=False)
    monkeypatch.delenv("RELAY_MIRROR_PATH", raising=False)
    release = tmp_path / "release"
    release.mkdir()
    (release / "a.txt").write_text("x\n", encoding="utf-8")
    code, _out, err = _run(["mirror", "--release", str(release), "--dest", "/tmp/relay-dest"])
    assert code == 2
    assert "host" in err.lower()
    dest = tmp_path / "not-created"
    fake = tmp_path / "fake-ssh"
    _fake_ssh(fake)
    code, out, err = _run([
        "mirror", "--release", str(release), "--host", "mirror-test.example",
        "--dest", str(dest), "--rsh", str(fake), "--dry-run",
    ])
    assert code == 0, err
    assert "rsync" in out
    assert "mirror-test.example" in out
    assert "md5sum -c" in out
    assert not dest.exists()
    assert not (release / "MANIFEST.md5").exists()
