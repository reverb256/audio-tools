"""Cue-sheet (YAML) loading and validation."""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

import yaml

TRANSITIONS = ("long_blend", "eq_bass_swap", "filter_sweep", "cut")
DEFAULT_ROOT = "/data/media/archive"


class CueError(ValueError):
    pass


@dataclass
class Track:
    file: Path
    target_bpm: float
    title: str = ""
    bpm: float | None = None          # source BPM override (skip detection)
    first_downbeat: float | None = None  # seconds, override detection
    start_bar: int = 0                # bars to skip from the first downbeat
    play_bars: int | None = None      # bars to play from start_bar (default: to the end)
    mix_in_bars: int = 32             # overlap with the PREVIOUS track
    transition: str = "long_blend"    # how we come IN from the previous track
    ramp: bool = False                # ramp tempo from previous target to ours across the overlap
    gain_db: float = 0.0


@dataclass
class CueSheet:
    tracks: list[Track]
    output: Path
    root: Path
    sample_rate: int = 48000
    phrase_bars: int = 16
    beats_per_bar: int = 4
    target_lufs: float = -9.0
    target_tp: float = -1.0
    loudnorm: bool = True
    tracklist: Path | None = None
    extra: dict = field(default_factory=dict)


def _resolve(root: Path, p: str) -> Path:
    path = Path(os.path.expanduser(p))
    if not path.is_absolute():
        path = root / path
    return path


def _parse_track(i: int, raw: dict, root: Path, phrase: int, default_bpm) -> Track:
    if not isinstance(raw, dict) or "file" not in raw:
        raise CueError(f"track {i}: needs a 'file'")
    unknown = set(raw) - set(Track.__dataclass_fields__)
    if unknown:
        raise CueError(f"track {i}: unknown keys {sorted(unknown)}")
    tgt = raw.get("target_bpm", default_bpm)
    if tgt is None:
        raise CueError(f"track {i}: needs target_bpm (or a top-level target_bpm)")
    t = Track(
        file=_resolve(root, raw["file"]),
        target_bpm=float(tgt),
        title=str(raw.get("title") or Path(raw["file"]).stem),
        bpm=float(raw["bpm"]) if raw.get("bpm") else None,
        first_downbeat=float(raw["first_downbeat"]) if raw.get("first_downbeat") is not None else None,
        start_bar=int(raw.get("start_bar", 0)),
        play_bars=int(raw["play_bars"]) if raw.get("play_bars") else None,
        mix_in_bars=int(raw.get("mix_in_bars", 32 if i else 0)),
        transition=str(raw.get("transition", "long_blend")),
        ramp=bool(raw.get("ramp", False)),
        gain_db=float(raw.get("gain_db", 0.0)),
    )
    if t.transition not in TRANSITIONS:
        raise CueError(f"track {i}: transition must be one of {TRANSITIONS}")
    if not 60 <= t.target_bpm <= 200:
        raise CueError(f"track {i}: target_bpm {t.target_bpm} out of range 60..200")
    if i == 0:
        t.mix_in_bars = 0
    elif t.transition == "cut":
        t.mix_in_bars = 0
    elif t.mix_in_bars <= 0 or t.mix_in_bars % phrase:
        raise CueError(f"track {i}: mix_in_bars must be a positive multiple of phrase_bars ({phrase})")
    if t.play_bars is not None and t.play_bars % phrase:
        raise CueError(f"track {i}: play_bars should be a multiple of phrase_bars ({phrase})")
    return t


def load(path: str | Path, root: str | Path | None = None) -> CueSheet:
    path = Path(path)
    data = yaml.safe_load(path.read_text())
    if not isinstance(data, dict) or not data.get("tracks"):
        raise CueError("cue sheet needs a top-level 'tracks' list")
    root_p = Path(root or data.get("root") or os.environ.get("DJMIX_ROOT") or DEFAULT_ROOT)
    phrase = int(data.get("phrase_bars", 16))
    if phrase not in (16, 32):
        raise CueError("phrase_bars must be 16 or 32")
    default_bpm = data.get("target_bpm")
    tracks = [_parse_track(i, t, root_p, phrase, default_bpm) for i, t in enumerate(data["tracks"])]
    out = _resolve(root_p, data.get("output", "mixes/djmix.wav"))
    tl = data.get("tracklist")
    return CueSheet(
        tracks=tracks,
        output=out,
        root=root_p,
        sample_rate=int(data.get("sample_rate", 48000)),
        phrase_bars=phrase,
        beats_per_bar=int(data.get("beats_per_bar", 4)),
        target_lufs=float(data.get("target_lufs", -9.0)),
        target_tp=float(data.get("target_tp", -1.0)),
        loudnorm=bool(data.get("loudnorm", True)),
        tracklist=_resolve(root_p, tl) if tl else out.with_suffix(".tracklist.txt"),
    )
