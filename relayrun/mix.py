"""Continuous DJ mix from a cue sheet.

The cue sheet is YAML. ``relay-run mix`` places each source on a shared
timeline and overlaps neighbours with an equal-power crossfade, or joins them
with a hard cut. It does not time-stretch, detect beats, or render new audio.
Bar alignment comes from the cue: a blend whose length is an even number of
bars has its midpoint on a bar line.

Cue sheet::

    version: 1
    sample_rate: 44100          # default 48000
    output: mix.wav             # optional; relative to the cue file
    tracks:
      - title: Amber Kiln       # required
        file: amber.wav         # relative to --root, else the cue directory
        slug: amber-kiln        # optional
        bpm: 120                # required when blend length is in bars
        start_s: 0              # source offset, default 0
        duration_s: 8           # source length, default: to the end of the file
        transition: blend       # blend (equal-power) or cut. First track is a cut.
        blend_s: 2.0            # overlap in seconds
        mix_in_bars: 16         # used only when blend_s is omitted
        beats_per_bar: 4        # default 4
        gain_db: 0              # optional trim

``split-sideb`` cuts the mastered mix at the midpoint of each blend, then snaps
that sample to a zero crossing. Write ``mix_in_bars`` as a multiple of the
phrase (16 or 32) when the cut should land on a bar line.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import yaml

from relayrun.audio import load_audio
from relayrun.errors import RelayError

_FORBIDDEN = set('\\/:*?"\'<>|')


def slugify(title: str) -> str:
    slug = re.sub(r"[^a-z0-9]+", "-", title.lower()).strip("-")
    return slug or "track"


def check_title(title: str, where: str) -> str:
    title = str(title).strip()
    if not title:
        raise RelayError(f"{where}: title is empty", exit_code=2)
    if any(ch in title for ch in _FORBIDDEN) or "\n" in title or "\t" in title:
        raise RelayError(f"{where}: title {title!r} contains a character that cannot go in a filename", exit_code=2)
    return title


@dataclass
class CueTrack:
    title: str
    file: Path
    slug: str
    bpm: float | None
    start_s: float
    duration_s: float | None
    transition: str
    blend_s: float | None
    mix_in_bars: int | None
    beats_per_bar: int
    gain_db: float
    index: int


@dataclass
class Cue:
    path: Path
    root: Path
    sample_rate: int
    output: Path | None
    tracks: list[CueTrack] = field(default_factory=list)


def _resolve(root: Path, value: str) -> Path:
    path = Path(value)
    if path.is_absolute():
        return path
    resolved = (root / path).resolve()
    root_r = root.resolve()
    if resolved != root_r and root_r not in resolved.parents:
        raise RelayError(f"cue path escapes the root: {value}", exit_code=2)
    return resolved


def load_cue(path: Path, root: Path | None = None) -> Cue:
    path = Path(path)
    if not path.is_file():
        raise RelayError(f"cue sheet not found: {path}", exit_code=2)
    try:
        data = yaml.safe_load(path.read_text(encoding="utf-8"))
    except yaml.YAMLError as exc:
        raise RelayError(f"cue sheet is not valid YAML: {exc}", exit_code=2) from exc
    if not isinstance(data, dict) or not isinstance(data.get("tracks"), list) or not data["tracks"]:
        raise RelayError("cue sheet needs a top-level tracks list", exit_code=2)
    cue_root = Path(root) if root else path.parent
    sample_rate = int(data.get("sample_rate") or 48000)
    if sample_rate < 8000:
        raise RelayError(f"sample_rate {sample_rate} is too low", exit_code=2)
    output = None
    if data.get("output"):
        output = _resolve(path.parent, str(data["output"]))
    known = {
        "title", "file", "slug", "bpm", "start_s", "duration_s", "transition",
        "blend_s", "mix_in_bars", "beats_per_bar", "gain_db",
    }
    tracks: list[CueTrack] = []
    for index, raw in enumerate(data["tracks"]):
        where = f"track {index + 1}"
        if not isinstance(raw, dict):
            raise RelayError(f"{where}: expected a mapping", exit_code=2)
        unknown = set(raw) - known
        if unknown:
            raise RelayError(f"{where}: unknown keys {sorted(unknown)}", exit_code=2)
        if "file" not in raw or "title" not in raw:
            raise RelayError(f"{where}: needs title and file", exit_code=2)
        title = check_title(raw["title"], where)
        transition = str(raw.get("transition") or ("cut" if index == 0 else "blend")).lower()
        if transition == "long_blend":
            transition = "blend"
        if transition == "hard_cut":
            transition = "cut"
        if index == 0:
            transition = "cut"
        if transition not in ("blend", "cut"):
            raise RelayError(f"{where}: transition must be blend or cut", exit_code=2)
        blend_s = float(raw["blend_s"]) if raw.get("blend_s") is not None else None
        mix_in_bars = int(raw["mix_in_bars"]) if raw.get("mix_in_bars") is not None else None
        if blend_s is not None and blend_s < 0:
            raise RelayError(f"{where}: blend_s must be >= 0", exit_code=2)
        if mix_in_bars is not None and mix_in_bars < 0:
            raise RelayError(f"{where}: mix_in_bars must be >= 0", exit_code=2)
        tracks.append(CueTrack(
            title=title,
            file=_resolve(cue_root, str(raw["file"])),
            slug=slugify(str(raw["slug"])) if raw.get("slug") else slugify(title),
            bpm=float(raw["bpm"]) if raw.get("bpm") is not None else None,
            start_s=float(raw.get("start_s") or 0.0),
            duration_s=float(raw["duration_s"]) if raw.get("duration_s") is not None else None,
            transition=transition,
            blend_s=blend_s,
            mix_in_bars=mix_in_bars,
            beats_per_bar=int(raw.get("beats_per_bar") or 4),
            gain_db=float(raw.get("gain_db") or 0.0),
            index=index,
        ))
    return Cue(path=path, root=cue_root, sample_rate=sample_rate, output=output, tracks=tracks)


def blend_seconds(track: CueTrack) -> float:
    if track.index == 0 or track.transition == "cut":
        return 0.0
    if track.blend_s is not None:
        return float(track.blend_s)
    if track.mix_in_bars:
        if not track.bpm:
            raise RelayError(f"{track.title}: mix_in_bars needs bpm", exit_code=2)
        return track.mix_in_bars * track.beats_per_bar * 60.0 / track.bpm
    raise RelayError(f"{track.title}: blend transition needs blend_s or mix_in_bars", exit_code=2)


def _resample(pcm: np.ndarray, src_sr: int, dst_sr: int) -> np.ndarray:
    if src_sr == dst_sr:
        return pcm
    from scipy.signal import resample_poly

    # Rational approximation is exact for 48000 <-> 44100 (147:160).
    from math import gcd
    factor = gcd(src_sr, dst_sr)
    up, down = dst_sr // factor, src_sr // factor
    channels = [resample_poly(pcm[:, ch], up, down) for ch in range(pcm.shape[1])]
    return np.stack(channels, axis=1)


def load_segment(track: CueTrack, sample_rate: int) -> np.ndarray:
    if not track.file.is_file():
        raise RelayError(f"missing cue audio: {track.file}", exit_code=2)
    pcm, sr = load_audio(track.file)
    if pcm.shape[1] == 1:
        pcm = np.repeat(pcm, 2, axis=1)
    else:
        pcm = pcm[:, :2]
    start = max(0, int(round(track.start_s * sr)))
    if track.duration_s is None:
        end = pcm.shape[0]
    else:
        end = min(pcm.shape[0], start + max(0, int(round(track.duration_s * sr))))
    if end - start < 2:
        raise RelayError(f"{track.title}: source window is empty", exit_code=2)
    segment = pcm[start:end]
    if track.gain_db:
        segment = segment * (10 ** (track.gain_db / 20.0))
    segment = _resample(segment, sr, sample_rate)
    return np.ascontiguousarray(segment, dtype=np.float64)


def assemble(cue: Cue) -> tuple[np.ndarray, dict]:
    """Render the cue to a float64 stereo buffer and a timeline.

    The timeline's ``cuts`` are blend midpoints as a fraction of the premaster,
    which is what ``split-sideb`` snaps onto the mastered PCM.
    """
    segments: list[tuple[CueTrack, np.ndarray, int]] = []
    for track in cue.tracks:
        pcm = load_segment(track, cue.sample_rate)
        blend = int(round(blend_seconds(track) * cue.sample_rate))
        if blend:
            blend = min(blend, pcm.shape[0] - 1)
        segments.append((track, pcm, blend))
    # Length: each new segment extends the cursor by its non-overlapping tail.
    cursor = 0
    placed: list[tuple[CueTrack, np.ndarray, int, int]] = []
    for track, pcm, blend in segments:
        start = max(0, cursor - blend) if placed else 0
        if placed:
            blend = min(blend, cursor - start, pcm.shape[0] - 1, placed[-1][1].shape[0] - 1)
            start = cursor - blend
        placed.append((track, pcm, start, blend))
        cursor = start + pcm.shape[0]
    total = cursor
    mix = np.zeros((total, 2), dtype=np.float64)
    cuts = []
    track_rows = []
    for track, pcm, start, blend in placed:
        piece = pcm.copy()
        if blend > 0:
            # Equal-power overlap. This is the blend itself, not a fade at a later cut.
            fade = np.linspace(0.0, np.pi / 2.0, blend, endpoint=False)
            mix[start:start + blend] *= np.cos(fade)[:, None]
            piece[:blend] *= np.sin(fade)[:, None]
        if track_rows:
            mid = start + (blend // 2)
            cuts.append({
                "after_title": track_rows[-1]["title"],
                "before_title": track.title,
                "mid_sample": int(mid),
                "mid_s": round(mid / cue.sample_rate, 6),
                "fraction": (mid / total) if total else 0.0,
                "blend_samples": int(blend),
            })
        mix[start:start + piece.shape[0]] += piece
        track_rows.append({
            "index": track.index + 1,
            "title": track.title,
            "slug": track.slug,
            "file": track.file.name,
            "start_sample": int(start),
            "end_sample": int(start + piece.shape[0]),
            "blend_in_samples": int(blend),
        })
    # Keep a little headroom so the WAV writer does not wrap, without fading edges.
    peak = float(np.max(np.abs(mix))) if mix.size else 0.0
    if peak > 0.99:
        mix *= 0.99 / peak
    timeline = {
        "sample_rate": cue.sample_rate,
        "frames": int(total),
        "duration_s": round(total / cue.sample_rate, 6) if cue.sample_rate else 0.0,
        "tracks": track_rows,
        "cuts": cuts,
    }
    return mix, timeline


def describe(timeline: dict) -> str:
    lines = [
        f"mix: {timeline['frames']} frames, {timeline['duration_s']:.3f}s, {timeline['sample_rate']} Hz",
    ]
    for row in timeline["tracks"]:
        lines.append(
            f"  {row['index']:02d} {row['title']}  start {row['start_sample']}  "
            f"blend_in {row['blend_in_samples']}"
        )
    for cut in timeline["cuts"]:
        lines.append(
            f"  cut {cut['after_title']} | {cut['before_title']}  "
            f"mid {cut['mid_s']:.3f}s  fraction {cut['fraction']:.6f}"
        )
    return "\n".join(lines)
