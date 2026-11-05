"""Plan YAML loading, validation and root-confined path resolution.

One plan describes a whole mix: the mix title/slug/output dir, then one entry per
track (style prompt, bpm, Camelot key, duration, instrumental/lyrics, takes, job
placement, optional djmix cue fields).

Every path the pipeline touches is derived here and validated to resolve under
``--root`` (default ``/music``); escapes (``..``, absolute paths outside the
root, symlinked parents) are rejected before anything runs.
"""
from __future__ import annotations

import os
import re
from dataclasses import dataclass, field
from pathlib import Path

import yaml

TRANSITIONS = ("long_blend", "eq_bass_swap", "filter_sweep", "cut")
JOB_KINDS = ("cpu", "gpu")
JOB_HOSTS = ("nexus", "sentry")
DEFAULT_ROOT = "/music"
DEFAULT_TAKES = 3
DEFAULT_PHRASE_BARS = 16
DEFAULT_BEATS_PER_BAR = 4

_SLUG_RE = re.compile(r"^[a-z0-9][a-z0-9-]*$")
_CAMELOT_RE = re.compile(r"^(?:[1-9]|1[0-2])[AB]$")
_NUM_RE = re.compile(r"\d+(?:\.\d+)?")

_TOP_KEYS = {
    "title", "slug", "output", "sample_rate", "phrase_bars", "beats_per_bar",
    "takes", "backend", "gen", "acestep", "target_lufs", "target_tp", "tracks",
    "root",
}
_TRACK_KEYS = {
    "title", "slug", "style", "bpm", "key", "duration", "instrumental", "lyrics",
    "takes", "job", "play_bars", "mix_in_bars", "transition", "ramp", "seed",
}
_JOB_KEYS = {"kind", "host"}


class PlanError(ValueError):
    """Invalid plan content (bad field, bad job placement, path escape, ...)."""


def slugify(text: str) -> str:
    s = re.sub(r"[^a-z0-9]+", "-", str(text).strip().lower()).strip("-")
    s = re.sub(r"-{2,}", "-", s)
    if not s or not _SLUG_RE.match(s):
        raise PlanError(f"cannot derive a safe slug from {text!r}; set an explicit 'slug'")
    return s


def parse_duration(value) -> float:
    """``mm:ss``, ``hh:mm:ss`` or a plain number of seconds -> seconds."""
    if isinstance(value, bool):
        raise PlanError(f"duration must be mm:ss or seconds, got {value!r}")
    if isinstance(value, (int, float)):
        seconds = float(value)
    elif isinstance(value, str):
        s = value.strip()
        if _NUM_RE.fullmatch(s):
            seconds = float(s)
        else:
            parts = s.split(":")
            if not 2 <= len(parts) <= 3 or not all(_NUM_RE.fullmatch(p) for p in parts):
                raise PlanError(f"duration must be mm:ss, hh:mm:ss or seconds, got {value!r}")
            seconds = 0.0
            for p in parts:
                seconds = seconds * 60 + float(p)
    else:
        raise PlanError(f"duration must be mm:ss or seconds, got {value!r}")
    if seconds <= 0:
        raise PlanError(f"duration must be positive, got {value!r}")
    return seconds


def resolve_under(root: Path | str, value: Path | str, what: str = "path") -> Path:
    """Resolve ``value`` relative to ``root`` and refuse anything outside it."""
    root_r = Path(os.path.expanduser(str(root))).resolve()
    raw = Path(os.path.expanduser(str(value)))
    full = raw if raw.is_absolute() else root_r / raw
    try:
        resolved = full.resolve(strict=False)
    except OSError as exc:  # pragma: no cover - platform dependent
        raise PlanError(f"{what} {value!r} cannot be resolved: {exc}") from exc
    if resolved != root_r and root_r not in resolved.parents:
        raise PlanError(f"{what} {value!r} escapes root {root_r} (resolved to {resolved})")
    return resolved


# ------------------------------------------------------------------ Camelot
CAMELOT_MINOR = {"G#": "1A", "D#": "2A", "A#": "3A", "F": "4A", "C": "5A", "G": "6A",
                 "D": "7A", "A": "8A", "E": "9A", "B": "10A", "F#": "11A", "C#": "12A"}
CAMELOT_MAJOR = {"B": "1B", "F#": "2B", "C#": "3B", "G#": "4B", "D#": "5B", "A#": "6B",
                 "F": "7B", "C": "8B", "G": "9B", "D": "10B", "A": "11B", "E": "12B"}


def camelot_to_tonic(camelot: str) -> tuple[str, str]:
    """``8A`` -> ``("A", "min")``; ``12B`` -> ``("E", "maj")``."""
    if not _CAMELOT_RE.match(str(camelot)):
        raise PlanError(f"invalid Camelot key {camelot!r} (expected e.g. 8A / 12B)")
    for table, mode in ((CAMELOT_MINOR, "min"), (CAMELOT_MAJOR, "maj")):
        for tonic, cam in table.items():
            if cam == camelot:
                return tonic, mode
    raise PlanError(f"invalid Camelot key {camelot!r}")  # pragma: no cover


def camelot_of(tonic: str, mode: str) -> str:
    tonic = tonic.rstrip("m").upper()
    if tonic not in CAMELOT_MINOR:
        raise PlanError(f"unknown tonic {tonic!r}")
    return (CAMELOT_MINOR if mode.startswith("min") else CAMELOT_MAJOR)[tonic]


# ------------------------------------------------------------------ model
@dataclass
class Job:
    kind: str
    host: str

    def __str__(self) -> str:
        return f"{self.kind}@{self.host}"


@dataclass
class Track:
    index: int
    title: str
    slug: str
    style: str
    bpm: float
    key: str                       # Camelot, e.g. 8A
    duration: float                # seconds
    instrumental: bool
    takes: int
    job: Job
    lyrics: str | None = None
    play_bars: int | None = None
    mix_in_bars: int = 32
    transition: str = "long_blend"
    ramp: bool = True
    seed: int | None = None

    @property
    def duration_hms(self) -> str:
        from mixgen.util import fmt_hms

        return fmt_hms(self.duration)


@dataclass
class Plan:
    title: str
    slug: str
    root: Path
    output: Path                   # resolved, under root
    tracks: list[Track]
    source: Path | None = None
    sample_rate: int = 48000
    phrase_bars: int = DEFAULT_PHRASE_BARS
    beats_per_bar: int = DEFAULT_BEATS_PER_BAR
    target_lufs: float = -9.0
    target_tp: float = -1.0
    backend: str | None = None
    acestep: dict = field(default_factory=dict)

    # ---- paths -------------------------------------------------------
    @property
    def gen_dir(self) -> Path:
        return resolve_under(self.root, Path("gen") / self.slug, "gen dir")

    def track_gen_dir(self, track: Track) -> Path:
        return resolve_under(self.root, Path("gen") / self.slug / track.slug, "track gen dir")

    def take_path(self, track: Track, take_idx: int) -> Path:
        return self.track_gen_dir(track) / f"take-{take_idx}.wav"

    @property
    def qa_dir(self) -> Path:
        return self.output / "qa"

    def qa_track_dir(self, track: Track) -> Path:
        return self.qa_dir / track.slug

    def qa_report_path(self, track: Track) -> Path:
        return self.qa_track_dir(track) / "report.json"

    def qa_summary_path(self) -> Path:
        return self.qa_dir / "summary.json"

    @property
    def master_dir(self) -> Path:
        return self.output / "mastered"

    def master_path(self, track: Track) -> Path:
        return self.master_dir / f"{track.slug}.wav"

    def mix_wav(self) -> Path:
        return self.output / f"{self.slug}.wav"

    def mix_mp3(self) -> Path:
        return self.output / f"{self.slug}.mp3"

    def cue_path(self) -> Path:
        return self.output / f"{self.slug}.cue.yaml"

    def tracklist_path(self) -> Path:
        return self.output / f"{self.slug}.tracklist.txt"

    def chapters_path(self) -> Path:
        return self.output / f"{self.slug}.chapters.txt"

    def rel_to_root(self, path: Path) -> str:
        """Path relative to root for cue sheets (djmix resolves them itself)."""
        try:
            return str(path.relative_to(self.root))
        except ValueError:  # pragma: no cover - guarded by resolve_under
            raise PlanError(f"path {path} is not under root {self.root}")


# ------------------------------------------------------------------ load
def _check_job(index: int, raw) -> Job:
    if not isinstance(raw, dict):
        raise PlanError(f"track {index}: 'job' must be a mapping with kind/host")
    unknown = set(raw) - _JOB_KEYS
    if unknown:
        raise PlanError(f"track {index}: unknown job keys {sorted(unknown)}")
    kind = str(raw.get("kind", "")).lower()
    host = str(raw.get("host", "")).lower()
    if kind not in JOB_KINDS:
        raise PlanError(f"track {index}: job.kind must be one of {JOB_KINDS}, got {raw.get('kind')!r}")
    if host not in JOB_HOSTS:
        raise PlanError(f"track {index}: job.host must be one of {JOB_HOSTS}, got {raw.get('host')!r}")
    return Job(kind=kind, host=host)


def _parse_track(index: int, raw, root: Path, defaults: dict, phrase: int) -> Track:
    if not isinstance(raw, dict):
        raise PlanError(f"track {index}: must be a mapping")
    unknown = set(raw) - _TRACK_KEYS
    if unknown:
        raise PlanError(f"track {index}: unknown keys {sorted(unknown)}")
    for required in ("title", "style", "bpm", "key", "duration", "job"):
        if raw.get(required) in (None, ""):
            raise PlanError(f"track {index}: missing required field '{required}'")
    title = str(raw["title"]).strip()
    slug_raw = raw.get("slug")
    slug = slugify(slug_raw if slug_raw else title)
    style = str(raw["style"]).strip()
    if not style:
        raise PlanError(f"track {index}: 'style' prompt must not be empty")
    try:
        bpm = float(raw["bpm"])
    except (TypeError, ValueError) as exc:
        raise PlanError(f"track {index}: bpm must be a number, got {raw['bpm']!r}") from exc
    if not 60 <= bpm <= 200:
        raise PlanError(f"track {index}: bpm {bpm} out of range 60..200")
    key = str(raw["key"]).upper()
    if not _CAMELOT_RE.match(key):
        raise PlanError(f"track {index}: key must be a Camelot key like 8A, got {raw['key']!r}")
    duration = parse_duration(raw["duration"])
    lyrics = raw.get("lyrics")
    if lyrics is not None and not isinstance(lyrics, str):
        raise PlanError(f"track {index}: lyrics must be a string")
    if raw.get("instrumental") is None:
        instrumental = lyrics is None
    else:
        instrumental = bool(raw["instrumental"])
    if lyrics and instrumental:
        raise PlanError(f"track {index}: has lyrics but is marked instrumental")
    takes = int(raw.get("takes", defaults["takes"]))
    if takes < 1:
        raise PlanError(f"track {index}: takes must be >= 1")
    play_bars = raw.get("play_bars")
    if play_bars is not None:
        play_bars = int(play_bars)
        if play_bars <= 0 or play_bars % phrase:
            raise PlanError(f"track {index}: play_bars must be a positive multiple of phrase_bars ({phrase})")
    transition = str(raw.get("transition", "long_blend"))
    if transition not in TRANSITIONS:
        raise PlanError(f"track {index}: transition must be one of {TRANSITIONS}, got {transition!r}")
    mix_in = int(raw.get("mix_in_bars", 32))
    if index == 0 or transition == "cut":
        mix_in = 0
    elif mix_in <= 0 or mix_in % phrase:
        raise PlanError(f"track {index}: mix_in_bars must be a positive multiple of phrase_bars ({phrase})")
    ramp = bool(raw.get("ramp", True))
    seed = raw.get("seed")
    if seed is not None:
        seed = int(seed)
    return Track(index=index, title=title, slug=slug, style=style, bpm=bpm, key=key,
                 duration=duration, instrumental=instrumental, takes=takes,
                 job=_check_job(index, raw["job"]), lyrics=lyrics, play_bars=play_bars,
                 mix_in_bars=mix_in, transition=transition, ramp=ramp, seed=seed)


def load_plan(path: str | Path, root: str | Path | None = None) -> Plan:
    path = Path(path)
    try:
        data = yaml.safe_load(path.read_text())
    except OSError as exc:
        raise PlanError(f"cannot read plan {path}: {exc}") from exc
    if not isinstance(data, dict):
        raise PlanError("plan YAML must be a mapping")
    unknown = set(data) - _TOP_KEYS
    if unknown:
        raise PlanError(f"unknown top-level keys {sorted(unknown)}")
    title = str(data.get("title") or "").strip()
    if not title:
        raise PlanError("plan needs a 'title'")
    slug = slugify(data["slug"] if data.get("slug") else title)
    root_p = Path(os.path.expanduser(str(root or data.get("root") or DEFAULT_ROOT))).resolve()
    output = resolve_under(root_p, data.get("output") or Path("mixes") / slug, "output dir")
    if output == root_p:
        raise PlanError("output dir must not be the root itself")
    phrase = int(data.get("phrase_bars", DEFAULT_PHRASE_BARS))
    if phrase not in (16, 32):
        raise PlanError("phrase_bars must be 16 or 32")
    takes = int(data.get("takes", DEFAULT_TAKES))
    if takes < 1:
        raise PlanError("takes must be >= 1")
    raw_tracks = data.get("tracks")
    if not isinstance(raw_tracks, list) or not raw_tracks:
        raise PlanError("plan needs a non-empty 'tracks' list")
    tracks = [_parse_track(i, t, root_p, {"takes": takes}, phrase)
              for i, t in enumerate(raw_tracks)]
    slugs = [t.slug for t in tracks]
    dupes = sorted({s for s in slugs if slugs.count(s) > 1})
    if dupes:
        raise PlanError(f"duplicate track slugs {dupes}")
    acestep = data.get("acestep") or {}
    if not isinstance(acestep, dict):
        raise PlanError("'acestep' must be a mapping")
    backend = data.get("backend") or (data.get("gen") or {}).get("backend")
    if backend is not None:
        backend = str(backend)
        if backend not in ("acestep", "fake"):
            raise PlanError(f"backend must be 'acestep' or 'fake', got {backend!r}")
    return Plan(
        title=title, slug=slug, root=root_p, output=output, tracks=tracks, source=path,
        sample_rate=int(data.get("sample_rate", 48000)),
        phrase_bars=phrase, beats_per_bar=int(data.get("beats_per_bar", DEFAULT_BEATS_PER_BAR)),
        target_lufs=float(data.get("target_lufs", -9.0)),
        target_tp=float(data.get("target_tp", -1.0)),
        backend=backend, acestep=acestep,
    )


def effective_play_bars(track: Track, phrase_bars: int) -> int:
    """Cue ``play_bars`` or the longest phrase-aligned fit inside the duration."""
    if track.play_bars:
        return track.play_bars
    bars = int(track.duration * track.bpm / (60.0 * DEFAULT_BEATS_PER_BAR))
    return (bars // phrase_bars) * phrase_bars
