"""Master a whole mix once, then cut it into one '(Mixed)' file per song.

Cuts sit at the midpoint of each cue-sheet blend (a bar line when the blend
length is an even number of bars) and snap to the nearest zero crossing inside
a short window. Pieces are contiguous slices of the mastered PCM: no fade and
no inserted silence. Concatenating them reproduces the mastered mix sample for
sample. Per-piece loudness is measured and recorded; it is allowed to vary,
because the pieces are not mastered on their own.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import soundfile as sf

from relayrun.audio import file_md5, measure_ebur128, pcm_md5, probe_format, read_pcm24, silence_regions, true_peak_dbtp, write_pcm24
from relayrun.errors import RelayError
from relayrun.master import master_file
from relayrun.mix import load_cue, slugify

DEFAULT_SNAP_MS = 20.0


def snap_zero_crossing(pcm: np.ndarray, index: int, window: int) -> int:
    """Return a cut index near ``index`` where the channel-sum changes sign.

    ``index`` is the first sample of the right-hand piece. The search stays
    inside ``[index - window, index + window]`` and never returns 0 or ``n``.
    If the window has no crossing, the quietest sample in the window is used.
    Moving the cut does not change the concatenated PCM: the pieces still tile
    the master.
    """
    n = int(pcm.shape[0])
    if n < 2:
        raise RelayError("master is too short to split")
    index = int(min(max(index, 1), n - 1))
    window = max(0, int(window))
    lo = max(1, index - window)
    hi = min(n - 1, index + window)
    mono = np.asarray(pcm[lo - 1:hi + 1], dtype=np.float64).sum(axis=1)
    best = None
    best_dist = None
    for offset in range(len(mono) - 1):
        left, right = mono[offset], mono[offset + 1]
        cut = (lo - 1) + offset + 1
        if left == 0 or right == 0 or (left < 0) != (right < 0):
            dist = abs(cut - index)
            if best is None or dist < best_dist or (dist == best_dist and cut < best):
                best, best_dist = cut, dist
    if best is not None:
        return int(best)
    region = np.max(np.abs(np.asarray(pcm[lo:hi + 1], dtype=np.float64)), axis=1)
    return int(lo + int(np.argmin(region)))


def map_cut_sample(fraction: float, report: dict) -> int:
    """Map a premaster fraction through trim and onto the mastered PCM."""
    n_in = int(report["input_frames"])
    start = int(report["trim_start_sample"])
    end = int(report["trim_end_sample"])
    n_out = int(report["output_frames"])
    trimmed_len = max(1, end - start)
    src = float(fraction) * n_in
    trimmed = src - start
    trimmed = min(max(trimmed, 1.0), trimmed_len - 1.0)
    return int(round(trimmed / trimmed_len * n_out))


def _unique_slug(title: str, used: set[str]) -> str:
    base = slugify(title)
    slug = base
    n = 2
    while slug in used:
        slug = f"{base}-{n}"
        n += 1
    used.add(slug)
    return slug


def cuts_from_timeline(timeline: dict, report: dict, pcm: np.ndarray, snap: int) -> list[int]:
    raw = []
    for cut in timeline.get("cuts") or []:
        fraction = cut.get("fraction")
        if fraction is None:
            frames = timeline.get("frames") or 0
            if not frames:
                raise RelayError("timeline cut is missing fraction and frames")
            fraction = float(cut["mid_sample"]) / float(frames)
        raw.append(map_cut_sample(float(fraction), report))
    snapped: list[int] = []
    previous = 0
    n = pcm.shape[0]
    for sample in raw:
        cut = snap_zero_crossing(pcm, sample, snap)
        if cut <= previous:
            cut = min(n - 1, previous + 1)
        if cut >= n:
            cut = n - 1
        snapped.append(cut)
        previous = cut
    return snapped


def _piece_receipt(path: Path, title: str) -> dict:
    from mixgen.util import atomic_write_json

    ebur_i, ebur_tp = measure_ebur128(path)
    info = probe_format(path)
    pcm, sr = sf.read(str(path), dtype="float64", always_2d=True)
    oversampled = true_peak_dbtp(pcm, sr, oversample=4)
    regions = silence_regions(pcm, sr)
    report = {
        "file": path.name,
        "title": title,
        "duration_s": round(info["duration_s"], 4),
        "lufs_i": round(ebur_i, 2),
        "true_peak_dbtp": round(max(ebur_tp, oversampled), 2),
        "lead_silence_s": regions["leading_s"],
        "tail_silence_s": regions["trailing_s"],
        "md5": file_md5(path),
        "sample_rate": info["sample_rate"],
        "bit_depth": 24,
        "channels": info["channels"],
        "verdict": "RECORDED",
        "notes": ["side B piece; loudness is measured, not re-mastered"],
    }
    atomic_write_json(Path(str(path) + ".json"), report)
    return report


def split_mastered(master_wav: Path, timeline: dict, out_dir: Path, report: dict,
                   *, snap_ms: float = DEFAULT_SNAP_MS) -> dict:
    """Slice ``master_wav`` on zero crossings. Pieces tile it exactly."""
    from mixgen.util import atomic_write_json

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    pcm, sr = read_pcm24(master_wav)
    snap = int(round(sr * snap_ms / 1000.0))
    # Snap against the float view so a 24-bit zero is a real zero crossing.
    float_pcm = pcm.astype(np.float64) / float(np.iinfo(np.int32).max)
    cuts = cuts_from_timeline(timeline, report, float_pcm, snap)
    bounds = [0, *cuts, pcm.shape[0]]
    tracks = timeline.get("tracks") or []
    if len(tracks) != len(bounds) - 1:
        raise RelayError(
            f"cue has {len(tracks)} tracks but {len(bounds) - 1} pieces; the timeline does not match the mix"
        )
    used: set[str] = set()
    pieces = []
    rebuilt = []
    for index, track in enumerate(tracks):
        start, end = bounds[index], bounds[index + 1]
        if end <= start:
            raise RelayError(f"empty piece for {track.get('title')}")
        title = track["title"]
        mixed = title if title.endswith(" (Mixed)") else f"{title} (Mixed)"
        slug = _unique_slug(track.get("slug") or title, used)
        name = f"{index + 1:02d}-{slug}.wav"
        chunk = np.ascontiguousarray(pcm[start:end])
        dest = out_dir / name
        write_pcm24(dest, chunk, sr)
        back, back_sr = read_pcm24(dest)
        if back_sr != sr or not np.array_equal(back, chunk):
            raise RelayError(f"{name}: rewritten PCM does not match the master slice")
        rebuilt.append(back)
        receipt = _piece_receipt(dest, mixed)
        pieces.append({
            "index": index + 1,
            "title": mixed,
            "source_title": title,
            "slug": slug,
            "file": name,
            "start_sample": int(start),
            "end_sample": int(end),
            "frames": int(end - start),
            "md5": receipt["md5"],
            "lufs_i": receipt["lufs_i"],
            "true_peak_dbtp": receipt["true_peak_dbtp"],
            "duration_s": receipt["duration_s"],
        })
    joined = np.concatenate(rebuilt, axis=0) if rebuilt else np.zeros((0, pcm.shape[1]), dtype=pcm.dtype)
    if not np.array_equal(joined, pcm):
        raise RelayError("concatenated side B pieces do not match the mastered mix PCM")
    digest = pcm_md5(pcm)
    if pcm_md5(joined) != digest:
        raise RelayError("PCM md5 of the concatenated pieces does not match the mastered mix")
    manifest = {
        "full_mix": master_wav.name,
        "full_mix_publish": False,
        "sample_rate": sr,
        "frames": int(pcm.shape[0]),
        "pcm_md5": digest,
        "snap_ms": snap_ms,
        "cuts": cuts,
        "pieces": pieces,
        "master": {
            "verdict": report.get("verdict"),
            "lufs_i": report.get("lufs_i"),
            "true_peak_dbtp": report.get("true_peak_dbtp"),
            "md5": report.get("md5"),
        },
    }
    atomic_write_json(out_dir / "split-manifest.json", manifest)
    return manifest


def load_timeline(path: Path | None, cue_path: Path | None, root: Path | None) -> dict:
    if path is not None:
        path = Path(path)
        if not path.is_file():
            raise RelayError(f"timeline not found: {path}", exit_code=2)
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except json.JSONDecodeError as exc:
            raise RelayError(f"timeline is not JSON: {exc}", exit_code=2) from exc
        if not isinstance(data, dict) or "tracks" not in data:
            raise RelayError("timeline needs a tracks list", exit_code=2)
        return data
    if cue_path is None:
        raise RelayError("split-sideb needs --timeline or --cue", exit_code=2)
    # Rebuild the timeline from the cue. This matches mix when the sources are unchanged.
    from relayrun.mix import assemble

    cue = load_cue(Path(cue_path), root)
    _pcm, timeline = assemble(cue)
    return timeline
