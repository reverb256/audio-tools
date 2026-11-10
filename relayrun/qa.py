"""Per-take QA: duration, clipping, silence, and integrated loudness.

Takes live at ``<renders>/<album>/<NN-slug>/take-N.flac`` (``.wav`` is also read).
Thresholds:

- duration within ``max(1 s, 2%)`` of the design ``duration``
- clipping when any sample peaks at or above -0.1 dBFS (near full scale)
- leading, trailing, or internal silence longer than 0.5 s below -60 dBFS RMS
- integrated loudness outside -40..-8 LUFS (a raw-take sanity window, not the
  release target of -14 LUFS)
"""

from __future__ import annotations

import json
import re
from pathlib import Path

import numpy as np

from relayrun.audio import (
    SILENCE_LIMIT_S,
    integrated_lufs,
    is_clipped,
    load_audio,
    peak_dbfs,
    silence_regions,
)
from relayrun.errors import RelayError

DURATION_ABS_S = 1.0
DURATION_RATIO = 0.02
LOUDNESS_MIN = -40.0
LOUDNESS_MAX = -8.0
CLIP_LIMIT_TEXT = "-0.1 dBFS"
_TAKE = re.compile(r"take-(\d+)\.(flac|wav|aiff|aif)$", re.I)
_DESIGN = re.compile(r"(\d\d-.+)\.json$")


def duration_tolerance(design_s: float) -> float:
    return max(DURATION_ABS_S, DURATION_RATIO * float(design_s))


def load_designs(designs: Path) -> list[dict]:
    designs = Path(designs)
    if not designs.is_dir():
        raise RelayError(f"designs directory not found: {designs}", exit_code=2)
    rows = []
    for path in sorted(designs.glob("*.json")):
        match = _DESIGN.match(path.name)
        if not match:
            continue
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except json.JSONDecodeError as exc:
            raise RelayError(f"{path.name}: invalid JSON ({exc})") from exc
        if not isinstance(data, dict):
            raise RelayError(f"{path.name}: expected an object")
        if "duration" not in data:
            raise RelayError(f"{path.name}: missing duration")
        try:
            duration = float(data["duration"])
        except (TypeError, ValueError) as exc:
            raise RelayError(f"{path.name}: duration is not a number") from exc
        rows.append({
            "stem": match.group(1),
            "title": str(data.get("title") or match.group(1)),
            "duration_s": duration,
            "path": path,
        })
    if not rows:
        raise RelayError(f"no NN-slug.json designs in {designs}", exit_code=2)
    return rows


def find_takes(takes_dir: Path, stem: str) -> list[tuple[int, Path]]:
    folder = takes_dir / stem
    if not folder.is_dir():
        return []
    found = []
    for path in sorted(folder.iterdir()):
        match = _TAKE.match(path.name)
        if match and path.is_file():
            found.append((int(match.group(1)), path))
    return sorted(found)


def check_take(path: Path, design_duration: float) -> dict:
    pcm, sr = load_audio(path)
    duration = pcm.shape[0] / float(sr)
    tol = duration_tolerance(design_duration)
    duration_ok = abs(duration - design_duration) <= tol
    regions = silence_regions(pcm, sr)
    silence_ok = (
        regions["leading_s"] <= SILENCE_LIMIT_S
        and regions["trailing_s"] <= SILENCE_LIMIT_S
        and regions["internal_s"] <= SILENCE_LIMIT_S
    )
    clipped = is_clipped(pcm)
    try:
        lufs = integrated_lufs(pcm, sr)
    except RelayError as exc:
        lufs = float("nan")
        loudness_note = str(exc)
    else:
        loudness_note = ""
    loudness_ok = np.isfinite(lufs) and LOUDNESS_MIN <= lufs <= LOUDNESS_MAX
    notes = []
    if not duration_ok:
        notes.append(
            f"duration {duration:.3f}s vs design {design_duration:.3f}s (tolerance {tol:.3f}s)"
        )
    if clipped:
        notes.append(f"clipping: sample peak {peak_dbfs(pcm):.2f} dBFS (limit {CLIP_LIMIT_TEXT})")
    if regions["leading_s"] > SILENCE_LIMIT_S:
        notes.append(f"leading silence {regions['leading_s']:.3f}s")
    if regions["trailing_s"] > SILENCE_LIMIT_S:
        notes.append(f"trailing silence {regions['trailing_s']:.3f}s")
    if regions["internal_s"] > SILENCE_LIMIT_S:
        notes.append(f"internal silence {regions['internal_s']:.3f}s")
    if not loudness_ok:
        shown = "unmeasurable" if not np.isfinite(lufs) else f"{lufs:.2f} LUFS"
        notes.append(f"integrated loudness {shown} outside {LOUDNESS_MIN:.0f}..{LOUDNESS_MAX:.0f}"
                     + (f" ({loudness_note})" if loudness_note else ""))
    verdict = "PASS" if not notes else "FAIL"
    return {
        "file": path.name,
        "path": str(path),
        "duration_s": round(duration, 4),
        "design_duration_s": design_duration,
        "duration_tolerance_s": round(tol, 4),
        "duration_ok": duration_ok,
        "peak_dbfs": round(peak_dbfs(pcm), 3),
        "clipped": clipped,
        "leading_silence_s": regions["leading_s"],
        "trailing_silence_s": regions["trailing_s"],
        "internal_silence_s": regions["internal_s"],
        "silence_ok": silence_ok,
        "lufs_i": None if not np.isfinite(lufs) else round(float(lufs), 2),
        "loudness_ok": bool(loudness_ok),
        "verdict": verdict,
        "notes": notes,
    }


def run_qa(designs: Path, takes_dir: Path, album: str) -> dict:
    rows = []
    for design in load_designs(designs):
        takes = find_takes(takes_dir, design["stem"])
        if not takes:
            rows.append({
                "album": album,
                "track": design["stem"],
                "title": design["title"],
                "take": None,
                "file": "",
                "path": str(takes_dir / design["stem"]),
                "duration_s": None,
                "design_duration_s": design["duration_s"],
                "duration_ok": False,
                "peak_dbfs": None,
                "clipped": False,
                "leading_silence_s": None,
                "trailing_silence_s": None,
                "internal_silence_s": None,
                "silence_ok": False,
                "lufs_i": None,
                "loudness_ok": False,
                "verdict": "FAIL",
                "notes": [f"no takes in {takes_dir / design['stem']} (expected take-N.flac)"],
            })
            continue
        for number, path in takes:
            row = check_take(path, design["duration_s"])
            row.update({
                "album": album,
                "track": design["stem"],
                "title": design["title"],
                "take": number,
            })
            rows.append(row)
    failed = [row for row in rows if row["verdict"] != "PASS"]
    return {
        "album": album,
        "designs": str(designs),
        "renders": str(takes_dir),
        "takes": len(rows),
        "failures": len(failed),
        "verdict": "FAIL" if failed else "PASS",
        "rows": rows,
    }


def format_tsv(report: dict) -> str:
    header = [
        "album", "track", "title", "take", "file", "duration_s", "design_duration_s",
        "duration_ok", "peak_dbfs", "clipped", "leading_silence_s", "trailing_silence_s",
        "internal_silence_s", "silence_ok", "lufs_i", "loudness_ok", "verdict", "notes",
    ]
    lines = ["\t".join(header)]
    for row in report["rows"]:
        notes = "; ".join(row.get("notes") or [])
        values = []
        for key in header[:-1]:
            value = row.get(key)
            values.append("" if value is None else str(value))
        values.append(notes.replace("\t", " "))
        lines.append("\t".join(values))
    return "\n".join(lines) + "\n"


def write_report(report: dict, path: Path, fmt: str) -> None:
    from mixgen.util import atomic_write_json, atomic_write_text

    path = Path(path)
    if fmt == "tsv":
        atomic_write_text(path, format_tsv(report))
    elif fmt == "json":
        atomic_write_json(path, report)
    else:
        raise RelayError(f"unknown report format {fmt}", exit_code=2)
