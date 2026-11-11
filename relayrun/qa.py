"""Per-take QA: duration, peak, flat-top clipping, silence, and loudness.

Takes live at ``<renders>/<album>/<NN-slug>/take-N.flac`` (``.wav`` is also read).
QA reads the file and does not change it. Thresholds:

- duration within ``max(1 s, 2%)`` of the design ``duration``
- sample peak at or above -0.1 dBFS is a warning (near full scale). It does
  not fail the take and it is not clipping
- clipping is a flat-top: consecutive identical samples at or above -0.1 dBFS,
  longer than a 20 Hz full-scale crest of that integer format (at least 3)
- a 4x oversampled peak is reported next to the sample peak. An intersample
  overshoot is not clipping and is not the release -1.0 dBTP gate
- leading or trailing silence fails when it is longer than 0.5 s below
  -60 dBFS RMS (loudest channel)
- internal dropout is the same level and duration, only when both sides have
  at least 0.05 s of non-silent audio. Edge silence, a zero crossing, a
  sub-context click, and a quiet passage are not dropouts
- integrated loudness outside -40..-8 LUFS fails. That is a raw-take window,
  not the release target of -14 LUFS / -1.0 dBTP

Near-zero samples are not evidence that a take was normalised. This check
does not record a normalisation verdict.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

import numpy as np

from relayrun.audio import (
    FLAT_TOP_REF_HZ,
    INTERSAMPLE_DISTINCT_DB,
    PEAK_WARN_DBFS,
    QA_OVERSAMPLE,
    SILENCE_CONTEXT_S,
    SILENCE_DB,
    SILENCE_FRAME_S,
    SILENCE_LIMIT_S,
    clipping_report,
    integrated_lufs,
    load_audio,
    probe_format,
    silence_regions,
)
from relayrun.errors import RelayError

DURATION_ABS_S = 1.0
DURATION_RATIO = 0.02
LOUDNESS_MIN = -40.0
LOUDNESS_MAX = -8.0
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


def qa_thresholds() -> dict:
    """Numbers the report is judging against. Release limits are quoted, not applied."""
    from relayrun.master import LUFS_TOLERANCE, TARGET_LUFS, TRUE_PEAK_MAX

    return {
        "duration_abs_s": DURATION_ABS_S,
        "duration_ratio": DURATION_RATIO,
        "duration_unit": "s",
        "peak_warning_dbfs": PEAK_WARN_DBFS,
        "peak_warning_unit": "dBFS re full scale 1.0",
        "peak_warning_fails_take": False,
        "clipping_definition": (
            "contiguous identical decoded samples at or above "
            f"{PEAK_WARN_DBFS} dBFS, longer than a {FLAT_TOP_REF_HZ:g} Hz full-scale crest "
            "of the file's integer format, and at least 3 samples"
        ),
        "clipping_fails_take": True,
        "silence_rms_dbfs": SILENCE_DB,
        "silence_rms_rule": "loudest channel, strictly below the threshold",
        "silence_limit_s": SILENCE_LIMIT_S,
        "silence_comparison": "fail when a reported silence duration is greater than silence_limit_s",
        "silence_frame_s": SILENCE_FRAME_S,
        "silence_context_s": SILENCE_CONTEXT_S,
        "raw_loudness_lufs": [LOUDNESS_MIN, LOUDNESS_MAX],
        "raw_loudness_unit": "LUFS integrated",
        "raw_loudness_is_release_target": False,
        "release_target_lufs": TARGET_LUFS,
        "release_lufs_tolerance": LUFS_TOLERANCE,
        "release_true_peak_max_dbtp": TRUE_PEAK_MAX,
        "qa_oversample": QA_OVERSAMPLE,
        "intersample_distinct_db": INTERSAMPLE_DISTINCT_DB,
        "intersample_fails_take": False,
        "normalization_assessed": False,
        "audio_rewritten": False,
    }


def _silence_notes(regions: dict) -> list[str]:
    notes = []
    limit = SILENCE_LIMIT_S
    level = SILENCE_DB
    if regions["leading_s"] > limit:
        notes.append(
            f"leading silence {regions['leading_s']:.3f} s "
            f"(threshold > {limit:.1f} s of loudest-channel RMS below {level:.0f} dBFS)"
        )
    if regions["trailing_s"] > limit:
        notes.append(
            f"trailing silence {regions['trailing_s']:.3f} s "
            f"(threshold > {limit:.1f} s of loudest-channel RMS below {level:.0f} dBFS)"
        )
    if regions["internal_s"] > limit:
        span = ""
        if regions.get("internal_start_s") is not None:
            span = (
                f" from {regions['internal_start_s']:.3f} s "
                f"to {regions['internal_end_s']:.3f} s"
            )
        notes.append(
            f"internal dropout {regions['internal_s']:.3f} s{span} "
            f"(threshold > {limit:.1f} s contiguous; loudest-channel RMS below {level:.0f} dBFS; "
            f"frame {regions['frame_s']:.3f} s; at least {regions['context_s']:.2f} s of "
            f"non-silent audio on both sides)"
        )
    return notes


def _peak_texts(report: dict) -> tuple[list[str], list[str]]:
    """Return ``(failure notes, warnings)`` for peak and flat-top evidence."""
    notes: list[str] = []
    warnings: list[str] = []
    if report["clipped"]:
        channel = report["flat_top_channel"]
        ch = "none" if channel is None else str(channel)
        level = report["flat_top_value_dbfs"]
        level_txt = "n/a" if level is None else f"{level:.2f} dBFS"
        notes.append(
            f"clipping: flat-top of {report['flat_top_samples']} samples "
            f"({report['flat_top_duration_s']:.6f} s) at {level_txt} on channel {ch} "
            f"of {report['channels']}; samples are identical; threshold >= "
            f"{report['flat_top_min_samples']} samples at or above {PEAK_WARN_DBFS} dBFS "
            f"({report['subtype'] or 'unspecified'} at {report['sample_rate']} Hz, "
            f"longer than a {report['crest_reference_hz']:g} Hz full-scale crest)"
        )
    elif report["near_full_scale"]:
        warnings.append(
            f"near-full-scale sample peak {report['sample_peak_dbfs']:.2f} dBFS "
            f"(warning at or above {PEAK_WARN_DBFS} dBFS); longest identical run "
            f"{report['flat_top_samples']} sample(s), flat-top threshold "
            f"{report['flat_top_min_samples']} samples; not clipping"
        )
    if report["intersample_distinct"] and not report["clipped"]:
        warnings.append(
            f"oversampled {report['oversample']}x peak {report['oversampled_peak_dbtp']:.2f} dBTP "
            f"exceeds sample peak {report['sample_peak_dbfs']:.2f} dBFS by "
            f"{report['intersample_overshoot_db']:.2f} dB "
            f"(distinct above {INTERSAMPLE_DISTINCT_DB} dB); not sample clipping"
        )
    return notes, warnings


def check_take(path: Path, design_duration: float) -> dict:
    pcm, sr = load_audio(path)
    meta = probe_format(path)
    duration = pcm.shape[0] / float(sr)
    tol = duration_tolerance(design_duration)
    duration_ok = abs(duration - design_duration) <= tol
    regions = silence_regions(pcm, sr)
    silence_ok = (
        regions["leading_s"] <= SILENCE_LIMIT_S
        and regions["trailing_s"] <= SILENCE_LIMIT_S
        and regions["internal_s"] <= SILENCE_LIMIT_S
    )
    peaks = clipping_report(pcm, sr, meta["subtype"])
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
            f"duration {duration:.3f} s vs design {design_duration:.3f} s "
            f"(tolerance {tol:.3f} s)"
        )
    clip_notes, warnings = _peak_texts(peaks)
    notes.extend(clip_notes)
    notes.extend(_silence_notes(regions))
    if not loudness_ok:
        shown = "unmeasurable" if not np.isfinite(lufs) else f"{lufs:.2f} LUFS"
        notes.append(
            f"integrated loudness {shown} outside {LOUDNESS_MIN:.0f}..{LOUDNESS_MAX:.0f} LUFS"
            + (f" ({loudness_note})" if loudness_note else "")
        )
    verdict = "PASS" if not notes else "FAIL"
    return {
        "file": path.name,
        "path": str(path),
        "sample_rate": int(sr),
        "channels": int(pcm.shape[1]),
        "subtype": meta["subtype"],
        "duration_s": round(duration, 4),
        "design_duration_s": design_duration,
        "duration_tolerance_s": round(tol, 4),
        "duration_ok": duration_ok,
        "peak_dbfs": peaks["sample_peak_dbfs"],
        "near_full_scale": peaks["near_full_scale"],
        "oversampled_peak_dbtp": peaks["oversampled_peak_dbtp"],
        "intersample_overshoot_db": peaks["intersample_overshoot_db"],
        "intersample_distinct": peaks["intersample_distinct"],
        "flat_top_samples": peaks["flat_top_samples"],
        "flat_top_min_samples": peaks["flat_top_min_samples"],
        "flat_top_channel": peaks["flat_top_channel"],
        "clipped": peaks["clipped"],
        "leading_silence_s": regions["leading_s"],
        "trailing_silence_s": regions["trailing_s"],
        "internal_silence_s": regions["internal_s"],
        "internal_start_s": regions["internal_start_s"],
        "internal_end_s": regions["internal_end_s"],
        "silence_ok": silence_ok,
        "lufs_i": None if not np.isfinite(lufs) else round(float(lufs), 2),
        "loudness_ok": bool(loudness_ok),
        "verdict": verdict,
        "warnings": warnings,
        "notes": notes,
        "peak": peaks,
        "silence": {
            "threshold_dbfs": regions["threshold_dbfs"],
            "limit_s": SILENCE_LIMIT_S,
            "frame_s": regions["frame_s"],
            "context_s": regions["context_s"],
            "channel_rule": regions["channel_rule"],
            "unit": "s",
        },
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
                "sample_rate": None,
                "channels": None,
                "subtype": None,
                "duration_s": None,
                "design_duration_s": design["duration_s"],
                "duration_ok": False,
                "peak_dbfs": None,
                "near_full_scale": False,
                "oversampled_peak_dbtp": None,
                "intersample_overshoot_db": None,
                "intersample_distinct": False,
                "flat_top_samples": None,
                "flat_top_min_samples": None,
                "flat_top_channel": None,
                "clipped": False,
                "leading_silence_s": None,
                "trailing_silence_s": None,
                "internal_silence_s": None,
                "internal_start_s": None,
                "internal_end_s": None,
                "silence_ok": False,
                "lufs_i": None,
                "loudness_ok": False,
                "verdict": "FAIL",
                "warnings": [],
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
        "thresholds": qa_thresholds(),
        "rows": rows,
    }


def format_tsv(report: dict) -> str:
    header = [
        "album", "track", "title", "take", "file", "duration_s", "design_duration_s",
        "duration_ok", "peak_dbfs", "near_full_scale", "oversampled_peak_dbtp",
        "intersample_overshoot_db", "flat_top_samples", "flat_top_min_samples",
        "clipped", "leading_silence_s", "trailing_silence_s",
        "internal_silence_s", "silence_ok", "lufs_i", "loudness_ok", "verdict",
        "warnings", "notes",
    ]

    def cell(key: str, row: dict) -> str:
        value = row.get(key)
        if key in ("notes", "warnings"):
            if isinstance(value, list):
                value = "; ".join(str(item) for item in value)
            text = "" if value is None else str(value)
            return text.replace("\t", " ")
        return "" if value is None else str(value)

    lines = ["\t".join(header)]
    for row in report["rows"]:
        lines.append("\t".join(cell(key, row) for key in header))
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
