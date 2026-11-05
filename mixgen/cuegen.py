"""Turn a mixgen plan into a djmix cue sheet, timeline, tracklist and chapters.

Bar math mirrors ``djmix.plan.build_plan``: a track starts ``mix_in_bars`` before
the previous one ends, and overlapping beats either hold the outgoing tempo or
ramp linearly into the incoming one. Everything is phrase aligned (16/32 bars).
"""
from __future__ import annotations

from dataclasses import dataclass

from mixgen.plan import Plan, Track, effective_play_bars
from mixgen.util import atomic_write_text, fmt_hms
import yaml


@dataclass
class Slot:
    index: int
    track: Track
    out_bar0: int
    bars: int
    overlap_in: int
    start_s: float
    end_s: float

    @property
    def duration_s(self) -> float:
        return self.end_s - self.start_s


def timeline(plan: Plan) -> list[Slot]:
    bpb = plan.beats_per_bar
    slots: list[Slot] = []
    out_bar = 0
    for i, track in enumerate(plan.tracks):
        bars = effective_play_bars(track, plan.phrase_bars)
        overlap = 0
        if i:
            overlap = track.mix_in_bars
            prev = slots[-1]
            if overlap > prev.bars:
                raise ValueError(f"track {i} '{track.title}': mix_in_bars {overlap} exceeds previous track ({prev.bars} bars)")
            out_bar = prev.out_bar0 + prev.bars - overlap
        slots.append(Slot(i, track, out_bar, bars, overlap, 0.0, 0.0))
    total_bars = slots[-1].out_bar0 + slots[-1].bars

    bar_bpm = [0.0] * (total_bars * bpb)
    for s in slots:
        for k in range(s.out_bar0 * bpb, (s.out_bar0 + s.bars) * bpb):
            bar_bpm[k] = s.track.bpm
    for i in range(1, len(slots)):
        cur, prev = slots[i], slots[i - 1]
        n = cur.overlap_in * bpb
        if n <= 0:
            continue
        start = cur.out_bar0 * bpb
        if cur.track.ramp:
            for k in range(n):
                frac = (k + 0.5) / n
                bar_bpm[start + k] = prev.track.bpm + (cur.track.bpm - prev.track.bpm) * frac
        else:
            for k in range(n):
                bar_bpm[start + k] = prev.track.bpm
    times = [0.0]
    for bpm in bar_bpm:
        times.append(times[-1] + 60.0 / bpm)
    for s in slots:
        s.start_s = times[s.out_bar0 * bpb]
        s.end_s = times[(s.out_bar0 + s.bars) * bpb]
    return slots


def total_duration(plan: Plan) -> float:
    slots = timeline(plan)
    return slots[-1].end_s


def build_cue(plan: Plan) -> dict:
    slots = timeline(plan)
    tracks = []
    for s in slots:
        t = s.track
        tracks.append({
            "file": plan.rel_to_root(plan.master_path(t)),
            "title": t.title,
            "target_bpm": t.bpm,
            "play_bars": s.bars,
            "mix_in_bars": 0 if s.index == 0 else t.mix_in_bars,
            "transition": t.transition,
            "ramp": bool(t.ramp) if s.index else False,
        })
    return {
        "title": plan.title,
        "output": plan.rel_to_root(plan.mix_wav()),
        "tracklist": plan.rel_to_root(plan.tracklist_path()),
        "sample_rate": plan.sample_rate,
        "phrase_bars": plan.phrase_bars,
        "beats_per_bar": plan.beats_per_bar,
        "target_lufs": plan.target_lufs,
        "target_tp": plan.target_tp,
        "loudnorm": True,
        "tracks": tracks,
    }


def write_cue(plan: Plan) -> str:
    text = yaml.safe_dump(build_cue(plan), sort_keys=False, allow_unicode=True)
    atomic_write_text(plan.cue_path(), text)
    return text


def tracklist(plan: Plan) -> str:
    slots = timeline(plan)
    lines = []
    for s in slots:
        t = s.track
        how = "" if s.index == 0 else f"  [{t.transition}, {s.overlap_in} bars{', ramp' if t.ramp and s.overlap_in else ''}]"
        lines.append(f"{fmt_hms(s.start_s)}  {t.title}  ({t.key}, {t.bpm:g} BPM, "
                     f"{s.bars} bars, {fmt_hms(s.duration_s)}){how}")
    lines.append(f"{fmt_hms(slots[-1].end_s)}  end")
    return "\n".join(lines) + "\n"


def chapters(plan: Plan) -> str:
    """YouTube chapter text: first chapter at 0:00, one per track start."""
    slots = timeline(plan)
    lines = [f"{fmt_hms(s.start_s)} {s.track.title}" for s in slots]
    if lines and not lines[0].startswith("00:00 "):
        lines[0] = "00:00 " + slots[0].track.title
    return "\n".join(lines) + "\n"


def write_text_outputs(plan: Plan) -> dict:
    write_cue(plan)
    atomic_write_text(plan.tracklist_path(), tracklist(plan))
    atomic_write_text(plan.chapters_path(), chapters(plan))
    return {"cue": str(plan.cue_path()), "tracklist": str(plan.tracklist_path()),
            "chapters": str(plan.chapters_path()),
            "duration_s": round(total_duration(plan), 3)}
