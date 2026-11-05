"""Turn a cue sheet + analyses into a bar-accurate output timeline (tempo map)."""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from djmix.analysis import Analysis
from djmix.cuesheet import CueSheet, Track


@dataclass
class Placement:
    index: int
    track: Track
    analysis: Analysis
    src_bar0: float          # source seconds of the first bar we play
    src_period: float        # source seconds per beat
    bars: int                # bars played
    out_bar0: int            # output bar index where this track starts
    overlap_in: int          # bars overlapping the previous track
    overlap_out: int = 0     # bars overlapping the next track



@dataclass
class Plan:
    cue: CueSheet
    placements: list[Placement]
    beat_bpm: np.ndarray            # BPM of each output beat
    beat_times: np.ndarray          # output seconds of each beat boundary (len = beats+1)
    warnings: list[str] = field(default_factory=list)

    @property
    def duration(self) -> float:
        return float(self.beat_times[-1])

    def out_beat_times(self, p: Placement) -> np.ndarray:
        bpb = self.cue.beats_per_bar
        b0 = p.out_bar0 * bpb
        return self.beat_times[b0:b0 + p.bars * bpb + 1]


def _phrase_floor(bars: int, phrase: int) -> int:
    return (bars // phrase) * phrase


def build_plan(cue: CueSheet, analyses: list[Analysis]) -> Plan:
    bpb = cue.beats_per_bar
    phrase = cue.phrase_bars
    warnings: list[str] = []
    placements: list[Placement] = []
    out_bar = 0
    for i, (t, a) in enumerate(zip(cue.tracks, analyses)):
        period = 60.0 / (t.bpm or a.bpm)
        bar_len = period * bpb
        bar0 = a.first_downbeat + t.start_bar * bar_len
        avail = int(np.floor((a.duration - bar0) / bar_len + 1e-6))
        bars = t.play_bars if t.play_bars else _phrase_floor(avail, phrase)
        if bars > avail:
            warnings.append(f"track {i} '{t.title}': play_bars {bars} > {avail} available; clipped")
            bars = _phrase_floor(avail, phrase)
        if bars <= 0:
            raise ValueError(f"track {i} '{t.title}': no full phrase of audio after start_bar")
        overlap = t.mix_in_bars
        if i and overlap > placements[-1].bars:
            raise ValueError(f"track {i}: mix_in_bars {overlap} longer than previous track ({placements[-1].bars} bars)")
        if i and overlap > bars:
            raise ValueError(f"track {i}: mix_in_bars {overlap} longer than the track itself ({bars} bars)")
        if i:
            out_bar = placements[-1].out_bar0 + placements[-1].bars - overlap
            placements[-1].overlap_out = overlap
        ratio = (t.bpm or a.bpm) / t.target_bpm
        if not 0.8 <= ratio <= 1.25:
            warnings.append(f"track {i} '{t.title}': stretch {a.bpm:.2f}->{t.target_bpm} BPM is {abs(1 - ratio) * 100:.0f}%, may sound rough")
        placements.append(Placement(i, t, a, bar0, period, bars, out_bar, overlap))

    total_bars = placements[-1].out_bar0 + placements[-1].bars
    bar_bpm = np.zeros(total_bars * bpb)
    for p in placements:  # each track's own target owns its bars ...
        s = p.out_bar0 * bpb
        bar_bpm[s:s + p.bars * bpb] = p.track.target_bpm
    for i in range(1, len(placements)):  # ... overlaps: hold outgoing tempo, or ramp
        prev, cur = placements[i - 1], placements[i]
        s = cur.out_bar0 * bpb
        n = cur.overlap_in * bpb
        if n == 0:
            continue
        if cur.track.ramp:
            frac = (np.arange(n) + 0.5) / n
            bar_bpm[s:s + n] = prev.track.target_bpm + (cur.track.target_bpm - prev.track.target_bpm) * frac
        else:
            bar_bpm[s:s + n] = prev.track.target_bpm
    beat_times = np.concatenate([[0.0], np.cumsum(60.0 / bar_bpm)])
    return Plan(cue, placements, bar_bpm, beat_times, warnings)


def fmt_time(sec: float) -> str:
    sec = max(0.0, sec)
    h, rem = divmod(int(round(sec)), 3600)
    m, s = divmod(rem, 60)
    return f"{h:02d}:{m:02d}:{s:02d}" if h else f"{m:02d}:{s:02d}"


def tracklist(plan: Plan) -> str:
    bpb = plan.cue.beats_per_bar
    lines = []
    for p in plan.placements:
        start = plan.beat_times[p.out_bar0 * bpb]
        key = f"{p.analysis.key}" + (f" / {p.analysis.camelot}" if p.analysis.camelot else "")
        how = "" if p.index == 0 else f"  [{p.track.transition}, {p.overlap_in} bars{', ramp' if p.track.ramp else ''}]"
        lines.append(f"{fmt_time(start)}  {p.track.title}  ({p.analysis.bpm:.2f} -> {p.track.target_bpm:g} BPM, {key}){how}")
    lines.append(f"{fmt_time(plan.duration)}  end")
    return "\n".join(lines) + "\n"


def describe(plan: Plan) -> str:
    bpb = plan.cue.beats_per_bar
    out = [f"djmix plan: {len(plan.placements)} tracks, {fmt_time(plan.duration)} total, "
           f"{plan.cue.sample_rate} Hz / 24-bit, phrase {plan.cue.phrase_bars} bars, "
           f"loudnorm {'%.1f LUFS / %.1f dBTP' % (plan.cue.target_lufs, plan.cue.target_tp) if plan.cue.loudnorm else 'off'}",
           f"output:    {plan.cue.output}", f"tracklist: {plan.cue.tracklist}", ""]
    for p in plan.placements:
        t0 = plan.beat_times[p.out_bar0 * bpb]
        t1 = plan.beat_times[(p.out_bar0 + p.bars) * bpb]
        out.append(f"[{p.index}] {p.track.title}")
        out.append(f"    file: {p.track.file}{'' if p.track.file.exists() else '  (MISSING)'}")
        out.append(f"    detected {p.analysis.bpm:.2f} BPM ({p.analysis.method}), key {p.analysis.key} {p.analysis.camelot}, "
                   f"first downbeat {p.analysis.first_downbeat:.3f}s")
        out.append(f"    plays source bars {p.track.start_bar}..{p.track.start_bar + p.bars} "
                   f"(from {p.src_bar0:.2f}s) -> out bars {p.out_bar0}..{p.out_bar0 + p.bars} "
                   f"({fmt_time(t0)}-{fmt_time(t1)}) at {p.track.target_bpm:g} BPM")
        if p.index:
            out.append(f"    in: {p.track.transition} over {p.overlap_in} bars{' with tempo ramp' if p.track.ramp else ''}")
    if plan.warnings:
        out.append("")
        out += [f"WARNING: {w}" for w in plan.warnings]
    out.append("")
    out.append("tracklist:")
    out.append(tracklist(plan))
    return "\n".join(out)
