"""djmix CLI: python -m djmix CUE.yaml [--dry-run] [--root DIR] [--engine auto|rubberband|librosa]."""
from __future__ import annotations

import argparse
import json
import sys
import time

from djmix import __version__
from djmix.analysis import analyze
from djmix.cuesheet import CueError, load
from djmix.plan import build_plan, describe


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="djmix", description="Render a beat/phrase-aligned DJ mix from a YAML cue sheet.")
    ap.add_argument("cue", help="cue sheet YAML")
    ap.add_argument("--root", help="base dir for relative paths (default: cue 'root', $DJMIX_ROOT, or /data/media/archive)")
    ap.add_argument("--dry-run", action="store_true", help="analyse and print the plan + tracklist; render nothing")
    ap.add_argument("--engine", choices=("auto", "rubberband", "librosa"), default="auto",
                    help="time-stretch engine (auto = rubberband CLI if installed)")
    ap.add_argument("--no-loudnorm", action="store_true")
    ap.add_argument("--json", action="store_true", help="print the final report as JSON")
    ap.add_argument("--version", action="version", version=f"djmix {__version__}")
    a = ap.parse_args(argv)
    try:
        cue = load(a.cue, a.root)
    except (CueError, OSError) as e:
        print(f"djmix: cue sheet error: {e}", file=sys.stderr)
        return 2
    if a.no_loudnorm:
        cue.loudnorm = False
    missing = [str(t.file) for t in cue.tracks if not t.file.exists()]
    if missing:
        print("djmix: missing input files:\n  " + "\n  ".join(missing), file=sys.stderr)
        return 2
    t0 = time.time()
    analyses = []
    for i, t in enumerate(cue.tracks):
        print(f"analysing [{i}] {t.file.name} ...", file=sys.stderr)
        analyses.append(analyze(t.file, bpm=t.bpm, first_downbeat=t.first_downbeat, beats_per_bar=cue.beats_per_bar))
    try:
        plan = build_plan(cue, analyses)
    except ValueError as e:
        print(f"djmix: plan error: {e}", file=sys.stderr)
        return 2
    print(describe(plan))
    if a.dry_run:
        print("(dry run: nothing rendered)")
        return 0
    from djmix.render import render
    report = render(plan, a.engine, log=lambda m: print(m, file=sys.stderr))
    report["elapsed_s"] = round(time.time() - t0, 1)
    print(json.dumps(report, indent=2) if a.json else
          f"wrote {report['output']} ({report['duration_s'] / 60:.1f} min)"
          + (f", {report['loudness']['integrated_lufs']:.1f} LUFS, {report['loudness']['true_peak_dbtp']:.1f} dBTP "
             f"[{report['loudness']['mode']}]" if 'loudness' in report else "")
          + f"; tracklist {report['tracklist']}")
    return 0
