"""``relay-run`` command line.

Subcommands: validate, qa, master, mix, split-sideb, album, mirror.
Each writing command accepts ``--dry-run`` and then does not write its outputs.
Failures exit non-zero and print a ``relay-run:`` line to stderr.

The tool does not render and does not use a GPU. Paths come from flags or from
``RELAY_DESIGNS_DIR``, ``RELAY_RENDERS_DIR``, ``RELAY_RELEASE_DIR``,
``RELAY_MIRROR_HOST``, ``RELAY_MIRROR_PATH``, and ``RELAY_MIRROR_RSH``.
Takes are laid out as ``<renders>/<album>/<NN-slug>/take-N.flac``.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import soundfile as sf

from relayrun import __version__
from relayrun.errors import RelayError

CUE_HELP = """
cue sheet (YAML):
  version: 1
  sample_rate: 44100
  output: mix.wav
  tracks:
    - title: Amber Kiln
      file: amber.wav          # relative to --root, else the cue directory
      bpm: 120
      transition: blend        # blend (equal-power) or cut
      blend_s: 2.0             # or mix_in_bars, which needs bpm
      start_s: 0
      duration_s: 8
The first track is always a cut. split-sideb places each cut at the midpoint
of the blend (a bar line when mix_in_bars is even) and snaps to a zero crossing.
"""


def _env_path(name: str, default: str) -> Path:
    value = os.environ.get(name)
    return Path(value) if value else Path(default)


def _designs(args) -> Path:
    if getattr(args, "designs", None):
        return Path(args.designs)
    return _env_path("RELAY_DESIGNS_DIR", "designs")


def _renders(args) -> Path:
    if getattr(args, "renders", None):
        return Path(args.renders)
    return _env_path("RELAY_RENDERS_DIR", "renders")


def _release(args) -> Path:
    if getattr(args, "release", None):
        return Path(args.release)
    return _env_path("RELAY_RELEASE_DIR", "release")


def _takes_dir(renders: Path, album: str | None) -> Path:
    if album:
        nested = renders / album
        if nested.is_dir():
            return nested
        if renders.is_dir() and any(renders.glob("*/take-*")):
            return renders
        return nested
    return renders


def cmd_validate(args) -> int:
    from relayrun.content_gate import check_tree, render_markdown, run_external, write_results

    designs = _designs(args)
    if not designs.is_dir():
        raise RelayError(f"designs directory not found: {designs}", exit_code=2)
    script = args.content_gate or os.environ.get("RELAY_CONTENT_GATE")
    report = Path(args.report) if args.report else None
    if script:
        code = run_external(Path(script), designs, None if args.dry_run else report)
        if code == 0:
            print(f"PASS  {designs}  (external content gate)")
        else:
            print(f"relay-run: validate FAIL {designs} (external content gate)", file=sys.stderr)
        return 0 if code == 0 else 1
    results = check_tree(designs)
    text = render_markdown(designs, results)
    if args.dry_run:
        sys.stdout.write(text)
    else:
        write_results(results, report)
        for result in results:
            print(f"{result.as_dict()['verdict']}  {result.name}  tracks={result.tracks} fails={len(result.fails)}")
    failed = [item for result in results for item in result.fails]
    if failed:
        for check, message in failed[:30]:
            print(f"relay-run: validate FAIL [{check}] {message}", file=sys.stderr)
        if len(failed) > 30:
            print(f"relay-run: validate FAIL ... {len(failed) - 30} more", file=sys.stderr)
        return 1
    return 0


def cmd_qa(args) -> int:
    from relayrun.qa import format_tsv, run_qa, write_report

    designs = _designs(args)
    renders = _renders(args)
    takes = _takes_dir(renders, args.album)
    if not takes.is_dir():
        raise RelayError(f"renders directory not found: {takes}", exit_code=2)
    album = args.album or takes.name
    report = run_qa(designs, takes, album)
    fmt = args.format
    if args.report:
        path = Path(args.report)
    else:
        path = takes / ("qa-report.tsv" if fmt == "tsv" else "qa-report.json")
    if args.dry_run:
        sys.stdout.write(format_tsv(report) if fmt == "tsv" else json.dumps(report, indent=2) + "\n")
        print(f"(dry run: {path} not written)")
    else:
        write_report(report, path, fmt)
        print(f"{report['verdict']}  {album}  takes={report['takes']} failures={report['failures']}  {path}")
    if report["verdict"] != "PASS":
        for row in report["rows"]:
            if row["verdict"] == "PASS":
                continue
            note = "; ".join(row.get("notes") or [])
            print(f"relay-run: qa FAIL {row.get('track')} take {row.get('take')}: {note}", file=sys.stderr)
        return 1
    return 0


def _master_one(src: Path, dst: Path, target: float, dry_run: bool) -> int:
    from relayrun.master import OUTPUT_SR, TRUE_PEAK_MAX, lufs_window, master_failure_message, master_file

    if not src.is_file():
        raise RelayError(f"input not found: {src}", exit_code=2)
    low, high = lufs_window()
    if dry_run:
        print(
            f"master {src} -> {dst}\n"
            f"  trim edges below -60 dBFS, keep 0.25 s pad (interior untouched, no fade)\n"
            f"  loudnorm I={target:g} LUFS, TP=-1.5, linear; "
            f"{OUTPUT_SR} Hz 24-bit stereo\n"
            f"  verify {low:.0f}..{high:.0f} LUFS, true peak <= {TRUE_PEAK_MAX:.1f} dBTP"
        )
        return 0
    report = master_file(src, dst, target_lufs=target)
    print(
        f"{report['verdict']}  {dst.name}  {report['lufs_i']} LUFS  "
        f"{report['true_peak_dbtp']} dBTP  {report['loudnorm_mode']}"
    )
    if report["verdict"] != "PASS":
        print(f"relay-run: {master_failure_message(report)}", file=sys.stderr)
        return 1
    return 0


def cmd_master(args) -> int:
    from relayrun.master import iter_side_a, resolve_take

    target = float(args.lufs)
    if args.input or args.output:
        if not args.input or not args.output:
            raise RelayError("master needs both --input and --output", exit_code=2)
        return _master_one(Path(args.input), Path(args.output), target, args.dry_run)
    designs = _designs(args)
    renders = _renders(args)
    release = _release(args)
    if not designs.is_dir():
        raise RelayError(f"designs directory not found: {designs}", exit_code=2)
    rows = iter_side_a(designs)
    if not rows:
        raise RelayError(f"no side A designs in {designs}", exit_code=2)
    failed = 0
    for stem, _data in rows:
        src = resolve_take(renders, args.album, stem, int(args.take))
        dst = release / "A" / f"{stem}.wav"
        failed += _master_one(src, dst, target, args.dry_run) != 0
    if failed:
        print(f"relay-run: master FAIL {failed} track(s)", file=sys.stderr)
        return 1
    return 0


def cmd_mix(args) -> int:
    from mixgen.util import atomic_write_json

    from relayrun.mix import assemble, describe, load_cue

    cue = load_cue(Path(args.cue), Path(args.root) if args.root else None)
    output = Path(args.output) if args.output else (cue.output or (cue.path.parent / "mix.wav"))
    pcm, timeline = assemble(cue)
    print(describe(timeline))
    if args.dry_run:
        print(f"(dry run: {output} not written)")
        return 0
    output.parent.mkdir(parents=True, exist_ok=True)
    sf.write(str(output), pcm, cue.sample_rate, subtype="PCM_24")
    timeline_path = output.with_suffix(".timeline.json")
    atomic_write_json(timeline_path, timeline)
    print(f"wrote {output}")
    print(f"wrote {timeline_path}")
    return 0


def cmd_split(args) -> int:
    from relayrun.master import master_failure_message, master_file
    from relayrun.mix import describe
    from relayrun.split import load_timeline, split_mastered

    mix = Path(args.mix)
    if not mix.is_file():
        raise RelayError(f"mix not found: {mix}", exit_code=2)
    out = Path(args.out) if args.out else mix.parent / "B"
    timeline = load_timeline(
        Path(args.timeline) if args.timeline else None,
        Path(args.cue) if args.cue else None,
        Path(args.root) if args.root else None,
    )
    if args.dry_run:
        print(f"split-sideb: master {mix} once -> {out / 'full-mix.wav'}")
        print("  then cut at each blend midpoint, snap to a zero crossing, no fades")
        print(describe(timeline))
        for row in timeline.get("tracks") or []:
            title = row["title"]
            mixed = title if str(title).endswith(" (Mixed)") else f"{title} (Mixed)"
            print(f"  piece {mixed}")
        print("(dry run: nothing written)")
        return 0
    report = master_file(mix, out / "full-mix.wav", target_lufs=float(args.lufs))
    print(
        f"{report['verdict']}  full mix  {report['lufs_i']} LUFS  {report['true_peak_dbtp']} dBTP"
    )
    if report["verdict"] != "PASS":
        print(f"relay-run: {master_failure_message(report)}", file=sys.stderr)
        return 1
    manifest = split_mastered(out / "full-mix.wav", timeline, out, report, snap_ms=float(args.snap_ms))
    print(f"pcm_md5 {manifest['pcm_md5']}  pieces={len(manifest['pieces'])}")
    for piece in manifest["pieces"]:
        print(f"  {piece['file']}  {piece['title']}  {piece['frames']} frames")
    return 0


def cmd_album(args) -> int:
    from mixgen.util import atomic_write_text

    from relayrun.album import (
        assert_names_absent,
        assert_sheet_safe,
        build_rows,
        name_tokens,
        render_tsv,
    )

    designs = _designs(args)
    release = _release(args)
    rows, problems = build_rows(designs, release)
    tokens: list[str] = []
    if designs.is_dir():
        for path in sorted(designs.glob("*.json")):
            try:
                data = json.loads(path.read_text(encoding="utf-8"))
            except json.JSONDecodeError:
                continue
            tokens.extend(name_tokens(data))
    if problems:
        for problem in problems:
            print(f"relay-run: album FAIL {problem}", file=sys.stderr)
        return 1
    text = render_tsv(rows)
    assert_sheet_safe(text)
    assert_names_absent(text, tokens)
    out = Path(args.out) if args.out else release / "RELEASE.tsv"
    if args.dry_run:
        sys.stdout.write(text)
        print(f"(dry run: {out} not written)")
        return 0
    atomic_write_text(out, text)
    print(f"wrote {out}  tracks={len(rows)}")
    return 0


def cmd_mirror(args) -> int:
    from relayrun.mirror import mirror

    host = args.host or os.environ.get("RELAY_MIRROR_HOST") or ""
    dest = args.dest or os.environ.get("RELAY_MIRROR_PATH") or ""
    rsh = args.rsh or os.environ.get("RELAY_MIRROR_RSH") or "ssh"
    release = Path(args.release) if args.release else _release(args)
    result = mirror(release, host, dest, rsh=rsh, dry_run=args.dry_run)
    if result["dry_run"]:
        print(" ".join(result["rsync"]))
        print(" ".join(result["verify"]))
        print("(dry run: nothing copied)")
        return 0
    print(f"mirrored {result['files']} files, checksums ok")
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="relay-run",
        description=(
            "Album release pipeline: validate designs, QA takes, master, mix, "
            "split a continuous mix, write RELEASE.tsv, and mirror the release. "
            "Does not render and does not use a GPU. "
            "Defaults: RELAY_DESIGNS_DIR=designs, RELAY_RENDERS_DIR=renders, "
            "RELAY_RELEASE_DIR=release. Takes: <renders>/<album>/<NN-slug>/take-N.flac."
        ),
    )
    parser.add_argument("--version", action="version", version=f"relay-run {__version__}")
    sub = parser.add_subparsers(dest="cmd", required=True)

    validate = sub.add_parser("validate", help="run the content gate on an album's designs")
    validate.add_argument("--designs", help="album directory, or a root of album directories")
    validate.add_argument("--report", help="also write a markdown report to this path")
    validate.add_argument("--content-gate", help="Python file with main(root, report); else the built-in gate")
    validate.add_argument("--dry-run", action="store_true", help="print the verdict and do not write GATE files")
    validate.set_defaults(func=cmd_validate)

    qa = sub.add_parser("qa", help="per-take duration, clipping, silence, and loudness checks")
    qa.add_argument("--designs", help="directory of NN-slug.json designs")
    qa.add_argument("--renders", help="renders root (default RELAY_RENDERS_DIR or ./renders)")
    qa.add_argument("--album", help="album slug; takes are <renders>/<album>/<NN-slug>/take-N.flac")
    qa.add_argument("--report", help="report path (default <takes>/qa-report.json)")
    qa.add_argument("--format", choices=("json", "tsv"), default="json")
    qa.add_argument("--dry-run", action="store_true", help="print the report and do not write it")
    qa.set_defaults(func=cmd_qa)

    master = sub.add_parser(
        "master",
        help="master side A to -14 LUFS, true peak <= -1.0 dBTP, 44.1 kHz 24-bit",
    )
    master.add_argument("--input", help="one source file (use with --output)")
    master.add_argument("--output", help="one 24-bit WAV (use with --input)")
    master.add_argument("--designs", help="directory of NN-slug.json designs")
    master.add_argument("--renders", help="renders root")
    master.add_argument("--release", help="release directory; writes A/<NN-slug>.wav")
    master.add_argument("--album", help="album slug under the renders root")
    master.add_argument("--take", type=int, default=1, help="take number to master (default 1)")
    master.add_argument("--lufs", type=float, default=-14.0, help="integrated loudness target (default -14)")
    master.add_argument("--dry-run", action="store_true", help="print the chain and do not write")
    master.set_defaults(func=cmd_master)

    mix = sub.add_parser(
        "mix",
        help="assemble a continuous mix from a cue sheet",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        description="Assemble a continuous mix from a cue sheet." + CUE_HELP,
    )
    mix.add_argument("--cue", required=True, help="cue sheet YAML")
    mix.add_argument("--root", help="base directory for relative audio paths")
    mix.add_argument("--output", help="output WAV (default: cue output, else mix.wav next to the cue)")
    mix.add_argument("--dry-run", action="store_true", help="print the track list and do not write")
    mix.set_defaults(func=cmd_mix)

    split = sub.add_parser(
        "split-sideb",
        help="master the whole mix once, then cut one '(Mixed)' file per song",
        description=(
            "Master the whole mix once to the release window, then cut it at the "
            "midpoint of each blend on a zero crossing. Pieces are sample-accurate "
            "and gapless: concatenating their PCM matches the mastered mix. "
            "Per-piece loudness is measured and is allowed to vary."
        ),
    )
    split.add_argument("--mix", required=True, help="continuous mix WAV")
    split.add_argument("--cue", help="cue sheet, used when --timeline is omitted")
    split.add_argument("--timeline", help="timeline JSON written by mix (mix.timeline.json)")
    split.add_argument("--root", help="base directory for cue audio paths")
    split.add_argument("--out", help="output directory (default <mix dir>/B)")
    split.add_argument("--lufs", type=float, default=-14.0)
    split.add_argument("--snap-ms", type=float, default=20.0, help="zero-crossing search window (default 20 ms)")
    split.add_argument("--dry-run", action="store_true", help="print the cuts and do not write")
    split.set_defaults(func=cmd_split)

    album = sub.add_parser("album", help="write RELEASE.tsv (songwriter column stays blank)")
    album.add_argument("--designs", help="directory of NN-slug.json designs")
    album.add_argument("--release", help="release directory with A/ and optional B/split-manifest.json")
    album.add_argument("--out", help="RELEASE.tsv path (default <release>/RELEASE.tsv)")
    album.add_argument("--dry-run", action="store_true", help="print the sheet and do not write it")
    album.set_defaults(func=cmd_album)

    mirror = sub.add_parser(
        "mirror",
        help="rsync the release directory to another host and verify checksums",
        description=(
            "Copy a release directory with rsync over ssh and check MANIFEST.md5 "
            "on the destination. The host and path come from --host/--dest or "
            "RELAY_MIRROR_HOST/RELAY_MIRROR_PATH. There is no default host."
        ),
    )
    mirror.add_argument("--release", help="release directory to copy")
    mirror.add_argument("--host", help="ssh host or user@host (or RELAY_MIRROR_HOST)")
    mirror.add_argument("--dest", help="absolute path on that host (or RELAY_MIRROR_PATH)")
    mirror.add_argument("--rsh", help="remote shell passed to rsync -e (default ssh, or RELAY_MIRROR_RSH)")
    mirror.add_argument("--dry-run", action="store_true", help="print the rsync and verify commands")
    mirror.set_defaults(func=cmd_mirror)
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        return int(args.func(args))
    except RelayError as exc:
        print(f"relay-run: {exc}", file=sys.stderr)
        return int(exc.exit_code)


if __name__ == "__main__":
    raise SystemExit(main())
