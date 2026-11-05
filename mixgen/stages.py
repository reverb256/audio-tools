"""Stage orchestration: gen -> qa -> master -> mix, each resumable.

A stage is skipped when its output already exists (unless ``--force``), and every
write is atomic (temp file/path + ``os.replace``), so an interrupted run resumes
cleanly. ``--dry-run`` prints what each stage would do without touching a backend
or rendering audio, and works with no audio present.
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

from mixgen import cuegen
from mixgen.backends import make_backend
from mixgen.master import master_file
from mixgen.plan import Plan, PlanError
from mixgen.qa import qa_track
from mixgen.util import atomic_write_json, fmt_hms

STAGES = ("gen", "qa", "master", "stems", "mix")
DEFAULT_STAGES = ("gen", "qa", "master", "mix")


def parse_stages(value: str) -> list[str]:
    names = [s.strip() for s in str(value).split(",") if s.strip()]
    unknown = [s for s in names if s not in STAGES]
    if unknown or not names:
        raise PlanError(f"--stages must be a comma list from {STAGES}, got {value!r}")
    # keep canonical order, drop duplicates
    return [s for s in STAGES if s in names]


def read_report(path: Path) -> dict | None:
    try:
        return json.loads(Path(path).read_text())
    except (OSError, ValueError):
        return None


def pick_path(plan: Plan, track) -> Path | None:
    report = read_report(plan.qa_report_path(track))
    if not report:
        return None
    pick = report.get("pick_file")
    if pick and Path(pick).exists():
        return Path(pick)
    return None


# ------------------------------------------------------------------ gen
def run_gen(plan: Plan, backend_name: str, force: bool, log=print) -> dict:
    backend = make_backend(plan, backend_name, log=log)
    made = skipped = 0
    for track in plan.tracks:
        log(f"  [{track.index + 1}/{len(plan.tracks)}] {track.title} ({track.slug})")
        targets: list[tuple[int, Path]] = []
        for n in range(1, track.takes + 1):
            dst = plan.take_path(track, n)
            if dst.exists() and not force:
                skipped += 1
                log(f"    take-{n}: skip ({dst})")
                continue
            dst.parent.mkdir(parents=True, exist_ok=True)
            targets.append((n, dst.with_name(dst.stem + ".tmp" + dst.suffix)))
        if not targets:
            continue
        try:
            # one job per track: --takes N is a batch, not one job per take
            backend.generate_track(track, targets, force=force)
            for n, tmp in targets:
                dst = plan.take_path(track, n)
                os.replace(tmp, dst)
                made += 1
                log(f"    take-{n}: wrote {dst}")
        finally:
            for _n, tmp in targets:
                try:
                    tmp.unlink()
                except FileNotFoundError:
                    pass
    return {"generated": made, "skipped": skipped, "backend": backend_name}


# ------------------------------------------------------------------ qa
def run_qa(plan: Plan, force: bool, log=print) -> dict:
    reports = []
    ran = skipped = 0
    for track in plan.tracks:
        report_path = plan.qa_report_path(track)
        if report_path.exists() and not force:
            rep = read_report(report_path)
            if rep:
                reports.append(rep)
                skipped += 1
                log(f"  [{track.index + 1}/{len(plan.tracks)}] {track.title}: skip (report exists)")
                continue
        takes = [plan.take_path(track, n) for n in range(1, track.takes + 1)]
        missing = [str(p) for p in takes if not p.exists()]
        if missing:
            raise PlanError(f"qa: missing generated takes for '{track.title}': {missing}")
        log(f"  [{track.index + 1}/{len(plan.tracks)}] {track.title}: analysing {len(takes)} takes ...")
        rep = qa_track(track, takes, plan.qa_track_dir(track))
        reports.append(rep)
        ran += 1
        log(f"    pick {rep['pick']} (score {rep['pick_score']:.3f})")
    summary = _summary(plan, reports)
    atomic_write_json(plan.qa_summary_path(), summary)
    return {"analysed": ran, "skipped": skipped, "summary": str(plan.qa_summary_path()),
            "all_pass": summary["all_pass"]}


def _summary(plan: Plan, reports: list[dict]) -> dict:
    rows = []
    for rep in reports:
        t = rep["track"]
        pick = next((k for k in rep["takes"] if k["take"] == int(rep["pick"].split("-")[1])), {})
        sc = pick.get("scores", {})
        rows.append({
            "index": t["index"], "title": t["title"], "slug": t["slug"],
            "pick": rep["pick"], "score": rep["pick_score"],
            "bpm_ok": sc.get("tempo", 0) == 1.0, "key_ok": sc.get("key", 0) == 1.0,
            "drums_ok": sc.get("drums", 0) >= 0.999, "tech_ok": sc.get("tech", 0) == 1.0,
        })
    return {
        "mix": plan.title, "slug": plan.slug,
        "tracks": rows,
        "all_pass": all(r["bpm_ok"] and r["key_ok"] and r["drums_ok"] and r["tech_ok"] for r in rows),
    }


# ------------------------------------------------------------------ master
def run_master(plan: Plan, force: bool, log=print) -> dict:
    made = skipped = 0
    for track in plan.tracks:
        dst = plan.master_path(track)
        if dst.exists() and not force:
            skipped += 1
            log(f"  [{track.index + 1}/{len(plan.tracks)}] {track.title}: skip (mastered exists)")
            continue
        src = pick_path(plan, track)
        if src is None:
            raise PlanError(f"master: no QA pick for '{track.title}' — run the qa stage first")
        log(f"  [{track.index + 1}/{len(plan.tracks)}] {track.title}: loudnorm {src.name} -> {dst}")
        master_file(src, dst, sample_rate=plan.sample_rate,
                    target_lufs=plan.target_lufs, target_tp=plan.target_tp)
        made += 1
    return {"mastered": made, "skipped": skipped}


# ------------------------------------------------------------------ stems (opt-in)
def run_stems(plan: Plan, force: bool, log=print) -> dict:
    """Optional ``gpu-music-job demucs`` step; never runs unless selected."""
    cfg = plan.stems or {}
    if not cfg.get("enabled"):
        raise PlanError(
            "the stems stage is opt-in: add a top-level 'stems: {enabled: true}' block "
            "to the plan (optional: model: htdemucs)")
    from mixgen.backends.acestep import AceStepBackend

    backend = AceStepBackend(plan, log=log)
    model = cfg.get("model") or None
    made = skipped = 0
    for track in plan.tracks:
        src = pick_path(plan, track)
        if src is None:
            raise PlanError(
                f"stems: no QA pick for '{track.title}' — run the qa stage first")
        rel = str(Path(src).relative_to(plan.gen_root)) if Path(src).is_relative_to(plan.gen_root) \
            else str(Path(src).relative_to(plan.root))
        out_rel = plan.stems_rel(track)
        result_path = plan.stems_dir(track) / "job-result.json"
        if result_path.exists() and not force:
            skipped += 1
            log(f"  [{track.index + 1}/{len(plan.tracks)}] {track.title}: skip "
                f"({result_path})")
            continue
        log(f"  [{track.index + 1}/{len(plan.tracks)}] {track.title}: demucs {rel} -> {out_rel}")
        backend.demucs(rel, out_rel, model=model, force=force)
        made += 1
    return {"separated": made, "skipped": skipped, "model": model or "htdemucs"}


# ------------------------------------------------------------------ mix
def _ffmpeg() -> str:
    return os.environ.get("DJMIX_FFMPEG") or os.environ.get("MIXGEN_FFMPEG") or "ffmpeg"


def run_mix(plan: Plan, force: bool, log=print) -> dict:
    missing = [str(plan.master_path(t)) for t in plan.tracks if not plan.master_path(t).exists()]
    if missing:
        raise PlanError(f"mix: missing mastered files: {missing}")
    artifacts = cuegen.write_text_outputs(plan)
    wav = plan.mix_wav()
    if wav.exists() and not force:
        log(f"  wav: skip ({wav})")
    else:
        log(f"  wav: djmix render -> {wav} ({fmt_hms(cuegen.total_duration(plan))})")
        subprocess.run([sys.executable, "-m", "djmix", str(plan.cue_path()),
                        "--root", str(plan.root)], check=True)
    mp3 = plan.mix_mp3()
    if mp3.exists() and not force and wav.exists():
        log(f"  mp3: skip ({mp3})")
    else:
        log(f"  mp3: ffmpeg 320k -> {mp3}")
        tmp = mp3.with_name(mp3.stem + ".tmp.mp3")
        try:
            subprocess.run([_ffmpeg(), "-hide_banner", "-nostats", "-y", "-i", str(wav),
                            "-codec:a", "libmp3lame", "-b:a", "320k", str(tmp)], check=True)
            os.replace(tmp, mp3)
        finally:
            try:
                tmp.unlink()
            except FileNotFoundError:
                pass
    # djmix writes its own tracklist; ours carries the plan's keys/bar counts.
    cuegen.write_text_outputs(plan)
    artifacts.update({"wav": str(wav), "mp3": str(mp3)})
    return artifacts


# ------------------------------------------------------------------ dry run
def _skip_run(ok: bool) -> str:
    return "SKIP" if ok else "RUN"


def dry_run(plan: Plan, stages: list[str], backend_name: str, force: bool, log=print) -> None:
    log(f"mixgen: {plan.title} ({plan.slug})")
    log(f"  root:   {plan.root}")
    log(f"  output: {plan.output}")
    log(f"  stages: {', '.join(stages)}")
    log("")
    if "gen" in stages:
        n_takes = sum(t.takes for t in plan.tracks)
        log(f"gen: backend={backend_name}, {len(plan.tracks)} tracks, "
            f"{n_takes} takes (one job per track)")
        for track in plan.tracks:
            log(f"  [{track.index + 1}/{len(plan.tracks)}] {track.title} ({track.slug}) "
                f"job={track.job} bpm={track.bpm:g} key={track.key} "
                f"dur={fmt_hms(track.duration)} takes={track.takes} "
                f"out-relpath={plan.track_gen_rel(track)}")
            log(f"    style: {track.style}")
            if track.lyrics:
                log(f"    lyrics: {track.lyrics!r}")
            for n in range(1, track.takes + 1):
                dst = plan.take_path(track, n)
                state = "SKIP" if (dst.exists() and not force) else "RUN"
                log(f"    take-{n}: {state} {dst}")
    if "qa" in stages:
        log(f"qa: {len(plan.tracks)} tracks -> report.json/report.md per track + summary.json")
        for track in plan.tracks:
            rep = read_report(plan.qa_report_path(track))
            state = _skip_run(plan.qa_report_path(track).exists() and not force)
            pick = rep["pick"] if rep else "(pending)"
            log(f"  [{track.index + 1}/{len(plan.tracks)}] {track.title}: {state} "
                f"in={plan.track_gen_dir(track)}/take-*.flac "
                f"report={plan.qa_report_path(track)} pick={pick}")
    if "master" in stages:
        log(f"master: {len(plan.tracks)} picks -> loudnorm {plan.target_lufs:g} LUFS / {plan.target_tp:g} dBTP")
        for track in plan.tracks:
            dst = plan.master_path(track)
            state = "SKIP" if (dst.exists() and not force) else "RUN"
            src = pick_path(plan, track)
            log(f"  [{track.index + 1}/{len(plan.tracks)}] {track.title}: {state} "
                f"{src if src else '(qa pick)'} -> {dst}")
    if "stems" in stages:
        cfg = plan.stems or {}
        model = cfg.get("model") or "htdemucs"
        enabled = bool(cfg.get("enabled"))
        log(f"stems: {'RUN' if enabled else 'DISABLED'} demucs model={model} "
            f"(opt-in; add 'stems: {{enabled: true}}' to the plan)")
        for track in plan.tracks:
            out = plan.stems_dir(track)
            state = _skip_run(enabled and (out / "job-result.json").exists() and not force)
            log(f"  [{track.index + 1}/{len(plan.tracks)}] {track.title}: {state} "
                f"in-relpath={plan.track_gen_rel(track)} out-relpath={plan.stems_rel(track)}")
    if "mix" in stages:
        log(f"mix: cue={plan.cue_path()}")
        log(f"  wav: {_skip_run(plan.mix_wav().exists() and not force)} {plan.mix_wav()}")
        log(f"  mp3: {_skip_run(plan.mix_mp3().exists() and not force)} {plan.mix_mp3()} (320k)")
        log(f"  tracklist: {plan.tracklist_path()}")
        log(f"  chapters:  {plan.chapters_path()}")
        log(f"  duration:  {fmt_hms(cuegen.total_duration(plan))}")
        for s in cuegen.timeline(plan):
            log(f"    {fmt_hms(s.start_s)}  {s.track.title}  "
                f"bars {s.out_bar0}..{s.out_bar0 + s.bars}  "
                f"{s.overlap_in} bar blend{' + ramp' if s.track.ramp and s.overlap_in else ''}")
    log("")
    log("(dry run: nothing generated, rendered or submitted)")


# ------------------------------------------------------------------ driver
def run(plan: Plan, stages: list[str], backend_name: str | None = None,
        force: bool = False, dry_run_mode: bool = False, log=print) -> dict:
    """Run the selected stages in canonical order and return a small report."""
    stages = [s for s in STAGES if s in stages]
    name = backend_name or plan.backend or "acestep"
    report: dict = {"mix": plan.slug, "backend": name, "stages": stages}
    if dry_run_mode:
        dry_run(plan, stages, name, force, log)
        report["dry_run"] = True
        return report
    log(f"mixgen: {plan.title} ({plan.slug}) root={plan.root}")
    if "gen" in stages:
        log("gen:")
        report["gen"] = run_gen(plan, name, force, log)
    if "qa" in stages:
        log("qa:")
        report["qa"] = run_qa(plan, force, log)
    if "master" in stages:
        log("master:")
        report["master"] = run_master(plan, force, log)
    if "stems" in stages:
        log("stems:")
        report["stems"] = run_stems(plan, force, log)
    if "mix" in stages:
        log("mix:")
        report["mix"] = run_mix(plan, force, log)
    return report

