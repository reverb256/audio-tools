"""ACE-Step v1.5 XL backend over the locked ``gpu-music-job`` wrapper.

One job per track: ``gpu-music-job acestep --takes N ...`` produces all takes
into a flat directory on the PVC, ``/music/gen/<mix>/<track>/`` (host
``/data/shared/music/gen/<mix>/<track>/``), containing FLAC audio plus the
wrapper's ``job-request.json`` and ``job-result.json``. We submit once, wait for
``job-result.json``, then copy the takes listed in ``audios[].path`` to mixgen's
take paths. There are no ``take-N`` folders to poll.

A job is done when ``job-result.json`` exists; while waiting we may ask
``gpu-music-job status <job-name>`` for progress and surface non-zero exit
codes (the wrapper uses 0/2/3). There is no wall-clock kill and no token/auth.

Stems (``gpu-music-job demucs``) are a separate opt-in step; see
:meth:`AceStepBackend.demucs`.

The field -> flag mapping lives in :mod:`mixgen.backends.acestep_flags` and is
the only place that knows flag names.

Configuration (plan ``acestep:`` block, environment wins):

===================  =============================  ===========================
plan key             env var                        default
===================  =============================  ===========================
wrapper              MIXGEN_ACESTEP_WRAPPER         gpu-music-job
command              MIXGEN_ACESTEP_COMMAND         (unset; full argv template)
poll_interval        MIXGEN_ACESTEP_POLL_INTERVAL   30 (seconds)
dry_run              MIXGEN_ACESTEP_DRY_RUN         false
extra_args           —                              []
===================  =============================  ===========================
"""
from __future__ import annotations

import json
import os
import re
import shlex
import shutil
import subprocess
import time
from dataclasses import dataclass, field
from pathlib import Path

from mixgen.backends.acestep_flags import build_argv, build_demucs_argv
from mixgen.backends.base import GenBackend, GenBackendError
from mixgen.plan import Plan, Track

_JOB_NAME_KEYS = ("job", "job_name", "name", "jobName")
_JOB_NAME_RE = re.compile(r"\bjob(?:[ _-]?name)?\s*[:=]?\s*([A-Za-z0-9][A-Za-z0-9._-]*)\b",
                          re.IGNORECASE)
# PVC mount and the plan root are the same library; job results usually report
# host paths like /data/shared/music/gen/... .
_PVC_MUSIC_ROOTS = ("/data/shared/music", "/music")


@dataclass
class AceStepConfig:
    wrapper: str = "gpu-music-job"
    command: str | None = None
    poll_interval: float = 30.0
    dry_run: bool = False
    extra_args: list[str] = field(default_factory=list)

    @classmethod
    def from_plan(cls, plan: Plan) -> "AceStepConfig":
        cfg = dict(plan.acestep or {})
        wrapper = os.environ.get("MIXGEN_ACESTEP_WRAPPER") or cfg.get("wrapper") or "gpu-music-job"
        command = os.environ.get("MIXGEN_ACESTEP_COMMAND") or cfg.get("command")
        poll = os.environ.get("MIXGEN_ACESTEP_POLL_INTERVAL") or cfg.get("poll_interval") or 30.0
        dry_env = os.environ.get("MIXGEN_ACESTEP_DRY_RUN")
        dry_run = (str(dry_env).lower() in ("1", "true", "yes", "on")
                   if dry_env not in (None, "") else bool(cfg.get("dry_run", False)))
        return cls(
            wrapper=str(wrapper),
            command=str(command) if command else None,
            poll_interval=float(poll),
            dry_run=dry_run,
            extra_args=[str(a) for a in (cfg.get("extra_args") or [])],
        )


def out_relpath_for(plan: Plan, track: Track) -> str:
    """``<mix>/<track>``: the job's flat output directory under ``/music/gen``."""
    return f"{plan.slug}/{track.slug}"


def job_fields(track: Track, out_relpath: str, *, dry_run: bool = False) -> dict:
    """The fields submitted for one track's batch job (see acestep_flags)."""
    fields: dict = {
        "prompt": track.style,
        "duration": int(round(track.duration)),
        "bpm": f"{track.bpm:g}",
        "key": track.key,
        "takes": int(track.takes),
        "out_relpath": out_relpath,
    }
    if track.seed is not None:
        fields["seed"] = int(track.seed)
    if track.lyrics:
        fields["lyrics"] = track.lyrics
    else:
        fields["instrumental"] = True
    if dry_run:
        fields["dry_run"] = True
    return fields


def parse_job_name(stdout: str | None) -> str | None:
    """Pull the job name the wrapper echoes on create (JSON or text)."""
    text = (stdout or "").strip()
    if not text:
        return None
    try:
        data = json.loads(text)
    except ValueError:
        data = None
    if isinstance(data, dict):
        for key in _JOB_NAME_KEYS:
            if data.get(key):
                return str(data[key])
    m = _JOB_NAME_RE.search(text)
    return m.group(1) if m else None


def read_result(path: Path) -> dict | None:
    """Parsed ``job-result.json``, or None while it is missing/being written."""
    try:
        data = json.loads(Path(path).read_text())
    except (OSError, ValueError):
        return None
    return data if isinstance(data, dict) else None


def audio_paths(result: dict) -> list[str]:
    """Audio paths from ``job-result.json -> audios[].path``."""
    audios = result.get("audios") or []
    paths = []
    for entry in audios:
        if isinstance(entry, dict):
            raw = entry.get("path") or entry.get("file")
        else:
            raw = entry
        if raw:
            paths.append(str(raw))
    return paths


def wait_for_result(result_path: Path, *, job_name: str | None = None,
                    status=None, poll_interval: float = 30.0,
                    sleep=time.sleep, log=None) -> dict:
    """Wait until ``job-result.json`` exists and parses, then return it.

    A job is done exactly when the result file exists — there is no wall-clock
    timeout. ``status`` (``job_name -> (returncode, output)``) is optional
    progress/error reporting; a non-zero status exit aborts with a clear error.
    """
    result_path = Path(result_path)
    while True:
        result = read_result(result_path)
        if result is not None:
            return result
        if status is not None and job_name:
            code, output = status(job_name)
            if code not in (0, None):
                detail = (output or "").strip()
                raise GenBackendError(
                    f"gpu-music-job status {job_name} failed ({code}): {detail or 'no output'}")
        if log:
            log(f"    waiting for {result_path.parent.name}/job-result.json"
                + (f" (job {job_name})" if job_name else "") + " ...")
        sleep(poll_interval)


class AceStepBackend(GenBackend):
    name = "acestep"

    def __init__(self, plan: Plan, config: AceStepConfig | None = None, *,
                 submit=None, status=None, sleep=time.sleep, clock=time.monotonic,
                 log=None, which=shutil.which):
        self.plan = plan
        self.config = config or AceStepConfig.from_plan(plan)
        self._submit = submit or self._run_submit
        # A custom command template may not be gpu-music-job, so only probe
        # status by default for the regular wrapper.
        self._status = status or (None if self.config.command else self._run_status)
        self._sleep = sleep
        self._clock = clock
        self._log = log
        self._which = which
        self._results: dict[str, dict] = {}
        self._job_names: dict[str, str | None] = {}

    # -- argv ---------------------------------------------------------
    def argv_for(self, track: Track, *, out_relpath: str | None = None,
                 dry_run: bool | None = None) -> list[str]:
        rel = out_relpath or out_relpath_for(self.plan, track)
        fields = job_fields(track, rel, dry_run=self.config.dry_run if dry_run is None else dry_run)
        if self.config.command:
            take_dir = self.plan.track_gen_dir(track)
            values = dict(fields, track=track.slug, mix=self.plan.slug,
                          out_relpath=rel, take_dir=str(take_dir),
                          gen_dir=str(self.plan.gen_dir), out=str(take_dir))
            try:
                parts = shlex.split(self.config.command)
                return [p.format_map(values) for p in parts]
            except KeyError as exc:
                raise GenBackendError(
                    f"acestep.command template references unknown placeholder {exc}; "
                    f"available: {sorted(values)}") from exc
        return build_argv(fields, wrapper=self.config.wrapper, extra=self.config.extra_args)

    def demucs_argv(self, in_relpath: str, out_relpath: str, model: str | None = None) -> list[str]:
        return build_demucs_argv(in_relpath, out_relpath, model=model,
                                 wrapper=self.config.wrapper, extra=self.config.extra_args)

    def _check_wrapper(self, argv: list[str]) -> None:
        if self._which(argv[0]) is None:
            raise GenBackendError(
                f"acestep backend is not configured: wrapper {argv[0]!r} was not found on PATH.\n"
                "  Set acestep.wrapper / MIXGEN_ACESTEP_WRAPPER to the gpu-music-job command,\n"
                "  or provide a full submit template via acestep.command / MIXGEN_ACESTEP_COMMAND\n"
                "  (placeholders: {prompt} {lyrics} {instrumental} {duration} {bpm} {key} "
                "{seed} {takes} {out_relpath}).\n"
                "  See README.md -> mixgen -> backends.")

    def _run(self, argv: list[str], what: str) -> str:
        self._check_wrapper(argv)
        try:
            proc = subprocess.run(argv, check=True, capture_output=True, text=True)
        except subprocess.CalledProcessError as exc:
            detail = (exc.stderr or exc.stdout or "").strip()
            hint = ""
            if exc.returncode == 2:
                hint = " (usage error)"
            elif exc.returncode == 3:
                hint = " (job failed)"
            raise GenBackendError(f"acestep {what} failed ({exc.returncode}{hint}): {detail}") from exc
        except OSError as exc:
            raise GenBackendError(f"acestep {what} could not start {argv[0]!r}: {exc}") from exc
        return proc.stdout or ""

    def _run_submit(self, argv: list[str]) -> str:
        return self._run(argv, "submit")

    def _run_status(self, job_name: str) -> tuple[int, str]:
        argv = [self.config.wrapper, "status", job_name]
        try:
            proc = subprocess.run(argv, capture_output=True, text=True)
        except OSError as exc:
            return 127, f"could not start {argv[0]!r}: {exc}"
        return proc.returncode, (proc.stdout or "") + (proc.stderr or "")

    # -- generate -----------------------------------------------------
    def _result_path(self, track: Track) -> Path:
        return self.plan.track_gen_dir(track) / "job-result.json"

    def _ensure_result(self, track: Track, *, force: bool = False) -> dict:
        key = track.slug
        if key in self._results and not force:
            return self._results[key]
        result_path = self._result_path(track)
        if not force:
            existing = read_result(result_path)
            if existing is not None and len(audio_paths(existing)) >= track.takes:
                self._results[key] = existing
                return existing
        # A stale result must not satisfy a fresh job: remove it before submit.
        if force or read_result(result_path) is not None:
            try:
                result_path.unlink()
            except FileNotFoundError:
                pass
        result_path.parent.mkdir(parents=True, exist_ok=True)
        rel = out_relpath_for(self.plan, track)
        argv = self.argv_for(track, out_relpath=rel)
        if self._log:
            self._log(f"    submit: {' '.join(shlex.quote(a) for a in argv[:7])}"
                      + (" ..." if len(argv) > 7 else ""))
        stdout = self._submit(argv) or ""
        job_name = parse_job_name(stdout)
        self._job_names[key] = job_name
        if job_name and self._log:
            self._log(f"    job: {job_name} (out-relpath {rel})")
        result = wait_for_result(
            result_path, job_name=job_name, status=self._status,
            poll_interval=self.config.poll_interval, sleep=self._sleep, log=self._log)
        self._results[key] = result
        return result

    def _local_audio(self, raw: str, result_path: Path) -> Path:
        p = Path(raw)
        if p.exists():
            return p
        for base in _PVC_MUSIC_ROOTS:
            try:
                rel = p.relative_to(base)
            except ValueError:
                continue
            cand = self.plan.root / rel
            if cand.exists():
                return cand
        cand = result_path.parent / p.name
        return cand if cand.exists() else p

    def generate_track(self, track: Track, targets: list[tuple[int, Path]],
                       *, force: bool = False) -> list[Path]:
        result = self._ensure_result(track, force=force)
        paths = audio_paths(result)
        if len(paths) < track.takes:
            raise GenBackendError(
                f"acestep: job-result.json for '{track.title}' lists {len(paths)} audio(s), "
                f"expected {track.takes} (--takes {track.takes})")
        result_path = self._result_path(track)
        out = []
        for take_idx, out_path in targets:
            if not 1 <= take_idx <= len(paths):
                raise GenBackendError(
                    f"acestep: take {take_idx} not in job-result.json "
                    f"({len(paths)} audio(s) listed)")
            src = self._local_audio(paths[take_idx - 1], result_path)
            if not src.exists():
                raise GenBackendError(
                    f"acestep: job-result.json lists {paths[take_idx - 1]!r} "
                    f"(take {take_idx}) but that file does not exist")
            Path(out_path).parent.mkdir(parents=True, exist_ok=True)
            if src.resolve() != Path(out_path).resolve():
                shutil.copyfile(src, out_path)
            out.append(Path(out_path))
        return out

    def generate(self, track: Track, take_idx: int, out_path: Path) -> Path:
        return self.generate_track(track, [(take_idx, out_path)])[0]

    # -- optional stems -----------------------------------------------
    def demucs(self, in_relpath: str, out_relpath: str, *, model: str | None = None,
               force: bool = False) -> dict:
        """Submit the optional ``gpu-music-job demucs`` stems step and wait for it."""
        result_path = self.plan.root / "gen" / out_relpath / "job-result.json"
        if not force:
            existing = read_result(result_path)
            if existing is not None:
                return existing
        result_path.parent.mkdir(parents=True, exist_ok=True)
        if force:
            try:
                result_path.unlink()
            except FileNotFoundError:
                pass
        argv = self.demucs_argv(in_relpath, out_relpath, model)
        if self._log:
            self._log(f"    submit: {' '.join(shlex.quote(a) for a in argv)}")
        stdout = self._submit(argv) or ""
        job_name = parse_job_name(stdout)
        if job_name and self._log:
            self._log(f"    job: {job_name} (out-relpath {out_relpath})")
        return wait_for_result(
            result_path, job_name=job_name, status=self._status,
            poll_interval=self.config.poll_interval, sleep=self._sleep, log=self._log)
