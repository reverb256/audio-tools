"""ACE-Step v1.5 XL backend over the ``gpu-music-job`` wrapper.

Generation is submitted as a GPU job (CPU-job style wrapper, k8s ``music``
namespace, CronJob->Job) and its output lands on the music-library NFS at
``<root>/gen/<mix-slug>/<track-slug>/take-N``. We submit, then poll that path;
there is no HTTP endpoint involved.

The field -> flag mapping lives in :mod:`mixgen.backends.acestep_flags` and is
the only place that needs updating when Infra publishes the real submit flags.

Configuration (plan ``acestep:`` block, environment wins):

===================  =============================  ===========================
plan key             env var                        default
===================  =============================  ===========================
wrapper              MIXGEN_ACESTEP_WRAPPER         gpu-music-job
command              MIXGEN_ACESTEP_COMMAND         (unset; full argv template)
poll_interval        MIXGEN_ACESTEP_POLL_INTERVAL   30 (seconds)
timeout              MIXGEN_ACESTEP_TIMEOUT         (unset = poll forever)
extra_args           —                              []
token_env            —                              MIXGEN_ACESTEP_TOKEN
===================  =============================  ===========================

Secrets are never part of the plan or argv: ``token_env`` names the environment
variable the wrapper itself reads.
"""
from __future__ import annotations

import os
import shlex
import shutil
import subprocess
import time
from dataclasses import dataclass, field
from pathlib import Path

from mixgen.backends.acestep_flags import build_argv
from mixgen.backends.base import GenBackend, GenBackendError
from mixgen.plan import Plan, Track


@dataclass
class AceStepConfig:
    wrapper: str = "gpu-music-job"
    command: str | None = None
    poll_interval: float = 30.0
    timeout: float | None = None
    extra_args: list[str] = field(default_factory=list)
    token_env: str = "MIXGEN_ACESTEP_TOKEN"

    @classmethod
    def from_plan(cls, plan: Plan) -> "AceStepConfig":
        cfg = dict(plan.acestep or {})
        wrapper = os.environ.get("MIXGEN_ACESTEP_WRAPPER") or cfg.get("wrapper") or "gpu-music-job"
        command = os.environ.get("MIXGEN_ACESTEP_COMMAND") or cfg.get("command")
        poll = os.environ.get("MIXGEN_ACESTEP_POLL_INTERVAL") or cfg.get("poll_interval") or 30.0
        timeout_raw = os.environ.get("MIXGEN_ACESTEP_TIMEOUT")
        if timeout_raw in (None, ""):
            timeout_raw = cfg.get("timeout")
        timeout = float(timeout_raw) if timeout_raw not in (None, "", 0) else None
        return cls(
            wrapper=str(wrapper),
            command=str(command) if command else None,
            poll_interval=float(poll),
            timeout=timeout,
            extra_args=[str(a) for a in (cfg.get("extra_args") or [])],
            token_env=str(cfg.get("token_env") or "MIXGEN_ACESTEP_TOKEN"),
        )


def job_fields(track: Track, take_idx: int) -> dict:
    """The fields submitted for one take (see acestep_flags for flag names)."""
    seed = (track.seed if track.seed is not None else 0) + take_idx
    fields = {
        "prompt": track.style,
        "duration": int(round(track.duration)),
        "bpm": f"{track.bpm:g}",
        "key": track.key,
        "seed": seed,
        "takes": 1,
    }
    if track.lyrics:
        fields["lyrics"] = track.lyrics
    else:
        fields["instrumental"] = True
    return fields


def take_dir_for(plan: Plan, track: Track) -> Path:
    return plan.track_gen_dir(track)


def find_take(take_dir: Path, take_idx: int) -> Path | None:
    """Existing output for ``take-N`` (``take-N.wav``, ``.flac``, ...), newest first."""
    hits = [p for p in take_dir.glob(f"take-{take_idx}.*")
            if p.is_file() and not p.name.endswith(".tmp") and ".tmp." not in p.name]
    return max(hits, key=lambda p: p.stat().st_mtime) if hits else None


def poll_for_take(take_dir: Path, take_idx: int, *, poll_interval: float = 30.0,
                  timeout: float | None = None, sleep=time.sleep,
                  clock=time.monotonic, log=None) -> Path | None:
    """Poll ``take_dir`` for ``take-N.*`` until it appears, timeout, or forever.

    ``sleep``/``clock`` are injectable so tests never wait for real time.
    """
    start = clock()
    while True:
        hit = find_take(take_dir, take_idx)
        if hit is not None:
            return hit
        if timeout is not None and clock() - start >= timeout:
            return None
        if log:
            log(f"    waiting for {take_dir.name}/take-{take_idx}* ...")
        sleep(poll_interval)


class AceStepBackend(GenBackend):
    name = "acestep"

    def __init__(self, plan: Plan, config: AceStepConfig | None = None, *,
                 submit=None, sleep=time.sleep, clock=time.monotonic, log=None,
                 which=shutil.which):
        self.plan = plan
        self.config = config or AceStepConfig.from_plan(plan)
        self._submit = submit or self._run_submit
        self._sleep = sleep
        self._clock = clock
        self._log = log
        self._which = which

    # -- argv ---------------------------------------------------------
    def argv_for(self, track: Track, take_idx: int, out_path: Path) -> list[str]:
        fields = job_fields(track, take_idx)
        if self.config.command:
            values = dict(fields, track=track.slug, mix=self.plan.slug,
                          out=str(out_path), take=take_idx, take_dir=str(take_dir_for(self.plan, track)))
            try:
                parts = shlex.split(self.config.command)
                return [p.format_map(values) for p in parts]
            except KeyError as exc:
                raise GenBackendError(
                    f"acestep.command template references unknown placeholder {exc}; "
                    f"available: {sorted(values)}") from exc
        return build_argv(fields, wrapper=self.config.wrapper, extra=self.config.extra_args)

    def _run_submit(self, argv: list[str]) -> None:
        if self._which(argv[0]) is None:
            raise GenBackendError(
                f"acestep backend is not configured: wrapper {argv[0]!r} was not found on PATH.\n"
                "  Set acestep.wrapper / MIXGEN_ACESTEP_WRAPPER to the gpu-music-job command,\n"
                "  or provide a full submit template via acestep.command / MIXGEN_ACESTEP_COMMAND\n"
                "  (placeholders: {prompt} {lyrics} {instrumental} {duration} {bpm} {key} "
                "{seed} {takes} {out}).\n"
                "  See README.md -> mixgen -> backends.")
        try:
            subprocess.run(argv, check=True, capture_output=True, text=True)
        except subprocess.CalledProcessError as exc:
            detail = (exc.stderr or exc.stdout or "").strip()
            raise GenBackendError(f"acestep submit failed ({exc.returncode}): {detail}") from exc
        except OSError as exc:
            raise GenBackendError(f"acestep submit could not start {argv[0]!r}: {exc}") from exc

    # -- generate -----------------------------------------------------
    def generate(self, track: Track, take_idx: int, out_path: Path) -> Path:
        take_dir = take_dir_for(self.plan, track)
        take_dir.mkdir(parents=True, exist_ok=True)
        argv = self.argv_for(track, take_idx, out_path)
        if self._log:
            self._log(f"    submit: {' '.join(shlex.quote(a) for a in argv[:6])}"
                      + (" ..." if len(argv) > 6 else ""))
        self._submit(argv)
        produced = poll_for_take(
            take_dir, take_idx, poll_interval=self.config.poll_interval,
            timeout=self.config.timeout, sleep=self._sleep, clock=self._clock,
            log=self._log)
        if produced is None:
            raise GenBackendError(
                f"acestep: timed out after {self.config.timeout:g}s waiting for "
                f"{take_dir}/take-{take_idx}.* (raise/remove acestep.timeout to poll forever)")
        if produced.resolve() != Path(out_path).resolve():
            Path(out_path).parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(produced, out_path)
        return Path(out_path)
