"""Map mixgen's generation fields onto the locked ``gpu-music-job`` CLI.

THIS IS THE ONLY FILE THAT KNOWS FLAG NAMES.

Submit argv (Infra's locked gpu-music-job contract)::

    gpu-music-job acestep --prompt TEXT [--lyrics TEXT|@file] [--instrumental]
        [--duration SEC] [--bpm N] [--key TEXT] [--seed N] [--takes N]
        [--out-relpath REL] [--dry-run]

``--takes N`` is one Job that produces all N takes (a batch), not one job per
take. Output is a flat directory ``/music/gen/<out-relpath>/`` on the PVC
(host ``/data/shared/music/gen/<out-relpath>/``): FLAC by default, plus the
wrapper's ``job-request.json`` and ``job-result.json``. The audio paths come
from ``job-result.json -> audios[].path``; there are no ``take-N`` folders.

Stems are a separate, optional step::

    gpu-music-job demucs --in-relpath REL --out-relpath REL [--model htdemucs]

``gpu-music-job status <job-name>`` reports a submitted job (the name is echoed
on create); a job is done when ``job-result.json`` exists. There is no token
or auth: the wrapper runs on the cluster with the target's own credentials.

Exit codes are the wrapper's 0/2/3; mixgen surfaces any non-zero as an error.
"""
from __future__ import annotations

FIELD_FLAGS: dict[str, list[str]] = {
    "prompt": ["--prompt"],
    "lyrics": ["--lyrics"],
    "instrumental": ["--instrumental"],
    "duration": ["--duration"],   # seconds
    "bpm": ["--bpm"],
    "key": ["--key"],             # Camelot, e.g. 8A
    "seed": ["--seed"],
    "takes": ["--takes"],         # one job, N takes (batch)
    "out_relpath": ["--out-relpath"],
    "dry_run": ["--dry-run"],
}

# Deterministic ordering so argv diffs are stable in tests and logs.
FIELD_ORDER: tuple[str, ...] = (
    "prompt", "lyrics", "instrumental", "duration", "bpm", "key", "seed",
    "takes", "out_relpath", "dry_run",
)

# Fields that are presence-only switches (no value argument).
BOOLEAN_FIELDS: frozenset[str] = frozenset({"instrumental", "dry_run"})

# The wrapper's subcommand for ACE-Step generation.
SUBCOMMAND = "acestep"

# Demucs is a separate optional step, not part of generation.
DEMUCS_SUBCOMMAND = "demucs"
DEMUCS_FIELD_FLAGS: dict[str, list[str]] = {
    "in_relpath": ["--in-relpath"],
    "out_relpath": ["--out-relpath"],
    "model": ["--model"],
}
DEMUCS_FIELD_ORDER: tuple[str, ...] = ("in_relpath", "out_relpath", "model")


def build_argv(fields: dict, wrapper: str = "gpu-music-job", extra: list[str] | None = None) -> list[str]:
    """Render a submit argv from the field dict.

    ``fields`` uses mixgen field names (``prompt``, ``bpm``, ...). Unknown names
    raise so a typo cannot silently drop a field from the job.
    """
    unknown = set(fields) - set(FIELD_FLAGS)
    if unknown:
        raise ValueError(f"unknown acestep job fields {sorted(unknown)}; update mixgen/backends/acestep_flags.py")
    argv = [wrapper, SUBCOMMAND]
    for name in FIELD_ORDER:
        if name not in fields:
            continue
        value = fields[name]
        if value is None:
            continue
        if name in BOOLEAN_FIELDS:
            if value:
                argv.extend(FIELD_FLAGS[name])
            continue
        argv.extend(FIELD_FLAGS[name])
        argv.append(str(value))
    if extra:
        argv.extend(str(a) for a in extra)
    return argv


def build_demucs_argv(in_relpath: str, out_relpath: str, *, model: str | None = None,
                      wrapper: str = "gpu-music-job", extra: list[str] | None = None) -> list[str]:
    """Render the optional stems argv (``gpu-music-job demucs ...``)."""
    fields: dict = {"in_relpath": in_relpath, "out_relpath": out_relpath}
    if model:
        fields["model"] = model
    argv = [wrapper, DEMUCS_SUBCOMMAND]
    for name in DEMUCS_FIELD_ORDER:
        if name in fields and fields[name] is not None:
            argv.extend(DEMUCS_FIELD_FLAGS[name])
            argv.append(str(fields[name]))
    if extra:
        argv.extend(str(a) for a in extra)
    return argv
