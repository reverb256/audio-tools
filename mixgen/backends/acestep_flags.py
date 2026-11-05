"""Map mixgen's generation fields onto the ``gpu-music-job`` submit CLI.

THIS IS THE ONLY FILE THAT KNOWS FLAG NAMES.

TODO(infra): the real submit flags for the ``gpu-music-job`` wrapper (CPU-job
style, k8s ``music`` namespace, CronJob->Job) are not confirmed yet. When they
are, edit ``FIELD_FLAGS`` and optionally ``FIELD_ORDER`` below and nothing else;
``build_argv`` and the acestep backend both go through this module.

Fields submitted per job (from Infra): prompt, lyrics | instrumental, duration,
bpm, key, seed, takes. Output files are NOT a flag: generation lands on the
music-library NFS at ``<root>/gen/<mix>/<track>/take-N`` and mixgen polls for
them.
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
    "takes": ["--takes"],
}

# Deterministic ordering so argv diffs are stable in tests and logs.
FIELD_ORDER: tuple[str, ...] = (
    "prompt", "lyrics", "instrumental", "duration", "bpm", "key", "seed", "takes",
)

# Fields that are presence-only switches (no value argument).
BOOLEAN_FIELDS: frozenset[str] = frozenset({"instrumental"})


def build_argv(fields: dict, wrapper: str = "gpu-music-job", extra: list[str] | None = None) -> list[str]:
    """Render a submit argv from the field dict.

    ``fields`` uses mixgen field names (``prompt``, ``bpm``, ...). Unknown names
    raise so a typo cannot silently drop a field from the job.
    """
    unknown = set(fields) - set(FIELD_FLAGS)
    if unknown:
        raise ValueError(f"unknown acestep job fields {sorted(unknown)}; update mixgen/backends/acestep_flags.py")
    argv = [wrapper]
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
