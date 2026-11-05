"""Mastering: reuse djmix's two-pass loudnorm chain (-9 LUFS / -1 dBTP)."""
from __future__ import annotations

import os
from pathlib import Path


def master_file(src: Path, dst: Path, *, sample_rate: int = 48000,
                target_lufs: float = -9.0, target_tp: float = -1.0) -> dict:
    """Loudness-normalise one take into ``dst`` atomically (tmp + rename)."""
    from djmix.render import loudnorm_2pass

    dst = Path(dst)
    dst.parent.mkdir(parents=True, exist_ok=True)
    tmp = dst.with_name(dst.stem + ".mastering.tmp.wav")
    try:
        report = loudnorm_2pass(Path(src), tmp, sample_rate, target_lufs, target_tp)
        os.replace(tmp, dst)
    finally:
        for leftover in (tmp, tmp.with_suffix(".tp.wav")):
            try:
                leftover.unlink()
            except FileNotFoundError:
                pass
    return report
