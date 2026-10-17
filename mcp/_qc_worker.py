#!/usr/bin/env python3
"""QC measurement worker for the audio MCP server.

Runs under envs/cpu (python 3.12: numpy, soundfile, pyloudnorm).
argv: FILE  ->  one JSON object on stdout. Read-only: never writes anything.

Measures:
  - pyloudnorm integrated loudness on the file as stored (documented usage:
    (samples, channels) array passed whole) -> "pyloudnorm_lufs".
  - pyloudnorm integrated loudness of the mono downmix -> used by the phase-1
    proof in audio-tools/README.md (reads 1-4 LU lower on wide stereo material).
  - sample peak (linear + dBFS).

The ffmpeg ebur128 cross-check (I / LRA / true peak) is done by the server.
"""
from __future__ import annotations

import json
import math
import sys


def main() -> int:
    path = sys.argv[1]

    import numpy as np
    import pyloudnorm as pyln
    import soundfile as sf

    data, sr = sf.read(path, always_2d=False)
    meter = pyln.Meter(sr)
    notes: list[str] = []

    out: dict = {
        "sample_rate_hz": int(sr),
        "channels": int(data.shape[1]) if data.ndim > 1 else 1,
        "frames": int(data.shape[0]),
        "duration_s": round(float(data.shape[0]) / float(sr), 3) if sr else None,
    }

    def integrated(arr) -> float | None:
        try:
            v = float(meter.integrated_loudness(arr))
            if not math.isfinite(v):
                return None
            return round(v, 3)
        except Exception as exc:  # pyloudnorm raises on very short (<0.4 s) audio
            notes.append(f"pyloudnorm: {type(exc).__name__}: {exc}")
            return None

    out["pyloudnorm_lufs"] = integrated(data)
    if data.ndim > 1:
        out["pyloudnorm_lufs_mono_downmix"] = integrated(data.mean(axis=1))
    else:
        out["pyloudnorm_lufs_mono_downmix"] = None

    if data.size:
        peak = float(np.max(np.abs(data)))
    else:
        peak = 0.0
    out["sample_peak_linear"] = peak
    out["sample_peak_dbfs"] = round(20.0 * math.log10(peak), 3) if peak > 0 else None

    if notes:
        out["notes"] = notes
    print(json.dumps(out))
    return 0


if __name__ == "__main__":
    sys.exit(main())
