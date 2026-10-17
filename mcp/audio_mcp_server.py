#!/usr/bin/env python3
"""audio — the audio lane's MCP facade (stdio, FastMCP).

Thin in-house facade over the pinned audio-tools envs, per the deep-research
design (``~/Documents/research/linux-audio-agent-tooling.md``, section 5
"a small stdio facade") and the ``audio-processing-toolchain`` skill.

Rules enforced in this file:
  * every path argument must realpath-resolve under ALLOWLIST_ROOTS
    (/home/j_kro/Music, /home/j_kro/Documents, the audio-tools repo itself);
  * every subprocess call is an argv array — never shell=True, never a shell string;
  * recipes are allowlisted, versioned data (``recipes/*.json``), not free-form shell;
  * renders always write NEW outputs — the input is only ever read;
  * tools return paths / hashes / compact JSON — never audio arrays.

Heavy lifting lives in the pinned envs:
  envs/cpu   — pyloudnorm / soundfile QC worker (``_qc_worker.py``)
  envs/ml    — torch-CPU demucs runner (``_demucs_runner.py``)
  system     — /usr/bin/ffmpeg (n9.0.2: loudnorm, ebur128, silenceremove),
               /usr/bin/sox (SoX_ng), /usr/bin/ffprobe.
Note: PATH may shadow system ffmpeg with a Hermes-private build on this host —
that is why the system binaries are invoked by absolute path.

Tools: audio_qc, dsp_render, stem_submit, stem_status, release_validate.
"""
from __future__ import annotations

import hashlib
import json
import math
import os
import re
import subprocess
import sys
import time
import uuid
from pathlib import Path

from mcp.server.fastmcp import FastMCP

ROOT = Path(__file__).resolve().parent.parent          # .../Projects/audio-tools
MCP_DIR = Path(__file__).resolve().parent
CPU_PY = ROOT / "envs" / "cpu" / "bin" / "python"
ML_PY = ROOT / "envs" / "ml" / "bin" / "python"
QC_WORKER = MCP_DIR / "_qc_worker.py"
DEMUCS_RUNNER = MCP_DIR / "_demucs_runner.py"
RECIPES_DIR = ROOT / "recipes"
JOBS_DIR = ROOT / "jobs"

FFMPEG = "/usr/bin/ffmpeg"
FFPROBE = "/usr/bin/ffprobe"
SOX = "/usr/bin/sox"

ALLOWLIST_ROOTS = [Path("/home/j_kro/Music"), Path("/home/j_kro/Documents"), ROOT]
HEADROOM_DBTP = -1.0
QC_TIMEOUT = 900
RENDER_TIMEOUT = 1800

STEM_MODELS = ("htdemucs", "htdemucs_ft", "htdemucs_6s")
TWO_STEMS_CHOICES = ("vocals", "drums", "bass", "other")

RECIPE_NAMES = ("loudnorm-2pass", "trim-silence", "normalize-peak", "fade", "highpass", "convert")

GATING_NOTE = (
    "pyloudnorm(stereo) and ffmpeg ebur128 agree within ~0.2 LU when both see the "
    "same channel layout; the mono-downmix pyloudnorm variant (phase-1 proof "
    "convention) can read 1-4 LU lower. Read both, never one alone."
)

PROFILES = {
    "streaming": {
        "tp_max_dbtp": -1.0,          # Spotify-recommended streaming true-peak headroom
        "target_lufs": -14.0,         # PLAYBACK target only (warn band, never auto-compress)
        "lufs_band_lu": 1.0,
        "head_silence_max_s": 2.0,
        "tail_silence_max_s": 3.0,
        "min_duration_s": 30.0,
        "min_sample_rate_hz": 44100,
        "formats": ("flac", "wav"),
    }
}

mcp = FastMCP("audio")


# ---------------------------------------------------------------- helpers ---

def _now_iso() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%S%z")


def _err(message, **extra) -> str:
    return json.dumps({"ok": False, "error": str(message), **extra})


def _tail(text: str | None, n: int = 800) -> str:
    text = text or ""
    return text[-n:]


def _resolve_safe(path_arg: str) -> Path:
    """realpath-resolve a path argument and require it under ALLOWLIST_ROOTS."""
    rp = Path(os.path.realpath(str(path_arg)))
    for root in ALLOWLIST_ROOTS:
        if rp == root or root in rp.parents:
            return rp
    raise PermissionError(
        f"path {str(path_arg)!r} resolves to {str(rp)!r}, outside the allowlist roots "
        f"{[str(r) for r in ALLOWLIST_ROOTS]}"
    )


def _clean_env() -> dict:
    env = dict(os.environ)
    env.pop("PYTHONPATH", None)   # a caller's site-packages must not shadow the envs
    env.pop("PYTHONHOME", None)
    return env


def _run(argv: list[str], timeout: int = QC_TIMEOUT) -> subprocess.CompletedProcess:
    return subprocess.run(argv, capture_output=True, text=True, timeout=timeout, env=_clean_env())


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _write_json(path: Path, obj: dict) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(obj, indent=2, default=str) + "\n")
    os.replace(tmp, path)


def _tool_versions() -> dict:
    def first_line(cmd: list[str]) -> str:
        try:
            r = subprocess.run(cmd, capture_output=True, text=True, timeout=30, env=_clean_env())
            text = (r.stdout or "") + (r.stderr or "")
            return text.strip().splitlines()[0] if text.strip() else "unknown"
        except Exception as exc:  # pragma: no cover
            return f"unavailable: {exc}"
    return {"ffmpeg": first_line([FFMPEG, "-version"]), "sox": first_line([SOX, "--version"])}


def _fmt_num(x: float) -> str:
    s = f"{float(x):.6f}".rstrip("0").rstrip(".")
    return s if s else "0"


def _ffprobe_json(path: Path) -> dict:
    r = _run([FFPROBE, "-v", "error", "-show_format", "-show_streams", "-of", "json", str(path)])
    if r.returncode != 0:
        raise RuntimeError(f"ffprobe failed on {path}: {_tail(r.stderr)}")
    return json.loads(r.stdout)


def _to_float(x):
    try:
        return float(x)
    except (TypeError, ValueError):
        return None


def _to_int(x):
    try:
        return int(x)
    except (TypeError, ValueError):
        return None


def _identity(probe: dict) -> dict:
    stream = next((s for s in probe.get("streams", []) if s.get("codec_type") == "audio"), None)
    fmt = probe.get("format", {}) or {}
    depth = None
    if stream:
        for key in ("bits_per_raw_sample", "bits_per_sample"):
            v = _to_int(stream.get(key))
            if v:
                depth = v
                break
    duration = _to_float(fmt.get("duration"))
    if duration is None and stream:
        duration = _to_float(stream.get("duration"))
    return {
        "container": fmt.get("format_name"),
        "codec": (stream or {}).get("codec_name"),
        "sample_rate_hz": _to_int((stream or {}).get("sample_rate")),
        "channels": _to_int((stream or {}).get("channels")),
        "channel_layout": (stream or {}).get("channel_layout"),
        "bit_depth": depth,
        "sample_fmt": (stream or {}).get("sample_fmt"),
        "duration_s": round(duration, 3) if duration is not None else None,
        "size_bytes": _to_int(fmt.get("size")),
        "bit_rate_bps": _to_int(fmt.get("bit_rate")),
    }


def _ebur128(path: Path) -> dict:
    r = _run([FFMPEG, "-hide_banner", "-nostats", "-i", str(path),
              "-af", "ebur128=peak=true", "-f", "null", "-"])
    if r.returncode != 0:
        raise RuntimeError(f"ebur128 scan failed on {path}: {_tail(r.stderr)}")
    tail = r.stderr.split("Summary:")[-1]

    def grab(pattern: str):
        m = re.search(pattern, tail)
        return float(m.group(1)) if m else None

    return {
        "integrated_lufs": grab(r"I:\s*([+-]?\d+(?:\.\d+)?)\s*LUFS"),
        "lra_lu": grab(r"LRA:\s*([+-]?\d+(?:\.\d+)?)\s*LU"),
        "true_peak_dbtp": grab(r"Peak:\s*([+-]?\d+(?:\.\d+)?)\s*dB"),
    }


def _pyloudnorm_worker(path: Path) -> dict:
    r = _run([str(CPU_PY), str(QC_WORKER), str(path)])
    if r.returncode != 0:
        raise RuntimeError(f"pyloudnorm worker failed on {path}: {_tail(r.stderr)}")
    return json.loads(r.stdout.strip().splitlines()[-1])


def _measure(path: Path) -> dict:
    """Full QC measurement: ffprobe identity + ebur128 + pyloudnorm worker."""
    ident = _identity(_ffprobe_json(path))
    loud = _ebur128(path)
    try:
        pyq = _pyloudnorm_worker(path)
    except Exception as exc:
        pyq = {"error": f"{type(exc).__name__}: {exc}"}
    py_lufs = pyq.get("pyloudnorm_lufs")
    py_mono = pyq.get("pyloudnorm_lufs_mono_downmix")
    eb = loud.get("integrated_lufs")
    tpk = loud.get("true_peak_dbtp")
    return {
        "identity": ident,
        "loudness": {
            "pyloudnorm_lufs": py_lufs,
            "pyloudnorm_lufs_mono_downmix": py_mono,
            "ebur128_lufs": eb,
            "delta_pyloudnorm_vs_ebur128_lu": round(py_lufs - eb, 2) if (py_lufs is not None and eb is not None) else None,
            "lra_lu": loud.get("lra_lu"),
            "gating_note": GATING_NOTE,
            **({"pyloudnorm_error": pyq["error"]} if "error" in pyq else {}),
        },
        "peaks": {
            "sample_peak_dbfs": pyq.get("sample_peak_dbfs"),
            "sample_peak_linear": pyq.get("sample_peak_linear"),
            "true_peak_dbtp": tpk,
            "true_peak_ok": (tpk <= HEADROOM_DBTP) if tpk is not None else None,
            "headroom_target_dbtp": HEADROOM_DBTP,
        },
    }


def _qc_summary(m: dict) -> dict:
    return {
        "ebur128_lufs": m["loudness"]["ebur128_lufs"],
        "pyloudnorm_lufs": m["loudness"]["pyloudnorm_lufs"],
        "lra_lu": m["loudness"]["lra_lu"],
        "true_peak_dbtp": m["peaks"]["true_peak_dbtp"],
        "sample_peak_dbfs": m["peaks"]["sample_peak_dbfs"],
        "duration_s": m["identity"]["duration_s"],
    }


# ---------------------------------------------------------- recipe loading ---

def _recipe(name: str) -> dict:
    p = RECIPES_DIR / f"{name}.json"
    if not p.is_file():
        raise RuntimeError(f"recipe data file missing: {p}")
    return json.loads(p.read_text())


def _merge_params(params, name: str, allowed: set) -> dict:
    supplied = dict(params or {})
    unknown = set(supplied) - allowed
    if unknown:
        raise ValueError(f"unknown params for {name!r}: {sorted(unknown)} (allowed: {sorted(allowed)})")
    merged = dict(_recipe(name).get("defaults", {}))
    merged.update(supplied)
    return merged


def _fnum(merged: dict, key: str, lo=None, hi=None) -> float:
    v = _to_float(merged.get(key))
    if v is None or not math.isfinite(v):
        raise ValueError(f"param {key!r} must be a finite number")
    if lo is not None and v < lo:
        raise ValueError(f"param {key!r}={v} below minimum {lo}")
    if hi is not None and v > hi:
        raise ValueError(f"param {key!r}={v} above maximum {hi}")
    return v


def _ichoice(merged: dict, key: str, choices) -> int:
    v = _to_int(merged.get(key))
    if v is None or v not in choices:
        raise ValueError(f"param {key!r} must be one of {sorted(choices)}")
    return v


def _schoice(merged: dict, key: str, choices) -> str:
    v = str(merged.get(key, "")).strip().lower()
    if v not in choices:
        raise ValueError(f"param {key!r} must be one of {sorted(choices)}")
    return v


def _pick_ext(in_path: Path, merged: dict, sox_recipe: bool = False) -> str:
    raw = merged.get("out_format")
    ext = str(raw).lower().lstrip(".") if raw else in_path.suffix.lower().lstrip(".")
    ok = ("flac", "wav") if sox_recipe else ("flac", "wav", "mp3")
    if ext not in ok:
        if ext == "mp3" and sox_recipe:
            raise ValueError("sox recipes cannot write mp3; convert to wav/flac first (recipe=convert)")
        ext = "flac"
    return ext


def _ff_encode_args(ext: str, sr, depth) -> list[str]:
    args: list[str] = []
    if sr:
        args += ["-ar", str(int(sr))]
    if ext == "flac":
        args += ["-c:a", "flac", "-sample_fmt", "s32" if (depth or 16) > 16 else "s16"]
    elif ext == "wav":
        if (depth or 16) <= 16:
            codec = "pcm_s16le"
        elif depth == 24:
            codec = "pcm_s24le"
        else:
            codec = "pcm_s32le"
        args += ["-c:a", codec]
    elif ext == "mp3":
        args += ["-c:a", "libmp3lame", "-q:a", "2"]
    return args


def _render_dir(in_path: Path, recipe: str, out_dir) -> Path:
    ts = time.strftime("%Y%m%d-%H%M%S")
    tag = f"{recipe}-{ts}-{uuid.uuid4().hex[:4]}"
    base = _resolve_safe(out_dir) if out_dir else in_path.parent / "renders"
    d = base / tag
    d.mkdir(parents=True, exist_ok=True)
    _resolve_safe(str(d))  # re-verify after mkdir (symlink safety)
    return d


def _parse_loudnorm_json(stderr: str):
    j = stderr.rfind("}")
    i = stderr.rfind("{", 0, j if j >= 0 else len(stderr))
    while i != -1 and j != -1:
        try:
            obj = json.loads(stderr[i:j + 1])
            if isinstance(obj, dict) and "input_i" in obj:
                return obj
        except json.JSONDecodeError:
            pass
        i = stderr.rfind("{", 0, i)
    return None


# --------------------------------------------------------------- recipes ---

def _recipe_loudnorm(in_path: Path, render_dir: Path, params) -> tuple[Path, list, dict]:
    """Two-pass EBU R128 loudnorm; remeasures the output and reports the honored mode."""
    merged = _merge_params(params, "loudnorm-2pass",
                           {"target_lufs", "target_tp", "target_lra", "out_format"})
    target_lufs = _fnum(merged, "target_lufs", -70, -5)
    target_tp = _fnum(merged, "target_tp", -9, 0)
    target_lra = _fnum(merged, "target_lra", 1, 50)
    ext = _pick_ext(in_path, merged)
    out = render_dir / f"{in_path.stem}_loudnorm-2pass.{ext}"
    if out.exists():
        raise ValueError(f"refusing to overwrite existing output: {out}")

    af1 = (f"loudnorm=I={_fmt_num(target_lufs)}:TP={_fmt_num(target_tp)}:"
           f"LRA={_fmt_num(target_lra)}:print_format=json")
    argv1 = [FFMPEG, "-hide_banner", "-nostats", "-i", str(in_path), "-af", af1, "-f", "null", "-"]
    r1 = _run(argv1, timeout=RENDER_TIMEOUT)
    if r1.returncode != 0:
        raise RuntimeError(f"loudnorm pass 1 failed: {_tail(r1.stderr)}")
    m1 = _parse_loudnorm_json(r1.stderr)
    if not m1:
        raise RuntimeError(f"loudnorm pass 1: no JSON measurement found: {_tail(r1.stderr)}")

    af2 = (f"loudnorm=I={_fmt_num(target_lufs)}:TP={_fmt_num(target_tp)}:"
           f"LRA={_fmt_num(target_lra)}:measured_I={m1['input_i']}:measured_TP={m1['input_tp']}:"
           f"measured_LRA={m1['input_lra']}:measured_thresh={m1['input_thresh']}:"
           f"offset={m1.get('target_offset', 0)}:linear=true:print_format=json")
    ident = _identity(_ffprobe_json(in_path))
    argv2 = [FFMPEG, "-hide_banner", "-nostats", "-y", "-i", str(in_path), "-af", af2, "-vn",
             *_ff_encode_args(ext, ident.get("sample_rate_hz"), ident.get("bit_depth")), str(out)]
    r2 = _run(argv2, timeout=RENDER_TIMEOUT)
    if r2.returncode != 0:
        raise RuntimeError(f"loudnorm pass 2 failed: {_tail(r2.stderr)}")
    m2 = _parse_loudnorm_json(r2.stderr) or {}
    mode = str(m2.get("normalization_type", "")).lower() or None

    extra = {"loudnorm": {
        "target_lufs": target_lufs, "target_tp": target_tp, "target_lra": target_lra,
        "pass1_measured": {k: m1.get(k) for k in
                           ("input_i", "input_tp", "input_lra", "input_thresh", "target_offset")},
        "reported_mode": mode,
        "pass2_reported_output": {k: m2.get(k) for k in
                                  ("output_i", "output_tp", "output_lra", "target_offset")},
    }}
    return out, [argv1, argv2], extra


def _recipe_trim_silence(in_path: Path, render_dir: Path, params) -> tuple[Path, list, dict]:
    """Trim leading/trailing silence via the ffmpeg silenceremove filter.

    Note: SoX_ng 14.8.0.1 has no 'silenceremove' effect (that is an ffmpeg filter
    name); this recipe therefore uses ffmpeg's silenceremove, with an areverse
    sandwich for the tail. Each trimmed end also drops up to min_silence_s of the
    first/last non-silent audio — inherent to the filter's detection window.
    """
    merged = _merge_params(params, "trim-silence",
                           {"threshold_db", "min_silence_s", "mode", "out_format"})
    threshold_db = _fnum(merged, "threshold_db", -80, -20)
    min_silence_s = _fnum(merged, "min_silence_s", 0.01, 5)
    mode = _schoice(merged, "mode", ("start", "end", "both"))
    ext = _pick_ext(in_path, merged)
    out = render_dir / f"{in_path.stem}_trim-silence.{ext}"
    if out.exists():
        raise ValueError(f"refusing to overwrite existing output: {out}")

    f1 = (f"silenceremove=start_periods=1:start_duration={_fmt_num(min_silence_s)}:"
          f"start_threshold={_fmt_num(threshold_db)}dB")
    if mode == "start":
        filt = f1
    elif mode == "end":
        filt = f"areverse,{f1},areverse"
    else:
        filt = f"{f1},areverse,{f1},areverse"

    ident = _identity(_ffprobe_json(in_path))
    argv = [FFMPEG, "-hide_banner", "-nostats", "-y", "-i", str(in_path), "-af", filt, "-vn",
            *_ff_encode_args(ext, ident.get("sample_rate_hz"), ident.get("bit_depth")), str(out)]
    r = _run(argv, timeout=RENDER_TIMEOUT)
    if r.returncode != 0:
        raise RuntimeError(f"trim-silence failed: {_tail(r.stderr)}")
    return out, [argv], {"trim_silence": {"mode": mode, "threshold_db": threshold_db,
                                          "min_silence_s": min_silence_s}}


def _recipe_normalize_peak(in_path: Path, render_dir: Path, params) -> tuple[Path, list, dict]:
    merged = _merge_params(params, "normalize-peak", {"target_dbfs", "out_format"})
    target_dbfs = _fnum(merged, "target_dbfs", -30, 0)
    ext = _pick_ext(in_path, merged, sox_recipe=True)
    out = render_dir / f"{in_path.stem}_normalize-peak.{ext}"
    if out.exists():
        raise ValueError(f"refusing to overwrite existing output: {out}")
    argv = [SOX, str(in_path), str(out), "norm", _fmt_num(target_dbfs)]
    r = _run(argv, timeout=RENDER_TIMEOUT)
    if r.returncode != 0:
        raise RuntimeError(f"sox norm failed: {_tail(r.stderr)}")
    return out, [argv], {"peak_normalize": {"target_dbfs": target_dbfs}}


def _recipe_fade(in_path: Path, render_dir: Path, params) -> tuple[Path, list, dict]:
    merged = _merge_params(params, "fade", {"fade_in_s", "fade_out_s", "curve", "out_format"})
    fin = _fnum(merged, "fade_in_s", 0, 3600)
    fout = _fnum(merged, "fade_out_s", 0, 3600)
    curve = _schoice(merged, "curve", ("t", "q", "h", "l", "p", "s"))
    if fin <= 0 and fout <= 0:
        raise ValueError("at least one of fade_in_s / fade_out_s must be > 0")
    dur = _identity(_ffprobe_json(in_path)).get("duration_s")
    if not dur:
        raise RuntimeError("could not determine input duration for fade")
    if fin + fout >= dur - 0.01:
        raise ValueError(f"fade lengths ({fin}+{fout}s) exceed duration ({dur}s)")
    ext = _pick_ext(in_path, merged, sox_recipe=True)
    out = render_dir / f"{in_path.stem}_fade.{ext}"
    if out.exists():
        raise ValueError(f"refusing to overwrite existing output: {out}")
    if fout > 0:
        # stop-position = end of file: fade-out occupies [dur-fout, dur], nothing discarded
        argv = [SOX, str(in_path), str(out), "fade", curve,
                _fmt_num(fin), _fmt_num(dur), _fmt_num(fout)]
    else:
        argv = [SOX, str(in_path), str(out), "fade", curve, _fmt_num(fin)]
    r = _run(argv, timeout=RENDER_TIMEOUT)
    if r.returncode != 0:
        raise RuntimeError(f"sox fade failed: {_tail(r.stderr)}")
    return out, [argv], {"fade": {"fade_in_s": fin, "fade_out_s": fout, "curve": curve,
                                  "duration_s": dur}}


def _recipe_highpass(in_path: Path, render_dir: Path, params) -> tuple[Path, list, dict]:
    merged = _merge_params(params, "highpass", {"freq_hz", "poles", "out_format"})
    freq = _fnum(merged, "freq_hz", 10, 20000)
    poles = _ichoice(merged, "poles", (1, 2))
    ext = _pick_ext(in_path, merged, sox_recipe=True)
    out = render_dir / f"{in_path.stem}_highpass.{ext}"
    if out.exists():
        raise ValueError(f"refusing to overwrite existing output: {out}")
    argv = [SOX, str(in_path), str(out), "highpass", f"-{poles}", _fmt_num(freq)]
    r = _run(argv, timeout=RENDER_TIMEOUT)
    if r.returncode != 0:
        raise RuntimeError(f"sox highpass failed: {_tail(r.stderr)}")
    return out, [argv], {"highpass": {"freq_hz": freq, "poles": poles}}


def _recipe_convert(in_path: Path, render_dir: Path, params) -> tuple[Path, list, dict]:
    merged = _merge_params(params, "convert", {"format", "sample_rate", "bits"})
    fmt = str(merged.get("format") or in_path.suffix.lower().lstrip(".") or "flac").lower()
    if fmt not in ("flac", "wav"):
        raise ValueError("convert format must be flac or wav")
    sr = None if merged.get("sample_rate") is None else _ichoice_isint(merged, "sample_rate", 8000, 192000)
    bits = None if merged.get("bits") is None else _ichoice(merged, "bits", (8, 16, 24, 32))
    out = render_dir / f"{in_path.stem}_convert.{fmt}"
    if out.exists():
        raise ValueError(f"refusing to overwrite existing output: {out}")
    # sox CLI order: options before the outfile apply to the output; effects come last
    argv = [SOX, str(in_path)]
    if bits:
        argv += ["-b", str(bits)]
    argv += [str(out)]
    if sr:
        argv += ["rate", str(sr)]
    r = _run(argv, timeout=RENDER_TIMEOUT)
    if r.returncode != 0:
        raise RuntimeError(f"sox convert failed: {_tail(r.stderr)}")
    return out, [argv], {"convert": {"format": fmt, "sample_rate": sr, "bits": bits}}


def _ichoice_isint(merged: dict, key: str, lo: int, hi: int) -> int:
    v = _to_int(merged.get(key))
    if v is None or v < lo or v > hi:
        raise ValueError(f"param {key!r} must be an integer in [{lo}, {hi}]")
    return int(v)


def _finalize_loudnorm(extra: dict, m_after: dict) -> None:
    ln = extra["loudnorm"]
    after_i = m_after["loudness"]["ebur128_lufs"]
    ln["remeasured_lufs"] = after_i
    ln["remeasured_delta_lu"] = round(after_i - ln["target_lufs"], 2) if after_i is not None else None
    ln["linear_reported"] = ln.get("reported_mode") == "linear"
    ln["applied_ok"] = after_i is not None and abs(after_i - ln["target_lufs"]) <= 0.75
    ln["honored"] = bool(ln["linear_reported"] and ln["applied_ok"])


# ------------------------------------------------------------- job plumbing —

def _job_path(job_id: str) -> Path:
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,79}", job_id or ""):
        raise ValueError("job_id must match [A-Za-z0-9][A-Za-z0-9_.-]{0,79}")
    p = JOBS_DIR / f"{job_id}.json"
    rp = Path(os.path.realpath(p))
    if rp.parent != Path(os.path.realpath(JOBS_DIR)):
        raise ValueError("job_id escapes the jobs directory")
    return p


def _runner_alive(pid) -> bool:
    pid = _to_int(pid)
    if not pid or pid <= 0:
        return False
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    try:
        cmdline = Path(f"/proc/{pid}/cmdline").read_bytes().decode("utf-8", "replace")
    except OSError:
        return False
    return "_demucs_runner.py" in cmdline


def _update_job(job_path: Path, **kw) -> dict:
    rec = json.loads(job_path.read_text())
    rec.update(kw)
    _write_json(job_path, rec)
    return rec


def _busy_job():
    """Return the running job record, reconciling stale 'running' records first."""
    JOBS_DIR.mkdir(parents=True, exist_ok=True)
    for f in sorted(JOBS_DIR.glob("*.json")):
        try:
            rec = json.loads(f.read_text())
        except (json.JSONDecodeError, OSError):
            continue
        state = rec.get("state")
        if state == "running" and not _runner_alive(rec.get("pid")):
            try:
                _update_job(f, state="failed", finished=_now_iso(),
                            note="runner process not alive and job never finalized (reconciled)")
            except OSError:
                pass
            continue
        if state == "starting":
            try:
                _update_job(f, state="failed", finished=_now_iso(),
                            note="server stopped before the runner was spawned (reconciled)")
            except OSError:
                pass
            continue
        if state == "running":
            return rec
    return None


# ------------------------------------------------------------------- tools ---

@mcp.tool()
def audio_qc(file: str) -> str:
    """Read-only QC of an audio file. Returns compact JSON with:
    identity (format, sample rate, channels, duration, bit depth via ffprobe),
    loudness (pyloudnorm integrated LUFS [stereo and mono-downmix variants] and
    ffmpeg ebur128 I/LRA, with the expected gating-delta note), sample peak (dBFS),
    true peak (dBTP) and true_peak_ok (<= -1 dBTP streaming headroom)."""
    try:
        p = _resolve_safe(file)
        if not p.is_file():
            return _err(f"not a file: {p}")
        m = _measure(p)
        return json.dumps({"ok": True, "file": str(p), **m}, default=str)
    except Exception as exc:
        return _err(f"{type(exc).__name__}: {exc}")


@mcp.tool()
def dsp_render(file: str, recipe: str, params: dict | None = None, out_dir: str | None = None) -> str:
    """Render an allowlisted versioned DSP recipe to a NEW output file (the input is never modified).

    Recipes: loudnorm-2pass (target_lufs/target_tp/target_lra; two-pass EBU R128,
    linear=true + measured values, output re-measured and the honored mode reported),
    trim-silence (threshold_db/min_silence_s/mode=start|end|both; ffmpeg silenceremove),
    normalize-peak (target_dbfs; sox norm), fade (fade_in_s/fade_out_s/curve; sox fade),
    highpass (freq_hz/poles; sox highpass), convert (format=flac|wav, sample_rate, bits=8|16|24|32; sox).
    Default output dir: <input dir>/renders/<recipe>-<timestamp>/.
    Returns output path, sha256, argv arrays, tool versions, before/after QC summary,
    and a receipt.json written beside the output."""
    try:
        in_path = _resolve_safe(file)
        if not in_path.is_file():
            return _err(f"not a file: {in_path}")
        if recipe not in RECIPE_NAMES:
            return _err(f"recipe must be one of {list(RECIPE_NAMES)}")
        render_dir = _render_dir(in_path, recipe, out_dir)
        if recipe == "loudnorm-2pass":
            out_path, argv_list, extra = _recipe_loudnorm(in_path, render_dir, params)
        elif recipe == "trim-silence":
            out_path, argv_list, extra = _recipe_trim_silence(in_path, render_dir, params)
        elif recipe == "normalize-peak":
            out_path, argv_list, extra = _recipe_normalize_peak(in_path, render_dir, params)
        elif recipe == "fade":
            out_path, argv_list, extra = _recipe_fade(in_path, render_dir, params)
        elif recipe == "highpass":
            out_path, argv_list, extra = _recipe_highpass(in_path, render_dir, params)
        else:
            out_path, argv_list, extra = _recipe_convert(in_path, render_dir, params)

        if Path(os.path.realpath(out_path)) == in_path:
            raise RuntimeError("refusing to overwrite the input file")
        if not out_path.is_file() or out_path.stat().st_size == 0:
            raise RuntimeError("recipe produced no output")

        m_before = _measure(in_path)
        m_after = _measure(out_path)
        qc_before, qc_after = _qc_summary(m_before), _qc_summary(m_after)
        if recipe == "loudnorm-2pass":
            _finalize_loudnorm(extra, m_after)

        versions = _tool_versions()
        receipt = {
            "recipe": recipe, "recipe_version": _recipe(recipe).get("version"),
            "created": _now_iso(),
            "input": str(in_path), "input_sha256": _sha256(in_path),
            "output": str(out_path), "output_sha256": _sha256(out_path),
            "argv": argv_list, "tool_versions": versions,
            "qc_before": qc_before, "qc_after": qc_after, "details": extra,
        }
        receipt_path = render_dir / "receipt.json"
        _write_json(receipt_path, receipt)

        result = {"ok": True, "recipe": recipe, "input": str(in_path), "output": str(out_path),
                  "output_sha256": receipt["output_sha256"], "receipt_path": str(receipt_path),
                  "argv": argv_list, "tool_versions": versions,
                  "qc_before": qc_before, "qc_after": qc_after}
        result.update(extra)
        return json.dumps(result, default=str)
    except Exception as exc:
        return _err(f"{type(exc).__name__}: {exc}")


@mcp.tool()
def stem_submit(file: str, model: str = "htdemucs", two_stems: str = "vocals") -> str:
    """Submit a detached demucs separation job (envs/ml, CPU, ~2x realtime on this host).

    model: htdemucs | htdemucs_ft | htdemucs_6s. two_stems: vocals|drums|bass|other,
    or "none"/"" for all four stems. Single-concurrency lock: while a job is running,
    a new submit returns {"busy": true, "running_job_id": ...}. Job records +
    outputs live under audio-tools/jobs/. Poll with stem_status(job_id)."""
    try:
        p = _resolve_safe(file)
        if not p.is_file():
            return _err(f"not a file: {p}")
        if model not in STEM_MODELS:
            return _err(f"model must be one of {list(STEM_MODELS)}")
        two = (two_stems or "").strip().lower()
        if two in ("", "none", "null"):
            two = None
        elif two not in TWO_STEMS_CHOICES:
            return _err(f"two_stems must be one of {list(TWO_STEMS_CHOICES)} or 'none'")

        busy = _busy_job()
        if busy:
            return json.dumps({"ok": True, "submitted": False, "busy": True,
                               "running_job_id": busy.get("job_id"),
                               "message": "single-concurrency lock: a stem job is already running"})

        JOBS_DIR.mkdir(parents=True, exist_ok=True)
        job_id = f"demucs-{time.strftime('%Y%m%d-%H%M%S')}-{uuid.uuid4().hex[:6]}"
        out_root = JOBS_DIR / job_id
        out_root.mkdir(parents=True, exist_ok=True)
        log_path = JOBS_DIR / f"{job_id}.log"
        job_path = JOBS_DIR / f"{job_id}.json"
        cmd = [str(ML_PY), "-m", "demucs.separate", "-n", model]
        if two:
            cmd.append(f"--two-stems={two}")
        cmd += ["-d", "cpu", "-o", str(out_root), str(p)]

        record = {"job_id": job_id, "state": "starting", "file": str(p), "model": model,
                  "two_stems": two, "out_root": str(out_root), "log": str(log_path),
                  "cmd": cmd, "cwd": str(ROOT), "created": _now_iso()}
        _write_json(job_path, record)
        with open(log_path, "ab") as log:
            proc = subprocess.Popen([str(ML_PY), str(DEMUCS_RUNNER), str(job_path)],
                                    stdout=log, stderr=log, stdin=subprocess.DEVNULL,
                                    start_new_session=True, cwd=str(ROOT), env=_clean_env())
        _update_job(job_path, pid=proc.pid, state="running", started=_now_iso(),
                    started_ts=time.time())
        return json.dumps({"ok": True, "submitted": True, "busy": False, "job_id": job_id,
                           "pid": proc.pid, "model": model, "two_stems": two,
                           "out_root": str(out_root), "log": str(log_path),
                           "note": "poll stem_status(job_id); poll again after 'done' to list stems"})
    except Exception as exc:
        return _err(f"{type(exc).__name__}: {exc}")


@mcp.tool()
def stem_status(job_id: str) -> str:
    """Status of a stem job: state (starting/running/done/failed), pid + liveness,
    elapsed seconds, exit code, and on done the stem files (name, path, bytes, sha256).
    Reconciles crashed runners to failed instead of reporting a phantom running job."""
    try:
        job_path = _job_path(job_id)
        if not job_path.is_file():
            return _err(f"unknown job: {job_id}")
        rec = json.loads(job_path.read_text())
        alive = None
        if rec.get("state") in ("starting", "running"):
            alive = _runner_alive(rec.get("pid"))
            if not alive:
                rec = _update_job(job_path, state="failed", finished=_now_iso(),
                                  note="runner process not alive and job never finalized — see log")
        elapsed = rec.get("elapsed_s")
        if elapsed is None and rec.get("started_ts"):
            if rec.get("state") == "running":
                elapsed = round(time.time() - float(rec["started_ts"]), 1)
        return json.dumps({"ok": True, "job_id": rec.get("job_id", job_id), "state": rec.get("state"),
                           "pid": rec.get("pid"), "alive": alive, "file": rec.get("file"),
                           "model": rec.get("model"), "two_stems": rec.get("two_stems"),
                           "out_root": rec.get("out_root"), "log": rec.get("log"),
                           "created": rec.get("created"), "started": rec.get("started"),
                           "elapsed_s": elapsed, "exit_code": rec.get("exit_code"),
                           "stems": rec.get("stems", []), "note": rec.get("note")}, default=str)
    except Exception as exc:
        return _err(f"{type(exc).__name__}: {exc}")


def _silence_spans(path: Path, noise_db: float = -50.0, min_s: float = 0.5):
    r = _run([FFMPEG, "-hide_banner", "-nostats", "-i", str(path),
              "-af", f"silencedetect=noise={_fmt_num(noise_db)}dB:d={_fmt_num(min_s)}",
              "-f", "null", "-"])
    if r.returncode != 0:
        raise RuntimeError(f"silencedetect failed: {_tail(r.stderr)}")
    starts = [float(x) for x in re.findall(r"silence_start:\s*([+-]?\d+(?:\.\d+)?)", r.stderr)]
    ends = [float(x) for x in re.findall(r"silence_end:\s*([+-]?\d+(?:\.\d+)?)", r.stderr)]
    dur = _identity(_ffprobe_json(path)).get("duration_s") or 0.0
    regions = list(zip(starts, ends[:len(starts)]))
    if len(starts) > len(ends):
        regions.append((starts[-1], dur))  # open silence region at EOF
    head = round(regions[0][1], 3) if regions and regions[0][0] <= 0.05 else 0.0
    tail = round(dur - regions[-1][0], 3) if regions and regions[-1][1] >= dur - 0.1 else 0.0
    return head, tail, len(regions)


@mcp.tool()
def release_validate(file: str, profile: str = "streaming") -> str:
    """Validate a rendered file against a release profile (VALIDATION ONLY — never
    publishes, never modifies the source). Checks: true peak <= -1 dBTP, wav/flac +
    sample rate >= 44.1 kHz, min duration, head/tail silence within threshold; the
    integrated loudness vs the -14 LUFS playback-target band is WARN-ONLY (never
    auto-compress). Writes a QC receipt JSON to <file dir>/qc/ and returns
    PASS/FAIL with per-check detail."""
    try:
        p = _resolve_safe(file)
        if not p.is_file():
            return _err(f"not a file: {p}")
        prof = PROFILES.get(profile)
        if not prof:
            return _err(f"unknown profile {profile!r}; available: {sorted(PROFILES)}")

        m = _measure(p)
        ident, loud, peaks = m["identity"], m["loudness"], m["peaks"]
        head, tail, n_regions = _silence_spans(p)

        def check(name, ok, detail, **vals):
            return {"check": name, "status": "pass" if ok else "fail", "detail": detail, **vals}

        checks = []
        fmt_ok = ident.get("container") in prof["formats"] or ident.get("codec") in prof["formats"]
        checks.append(check("format", fmt_ok,
                            f"container={ident.get('container')} codec={ident.get('codec')} "
                            f"(required: {'/'.join(prof['formats'])})"))
        sr = ident.get("sample_rate_hz")
        checks.append(check("sample_rate", isinstance(sr, int) and sr >= prof["min_sample_rate_hz"],
                            f"sample_rate={sr} Hz (required >= {prof['min_sample_rate_hz']})",
                            value=sr))
        dur = ident.get("duration_s")
        checks.append(check("min_duration", dur is not None and dur >= prof["min_duration_s"],
                            f"duration={dur}s (required >= {prof['min_duration_s']}s)", value=dur))
        tpk = peaks.get("true_peak_dbtp")
        checks.append(check("true_peak", tpk is not None and tpk <= prof["tp_max_dbtp"],
                            f"true_peak={tpk} dBTP (required <= {prof['tp_max_dbtp']} dBTP)", value=tpk))
        checks.append(check("head_silence", head <= prof["head_silence_max_s"],
                            f"head_silence={head}s (max {prof['head_silence_max_s']}s)", value=head))
        checks.append(check("tail_silence", tail <= prof["tail_silence_max_s"],
                            f"tail_silence={tail}s (max {prof['tail_silence_max_s']}s)", value=tail))

        i_lufs = loud.get("ebur128_lufs")
        within_band = i_lufs is not None and abs(i_lufs - prof["target_lufs"]) <= prof["lufs_band_lu"]
        loud_check = {"check": "loudness_band", "status": "ok" if within_band else "warn",
                      "hard": False,
                      "detail": f"integrated={i_lufs} LUFS vs -14 playback target "
                                f"(band +-{prof['lufs_band_lu']} LU) — REPORTED, warn-only; "
                                f"never auto-compress for a playback target",
                      "value": i_lufs}

        verdict = "PASS" if all(c["status"] == "pass" for c in checks) else "FAIL"
        receipt = {
            "kind": "release-qc-receipt", "profile": profile, "created": _now_iso(),
            "verdict": verdict, "file": str(p), "file_sha256": _sha256(p),
            "thresholds": prof, "checks": checks, "loudness_band": loud_check,
            "measured": {"ebur128_lufs": i_lufs,
                         "pyloudnorm_lufs": loud.get("pyloudnorm_lufs"),
                         "true_peak_dbtp": tpk, "sample_peak_dbfs": peaks.get("sample_peak_dbfs"),
                         "lra_lu": loud.get("lra_lu"),
                         "head_silence_s": head, "tail_silence_s": tail,
                         "silence_regions": n_regions, **ident},
        }
        qc_dir = p.parent / "qc"
        qc_dir.mkdir(parents=True, exist_ok=True)
        _resolve_safe(str(qc_dir))
        receipt_path = qc_dir / f"{p.name}.release-{profile}-{time.strftime('%Y%m%d-%H%M%S')}.json"
        _write_json(receipt_path, receipt)

        return json.dumps({"ok": True, "verdict": verdict, "profile": profile, "file": str(p),
                           "checks": checks, "loudness_band": loud_check,
                           "receipt_path": str(receipt_path)}, default=str)
    except Exception as exc:
        return _err(f"{type(exc).__name__}: {exc}")


if __name__ == "__main__":
    # stdio servers get SIGPIPE/killed-pipe races on Hermes shutdown; swallow the
    # harmless BrokenPipeError flush race without hiding real failures.
    import signal
    try:
        signal.signal(signal.SIGPIPE, signal.SIG_DFL)
    except (AttributeError, ValueError):  # pragma: no cover
        pass
    try:
        mcp.run()
    except BrokenPipeError:
        try:
            sys.stdout.close()
        except Exception:
            pass
    except ExceptionGroup as eg:  # PY>=3.11: TaskGroup wraps the pipe error
        if any(isinstance(e, BrokenPipeError) for e in eg.exceptions):
            pass
        else:
            raise
