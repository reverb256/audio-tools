#!/usr/bin/env python3
"""Detached stem-separation job runner for the audio MCP server.

Runs under envs/ml (python 3.12: torch CPU, torchaudio, demucs 4.1.0).
argv: JOB_JSON  ->  executes the argv array recorded in the job file, then
updates the job file atomically with the outcome (done/failed, stems, hashes).

The MCP server spawns this file detached (start_new_session=True, stdout/err
redirected to the job log). The job record is the single source of truth:
  state=starting  -> written by the server before spawn
  state=running   -> server adds pid after spawn
  state=done|failed -> written here when demucs exits
Every write is tmp-file + os.replace so readers never see a partial record.
"""
from __future__ import annotations

import hashlib
import json
import os
import subprocess
import sys
import time
from pathlib import Path


def now_iso() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%S%z")


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def update(job_path: str, **kw) -> None:
    p = Path(job_path)
    rec = json.loads(p.read_text())
    rec.update(kw)
    tmp = p.with_suffix(".tmp")
    tmp.write_text(json.dumps(rec, indent=2) + "\n")
    os.replace(tmp, p)


def main() -> int:
    job_path = sys.argv[1]
    job = json.loads(Path(job_path).read_text())
    env = {k: v for k, v in os.environ.items() if k not in ("PYTHONPATH", "PYTHONHOME")}
    t0 = time.time()
    try:
        with open(job["log"], "ab") as log:
            log.write(f"\n[runner] {now_iso()} exec: {' '.join(job['cmd'])}\n".encode())
            log.flush()
            rc = subprocess.call(
                job["cmd"], stdout=log, stderr=subprocess.STDOUT, env=env,
                cwd=job.get("cwd") or None,
            )
        stems = []
        out_root = Path(job["out_root"])
        for f in sorted(out_root.rglob("*.wav")):
            stems.append({
                "name": f.name,
                "path": str(f),
                "bytes": f.stat().st_size,
                "sha256": sha256(f),
            })
        if rc == 0 and stems:
            update(job_path, state="done", exit_code=rc, finished=now_iso(),
                   elapsed_s=round(time.time() - t0, 1), stems=stems)
        else:
            note = f"demucs exited {rc}" if rc != 0 else "demucs exited 0 but no stems found"
            update(job_path, state="failed", exit_code=rc, finished=now_iso(),
                   elapsed_s=round(time.time() - t0, 1), stems=stems, note=note)
        return 0
    except Exception as exc:  # record the failure instead of dying silently
        try:
            update(job_path, state="failed", finished=now_iso(),
                   elapsed_s=round(time.time() - t0, 1),
                   note=f"runner error: {type(exc).__name__}: {exc}")
        except Exception:
            pass
        return 1


if __name__ == "__main__":
    sys.exit(main())
