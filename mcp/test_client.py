#!/usr/bin/env python3
"""End-to-end test for the audio MCP server over a REAL MCP stdio client.

Run:  cd ~/Work/Projects/audio-tools && envs/server/bin/python mcp/test_client.py

Phases (no stubs, real files):
  1. initialize + list_tools (assert the 5 tools)
  2. audio_qc on the phase-1 proof file (assert LUFS/peak fields) + allowlist refusal
  3. make a 20 s clip with ffmpeg
  4. dsp_render loudnorm-2pass (target -14/-1) -> assert new output + linear honored
  5. stem_submit on the clip; assert the single-concurrency lock; poll stem_status to
     done; assert vocals.wav / no_vocals.wav exist
  6. release_validate on the loudnorm output -> assert the QC receipt was written
"""
import asyncio
import json
import os
import subprocess
import sys
import time
from pathlib import Path

from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

ROOT = Path(__file__).resolve().parent.parent
SERVER = ROOT / "mcp" / "audio-mcp.sh"
FFMPEG = "/usr/bin/ffmpeg"
SRC = Path("/home/j_kro/Music/acestep/final-round_00001.flac")
ART = ROOT / "mcp" / ".test-artifacts"


async def call(session: ClientSession, name: str, args: dict, timeout: float) -> dict:
    res = await asyncio.wait_for(session.call_tool(name, args), timeout)
    return json.loads(res.content[0].text)


async def main() -> None:
    ART.mkdir(parents=True, exist_ok=True)
    params = StdioServerParameters(command=str(SERVER), args=[], env={**os.environ})
    async with stdio_client(params) as (read, write):
        async with ClientSession(read, write) as session:
            await asyncio.wait_for(session.initialize(), 60)
            tools = await asyncio.wait_for(session.list_tools(), 60)
            names = sorted(t.name for t in tools.tools)
            print("TOOLS:", names)
            assert names == ["audio_qc", "dsp_render", "release_validate",
                             "stem_status", "stem_submit"], names
            print("PASS  tool count = 5")

            # --- 1. audio_qc on the phase-1 proof file -------------------------
            qc = await call(session, "audio_qc", {"file": str(SRC)}, 300)
            assert qc.get("ok") is True, qc
            lufs, peaks, ident = qc["loudness"], qc["peaks"], qc["identity"]
            assert abs(lufs["pyloudnorm_lufs_mono_downmix"] - (-14.27)) < 0.15, lufs
            assert abs(lufs["pyloudnorm_lufs"] - (-10.59)) < 0.15, lufs
            assert abs(lufs["ebur128_lufs"] - (-10.6)) < 0.15, lufs
            assert abs(peaks["sample_peak_dbfs"] - 0.0) < 0.01, peaks
            assert abs(peaks["true_peak_dbtp"] - 0.4) < 0.15, peaks
            assert peaks["true_peak_ok"] is False, peaks   # +0.4 dBTP violates -1 dBTP
            assert ident["sample_rate_hz"] == 48000 and ident["channels"] == 2, ident
            assert abs(ident["duration_s"] - 90.0) < 0.01
            print(f"PASS  audio_qc: pyln(stereo)={lufs['pyloudnorm_lufs']} "
                  f"pyln(mono)={lufs['pyloudnorm_lufs_mono_downmix']} "
                  f"ebur128={lufs['ebur128_lufs']} LRA={lufs['lra_lu']} "
                  f"peak={peaks['sample_peak_dbfs']}dBFS TPK={peaks['true_peak_dbtp']}dBTP "
                  f"tp_ok={peaks['true_peak_ok']}")

            bad = await call(session, "audio_qc", {"file": "/etc/passwd"}, 60)
            assert bad.get("ok") is False and "allowlist" in bad.get("error", ""), bad
            print("PASS  allowlist refusal outside /home/j_kro/{Music,Documents} + repo")

            # --- 2. make a 20 s clip ------------------------------------------
            clip = ART / "clip-20s.flac"
            subprocess.run([FFMPEG, "-y", "-v", "error", "-ss", "10", "-t", "20",
                            "-i", str(SRC), "-c:a", "flac", str(clip)], check=True)
            print(f"clip: {clip} ({clip.stat().st_size} bytes)")

            # --- 3. dsp_render loudnorm-2pass ---------------------------------
            rr = await call(session, "dsp_render",
                            {"file": str(clip), "recipe": "loudnorm-2pass",
                             "params": {"target_lufs": -14.0, "target_tp": -1.0}}, 900)
            assert rr.get("ok") is True, rr
            out = Path(rr["output"])
            assert out.is_file() and out.stat().st_size > 0, rr
            assert rr["receipt_path"] and Path(rr["receipt_path"]).is_file()
            ln = rr["loudnorm"]
            assert ln["reported_mode"] == "linear", ln
            assert ln["honored"] is True, ln
            assert abs(ln["remeasured_lufs"] - (-14.0)) <= 0.75, ln
            print(f"PASS  dsp_render loudnorm-2pass: {out.name} mode={ln['reported_mode']} "
                  f"remeasured={ln['remeasured_lufs']} LUFS (delta {ln['remeasured_delta_lu']}) "
                  f"honored={ln['honored']}")

            # --- 4. stems: submit, lock, poll ---------------------------------
            sub = await call(session, "stem_submit", {"file": str(clip)}, 120)
            assert sub.get("submitted") is True, sub
            job = sub["job_id"]
            print(f"stem_submit: job={job} pid={sub['pid']}")
            sub2 = await call(session, "stem_submit", {"file": str(clip)}, 120)
            assert sub2.get("busy") is True and sub2.get("running_job_id") == job, sub2
            print(f"PASS  single-concurrency lock: second submit busy -> {sub2['running_job_id']}")

            deadline = time.time() + 900
            st = None
            while time.time() < deadline:
                st = await call(session, "stem_status", {"job_id": job}, 120)
                if st["state"] in ("done", "failed"):
                    break
                await asyncio.sleep(5)
            assert st and st["state"] == "done", st
            stem_names = sorted(x["name"] for x in st["stems"])
            assert "vocals.wav" in stem_names and "no_vocals.wav" in stem_names, stem_names
            print(f"PASS  stems done: {stem_names} in {st['elapsed_s']}s "
                  f"(sha256 recorded: {[bool(x.get('sha256')) for x in st['stems']]})")

            # --- 5. release_validate on the loudnorm output --------------------
            rv = await call(session, "release_validate", {"file": str(out)}, 900)
            assert rv.get("ok") is True, rv
            receipt = Path(rv["receipt_path"])
            assert receipt.is_file(), rv
            assert rv["verdict"] in ("PASS", "FAIL"), rv
            detail = [(c["check"], c["status"]) for c in rv["checks"]]
            print(f"PASS  release_validate: verdict={rv['verdict']} checks={detail} "
                  f"loudness_band={rv['loudness_band']['status']}")
            print(f"      receipt: {receipt}")

    print("ALL CHECKS PASSED")


if __name__ == "__main__":
    asyncio.run(main())
