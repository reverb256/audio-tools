# audio-tools — the audio lane's toolchain (phase 1: 2026-10-16 · MCP facade: 2026-10-17)

PURPOSE. Agent-driven audio processing for the music pipeline: QC, DSP, stem
separation, mastering, release validation. Design follows the deep-research
brief (`~/Documents/research/linux-audio-agent-tooling.md`): a thin MCP facade
over pinned CLI workers (§5 "a small stdio facade"); CPU-first; GPU only via
the miner-handoff queue if a windowed need ever appears.

## Layout
- `envs/cpu`    — python 3.12 (uv): librosa 1.0.0 · pyloudnorm 0.14.0 · soundfile · ffmpeg-normalize 1.43.0 · numpy 2.5.4
- `envs/ml`     — python 3.12 (uv): torch 2.14.1+cpu · torchaudio · demucs 4.1.0
- `envs/server` — python 3.12 (uv): mcp 1.30.0, PINNED `mcp>=1.0,<2` (mcp 2.x renamed FastMCP → MCPServer and breaks every 1.x bridge)
- `recipes/`    — versioned DSP recipes as DATA (JSON defaults + docs; ranges validated in code, unknown params rejected)
- `mcp/`        — the `audio` stdio MCP server (see the MCP section below)
- `jobs/`       — stem job records / logs / outputs (runtime; gitignored)
- `reference/`  — shallow clones for design reference only (third-party repos; never run)

Environment rebuild (venvs are gitignored):
```bash
uv venv --python 3.12 envs/cpu    && uv pip install --python envs/cpu/bin/python    librosa pyloudnorm soundfile ffmpeg-normalize
uv venv --python 3.12 envs/ml     && uv pip install --python envs/ml/bin/python     torch torchaudio demucs
uv venv --python 3.12 envs/server && uv pip install --python envs/server/bin/python "mcp>=1.0,<2"
```

System tools (invoked by ABSOLUTE path in the server — on zephyr a Hermes-private
ffmpeg build shadows PATH): `/usr/bin/ffmpeg` n9.0.2 (loudnorm + ebur128 +
silenceremove), `/usr/bin/sox` SoX_ng 14.8.0.1, `/usr/bin/ffprobe`, `uv 0.13.1`.

## MCP server (`mcp/`)

`mcp/audio_mcp_server.py` (FastMCP, stdio, server name `audio`) + `mcp/audio-mcp.sh`
wrapper (`unset PYTHONPATH`; root resolved relative to the script; execs
`envs/server/bin/python mcp/audio_mcp_server.py`). Registered with Hermes as
`audio` — a new session (or `/reload-mcp`) is required for tool pickup.

| Tool | Contract |
|---|---|
| `audio_qc(file)` | Read-only. ffprobe identity (format, sr, channels, bit depth, duration) + pyloudnorm integrated LUFS (stereo **and** mono-downmix variants) + sample peak + ffmpeg ebur128 (I / LRA / TPK). Flags `true_peak_ok` (≤ −1 dBTP streaming headroom) and carries the gating-delta note. |
| `dsp_render(file, recipe, params, out_dir=None)` | Allowlisted versioned recipes → NEW output under `<input dir>/renders/<recipe>-<timestamp>/` + `receipt.json` (argv arrays, tool versions, before/after QC, sha256s). Never overwrites the input. |
| `stem_submit(file, model='htdemucs', two_stems='vocals')` | Detached demucs job (envs/ml, CPU). Single-concurrency lock: a second submit while one runs returns `busy` + the running job id. Job record JSON per job under `jobs/`. |
| `stem_status(job_id)` | State / pid + liveness / elapsed / exit code; on done the stem files (path, bytes, sha256). Reconciles crashed runners to `failed` instead of phantom-running. |
| `release_validate(file, profile='streaming')` | VALIDATION ONLY (never publishes, never touches sources): TPK ≤ −1 dBTP, flac/wav + sr ≥ 44.1 kHz, min duration, head/tail silence; the −14 LUFS playback-target band is WARN-ONLY (never auto-compress). Writes a QC receipt to `<file dir>/qc/`. |

Safety: every path argument must realpath-resolve under `/home/j_kro/Music`,
`/home/j_kro/Documents`, or this repo — anything else is refused; all
subprocesses use argument arrays (no shell strings, no `shell=True`); tools
return paths / hashes / compact JSON — never audio arrays.

Recipes (`recipes/*.json`): `loudnorm-2pass` (two-pass EBU R128: measure with
JSON, apply with `linear=true` + measured values, then RE-MEASURE the output and
report whether linear mode was honored — loudnorm silently falls back to dynamic
when the change would breach the true-peak ceiling) · `trim-silence` (ffmpeg
`silenceremove`; SoX_ng has no such effect — tail trim uses an `areverse`
sandwich so interior quiet passages are never touched) · `normalize-peak` (sox
`norm`) · `fade` (sox `fade`, stop-position = file end so nothing is discarded) ·
`highpass` (sox, 1/2-pole) · `convert` (sox; `-b` before the outfile, `rate`
last).

### Verified end-to-end — 2026-10-17, `envs/server/bin/python mcp/test_client.py` (real runs)
- `audio_qc` on `~/Music/acestep/final-round_00001.flac`: pyloudnorm(stereo) **−10.585** LUFS ·
  mono-downmix **−14.267** · ebur128 I **−10.6 LUFS** · LRA 3.4 · sample peak 0.0 dBFS ·
  TPK **+0.4 dBTP** → `true_peak_ok=false`. The phase-1 "1–4 LU pyloudnorm↔ebur128
  delta" was mostly the mono downmix — matched-layout agreement is ~0.01 LU; both
  values are now reported side by side rather than one alone.
- `dsp_render loudnorm-2pass` on a 20 s clip (−14/−1): mode `linear`, remeasured
  −13.9 LUFS, `honored=true`.
- `stem_submit` → second submit returns lock `busy` → `stem_status` done in 8.1 s:
  `vocals.wav` + `no_vocals.wav` (CPU, sha256 recorded per stem).
- `release_validate` on the loudnorm output: receipt written into the render's
  `qc/` dir; verdict FAIL only on min duration (20 s test clip vs the 30 s
  streaming profile), every other check pass.
- Hermes: `hermes mcp test audio` → connected (1172 ms), 5 tools.

## Proofs (phase 1, 2026-10-16 — re-verified through the MCP facade above)
- **pyloudnorm**: −14.27 LUFS integrated (mono-downmix convention); sample peak **0.00 dBFS**.
- **ffmpeg ebur128**: I −10.6 LUFS, TPK **+0.4 dBTP** → violates the −1 dBTP streaming
  headroom recommendation. Live example of why the QC gate exists.
- **Demucs htdemucs, CPU**: 20 s clip → `vocals.wav` + `no_vocals.wav` (~2× realtime).
  CPU-first stems are viable at this scale — GPU handoff not required for phase 1.

## Usage
```bash
# QC (measure) — direct:
envs/cpu/bin/python -c "import sys,soundfile as sf,pyloudnorm as pyln; d,sr=sf.read(sys.argv[1]); print(pyln.Meter(sr).integrated_loudness(d), 'LUFS')" FILE
/usr/bin/ffmpeg -i FILE -af ebur128=peak=true -f null -

# QC / DSP / stems — through the MCP server (any MCP client):
#   audio_qc(file) · dsp_render(file, recipe, params) · stem_submit(file) · stem_status(job_id) · release_validate(file)
hermes mcp test audio          # health check (connect + tool discovery)
envs/server/bin/python mcp/test_client.py   # full end-to-end proof (real runs)

# Stems — direct:
envs/ml/bin/python -m demucs.separate -n htdemucs --two-stems=vocals -d cpu -o OUT FILE
```

## Next
1. ~~Build the `audio-tools` MCP server (stdio)~~ — **done** (`mcp/`, 5 tools, registered as `audio`; restart/new session for pickup).
2. Wire a loudness + true-peak gate into the distribution lane (before DistroKid) — `release_validate` is the building block.
3. GPU variant of the ML env only if windowed batch separation is ever wanted.
