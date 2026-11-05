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
- `djmix/`     — cue-sheet DJ-mix renderer (see the djmix section below)
- `mixgen/`    — plan-driven pipeline: generate takes → QA → master → djmix mix
- `examples/`  — example cue sheets and mix plans (`lightworld-relay.plan.yaml`)
- `jobs/`       — stem job records / logs / outputs (runtime; gitignored)
- `reference/`  — shallow clones for design reference only (third-party repos; never run)

Environment rebuild (venvs are gitignored):
```bash
uv venv --python 3.12 envs/cpu    && uv pip install --python envs/cpu/bin/python    librosa pyloudnorm soundfile ffmpeg-normalize scipy pyyaml pytest
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

## djmix — beat-matched DJ mixes from a cue sheet (`djmix/`)

CPU-only, free tools only: librosa (tempo, beat grid, downbeat, key; madmom /
`keyfinder-cli` are used automatically when installed), rubberband CLI (pitch-preserving
time-stretch; falls back to librosa's phase vocoder), scipy (EQ / filters), ffmpeg
(EBU R128 measurement + 4x-oversampled true-peak limiter).

What it does, per cue sheet (`examples/dhtrance-cue.example.yaml`):
1. **Analyse** each WAV: BPM (fine comb fit, 0.01 BPM), sample-accurate beat grid,
   downbeat (strongest low-end onsets), key + Camelot code.
2. **Plan** an output tempo map in bars. Track *n+1* starts `mix_in_bars` before track *n*
   ends; every mix point sits on a 16- or 32-bar phrase boundary. With `ramp: true` the tempo
   glides from the old target BPM to the new one across the blend.
3. **Stretch** each track with one rubberband call and a per-beat time map, so every source
   beat lands on its output beat (also through ramps); rubberband's R3 delay is calibrated
   and removed.
4. **Transitions**: `long_blend` (equal-power), `eq_bass_swap` (highs fade in, one-beat bass
   swap on the phrase midpoint, zero-phase crossover so nothing drifts), `filter_sweep`
   (outgoing high-pass sweep, incoming low-pass opening), `cut` (hard cut on the bar).
5. **Render** one continuous 48 kHz / 24-bit stereo WAV plus a tracklist with timestamps;
   loudness two-pass to `target_lufs` (default −9 LUFS integrated) with a −1 dBTP ceiling.

Cue sheet keys — top level: `root`, `output`, `tracklist`, `phrase_bars` (16|32),
`target_bpm` (default per track), `target_lufs`, `target_tp`, `loudnorm`, `sample_rate`.
Per track: `file`, `title`, `target_bpm`, `start_bar` (bars to skip after the first
downbeat), `play_bars`, `mix_in_bars`, `transition`, `ramp`, `gain_db`, and overrides
`bpm` / `first_downbeat` if detection is ever wrong (check the `--dry-run` output).

```bash
# plan only: analysis, bar map, timestamps, warnings; renders nothing
envs/cpu/bin/python -m djmix examples/dhtrance-cue.example.yaml --dry-run
# render (paths relative to /data/media/archive unless --root / $DJMIX_ROOT)
envs/cpu/bin/python -m djmix music/mixes/dh-trance-01/cue.yaml --json
# on the cluster (music-tools image with djmix, see below), from nexus:
~/fleet-tools/cpu-job generic -- "python -m djmix /data/media/archive/music/mixes/dh-trance-01/cue.yaml --json"
```

Tests: `envs/cpu/bin/python -m pytest -q tests` renders two synthetic click tracks
(122 and 126 BPM) into a 124 BPM blend and asserts every beat of both tracks lands within
10 ms of the shared grid through the blend, for both stretch engines; plus a ramped
bass-swap and a full 48k/24-bit/loudness render. A 60 min render needs ~3 GB RAM and runs
single-threaded, one track at a time.

## mixgen — generate → QA → master → mix (`mixgen/`)

One plan YAML drives a whole mix: N takes per track from a generation backend, QA
that picks the best take, loudnorm mastering, and a djmix cue sheet/render with
tracklist, MP3 and YouTube chapters. Every stage is resumable (an existing output
is skipped unless `--force`) and every write is atomic (temp path + rename).

```bash
python -m mixgen examples/lightworld-relay.plan.yaml --root /music --dry-run
python -m mixgen examples/lightworld-relay.plan.yaml --root /music --backend fake
python -m mixgen plan.yaml --stages qa,master,mix --force     # rerun a subset
```

The console entry `mixgen` (pyproject `[project.scripts]`) is equivalent when the
package is installed.

**Plan format** (`examples/lightworld-relay.plan.yaml` is a full 12-track plan):

```yaml
title: My Mix
slug: my-mix                    # optional; derived from title
output: mixes/my-mix            # dir under --root (default /music)
sample_rate: 48000
phrase_bars: 16                 # 16 or 32
takes: 3                        # default takes per track
backend: acestep                # acestep | fake (CLI --backend overrides)
acestep:
  wrapper: gpu-music-job          # locked gpu-music-job contract, no token/auth
  poll_interval: 30              # seconds between job-result.json checks
  # dry_run: true                # pass --dry-run to the wrapper job
tracks:
  - title: Track One
    slug: track-one             # optional; derived from title
    style: "deep house, 120 BPM, A minor. ..."   # backend prompt, verbatim
    bpm: 120
    key: 8A                     # Camelot
    duration: "5:20"            # mm:ss, hh:mm:ss or seconds
    instrumental: true          # or: lyrics: "..." / lyrics: "@file"
    takes: 3                    # one job per track: --takes 3 (batch)
    job: {kind: gpu, host: zephyr}   # kind: cpu|gpu, host: nexus|zephyr
    play_bars: 160              # optional djmix cue fields
    mix_in_bars: 32             # default 32 (0 for the first track / a cut)
    transition: long_blend      # long_blend | eq_bass_swap | filter_sweep | cut
    ramp: true                  # default true
```

Any other job host (e.g. `forge`) or job kind is rejected at validation, as is
`gpu@nexus`/`gpu@sentry` (GPU jobs target `gpu@zephyr` only), or any path that
resolves outside `--root` (`..`, absolute paths, symlinked parents).
Per-track paths are derived: job output is a flat directory
`gen/<mix>/<track>/` (FLAC + `job-request.json`/`job-result.json`),
takes are copied there as `take-N.flac`, and QA/mastered files live under the
output dir.

**Stages** (`--stages gen,qa,master,stems,mix`, canonical order):

1. `gen` — one `gpu-music-job acestep --takes N` per track (a batch, not one job
   per take) writing a flat directory `gen/<mix>/<track>/` on the PVC, then each
   take is copied out of `job-result.json -> audios[].path`. `fake` synthesises
   a WAV (kick on the beat at the target BPM, a tonic-triad pad in the target key,
   16-bar drums-only intro/outro) for tests and offline runs.
2. `qa` — per take: tempo (librosa + comb fit, half/double folded), key (chroma +
   Krumhansl → Camelot, measured on the harmonic body), length, drums-only
   intro/outro ≥ 16 bars (HPSS harmonic vs percussive energy), clipping and
   silence. Weighted score; best take becomes the pick. Writes
   `qa/<track>/report.{json,md}` and `qa/summary.json`.
3. `master` — each pick through djmix's two-pass loudnorm chain (default
   −9 LUFS / −1 dBTP) into `mastered/<track>.wav`.
4. `stems` — opt-in only (never runs by default): `gpu-music-job demucs
   --in-relpath ... --out-relpath <mix>/<track>-stems [--model htdemucs]` on
   each QA pick. Enable with a top-level `stems: {enabled: true}` plan block and
   `--stages ...,stems`.
5. `mix` — writes a djmix cue sheet from the plan, renders the WAV with djmix
   (`--root` is passed through), then a 320k MP3, a timestamped tracklist and
   YouTube chapter text (first chapter at 0:00). Timestamps are bar/BPM/overlap
   maths, including tempo ramps and the `cut` case.


`--dry-run` prints every stage and, per track/take, the input, output and
skip/run decision. It never generates, renders or calls a backend, and works with
no audio present.

**ACE-Step backend config.** Generation runs through the locked `gpu-music-job`
contract on the default `gpu@zephyr` target; there is no token or auth. Submit
argv is `gpu-music-job acestep --prompt TEXT [--lyrics TEXT|@file]
[--instrumental] [--duration SEC] [--bpm N] [--key TEXT] [--seed N]
[--takes N] [--out-relpath REL] [--dry-run]`, one job per track with all takes
(`--takes N`). Output is a flat directory `/music/gen/<out-relpath>/` on the PVC
(host `/data/shared/music/gen/<out-relpath>/`), FLAC by default, plus
`job-request.json` and `job-result.json`; takes come from
`job-result.json -> audios[].path`. A job is done when `job-result.json` exists,
and `gpu-music-job status <job-name>` (name echoed on create) reports progress.
Non-zero wrapper exits (0/2/3) are surfaced as errors.

The field→flag mapping lives only in `mixgen/backends/acestep_flags.py`
(`FIELD_FLAGS` + `build_argv()`; `build_demucs_argv()` for the opt-in stems
step). Configuration comes from the plan's `acestep:` block or environment (env
wins): `MIXGEN_ACESTEP_WRAPPER` (default `gpu-music-job`),
`MIXGEN_ACESTEP_COMMAND` (full argv template with `{prompt} {lyrics}
{instrumental} {duration} {bpm} {key} {seed} {takes} {out_relpath}`),
`MIXGEN_ACESTEP_POLL_INTERVAL`, `MIXGEN_ACESTEP_DRY_RUN`.

Tests: `python -m pytest -q tests/test_mixgen.py` (synthetic audio; QA ranking,
Camelot conversion, cue bar/timestamp/chapter maths, plan validation, resumability,
acestep argv + job-result collection with a fake submitter, and a dry run of the
example plan).

## Next
1. ~~Build the `audio-tools` MCP server (stdio)~~ — **done** (`mcp/`, 5 tools, registered as `audio`; restart/new session for pickup).
2. Wire a loudness + true-peak gate into the distribution lane (before DistroKid) — `release_validate` is the building block.
3. GPU variant of the ML env only if windowed batch separation is ever wanted.
