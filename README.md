# audio-tools — the audio lane's toolchain (phase 1 acquired 2026-10-16)

PURPOSE. Agent-driven audio processing for the music pipeline: QC, DSP, stem
separation, mastering, release validation. Design follows the deep-research
brief (`~/Documents/research/linux-audio-agent-tooling.md`): thin MCP facade
over pinned CLI workers; CPU-first; GPU only via the miner-handoff queue if a
windowed need appears.

## Layout
- `envs/cpu` — python 3.12 (uv): librosa 1.0.0 · pyloudnorm · soundfile · ffmpeg-normalize 1.43.0 · numpy 2.5.4
- `envs/ml`  — python 3.12 (uv): torch 2.14.1+cpu · torchaudio · demucs 4.1.0
- `recipes/` — versioned DSP recipes (to come; recipes are data, not shell strings)
- `reference/` — shallow clones for design reference only: audio-mastering-mcp, stems-mcp, audio-processing-mcp

System tools used: `/usr/bin/ffmpeg` (n9.0.2, loudnorm + ebur128 filters), `SoX_ng 14.8.0.1`, `uv 0.13.1`.

## Proofs (2026-10-16, real file: ~/Music/acestep/final-round_00001.flac — 90 s, 48 kHz)
- **pyloudnorm**: −14.27 LUFS integrated; sample peak **0.00 dBFS**.
- **ffmpeg ebur128**: I −10.6 LUFS, TPK **+0.0 dBTP** → violates the −1 dBTP streaming headroom recommendation. Live example of why the QC gate exists. (pyloudnorm vs ebur128 difference = gating/implementation; read both and inspect the processing mode.)
- **Demucs htdemucs, CPU**: 20 s clip → `vocals.wav` + `no_vocals.wav` in **11.2 s wall (~2× realtime)**. CPU-first stems are viable at this scale — GPU handoff not required for phase 1.

## Usage
```bash
# QC (measure)
envs/cpu/bin/python - <<'PY'
import sys, soundfile as sf, pyloudnorm as pyln
d, sr = sf.read(sys.argv[1]); m = pyln.Meter(sr)
print(m.integrated_loudness(d.mean(axis=1) if d.ndim > 1 else d), "LUFS")
PY
/usr/bin/ffmpeg -i FILE -af ebur128=peak=true -f null -

# Stems
envs/ml/bin/python -m demucs.separate -n htdemucs --two-stems=vocals -d cpu -o OUT FILE
```

## Next
1. Build the `audio-tools` MCP server (stdio): `audio_qc`, `dsp_render` (allowlisted recipes), `stem_submit`, `release_validate`.
2. Wire a loudness + true-peak gate into the distribution lane (before DistroKid).
3. GPU variant of the ML env only if windowed batch separation is ever wanted.
