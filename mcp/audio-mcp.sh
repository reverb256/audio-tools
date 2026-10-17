#!/usr/bin/env bash
# audio MCP server wrapper — stdio facade over the audio-tools envs.
# (1) PYTHONPATH is cleared: a caller's site-packages (e.g. Hermes's own) must not
#     shadow envs/server — this exact shadowing broke gdrive-mcp on its first build.
# (2) Root is resolved relative to this script, so the repo can move.
# (3) Extra argv is deliberately ignored (hermes mcp add can leak flags into args).
set -euo pipefail
unset PYTHONPATH
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
exec "$ROOT/envs/server/bin/python" "$ROOT/mcp/audio_mcp_server.py"
