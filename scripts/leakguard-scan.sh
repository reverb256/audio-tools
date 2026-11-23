#!/usr/bin/env bash
# leakguard-scan — repo-local secret / state-file scan.
#
# Usage: scripts/leakguard-scan.sh <base-rev> <head-rev>
#          base may be the all-zero SHA (new branch): scans the whole head tree
#        scripts/leakguard-scan.sh --diff < some.diff
#          scans a unified diff from stdin (used by tests and CI)
#
# This is the in-repo floor. The fleet's global leakguard (identity patterns,
# host-local, never committed) still runs from ~/.config/git/hooks on every
# fleet host and chains to .git/hooks/pre-push. This script carries NO
# identity strings, only generic secret shapes and forbidden paths, so it is
# safe in public repos. Exit 1 = findings.
set -euo pipefail

zero=0000000000000000000000000000000000000000
empty_tree=4b825dc642cb6eb9a060e54bf8d69288fbee4904

if [[ "${1:-}" == "--diff" ]]; then
  diff=$(cat)
  label="stdin diff"
else
  base="${1:?base rev}"
  head="${2:?head rev}"
  [[ "$base" == "$zero" ]] && base=$empty_tree
  diff=$(git diff --no-color --unified=0 --diff-filter=ACMR "$base" "$head" --)
  label="$base..$head"
fi

fail=0
report() { echo "leakguard: $*" >&2; fail=1; }

# 1. Forbidden paths (tool/session state, keys, databases, env files).
path_re='(^|/)(\.freebuff|\.sisyphus|\.serena|\.playwright-mcp|\.codebuff|\.cursor|\.claude|\.hermes|\.opencode|\.continue|\.wrangler|playwright-report)/|(^|/)\.aider|(^|/)\.dev\.vars$|(^|/)\.env($|\.)|\.(db|db-wal|db-shm|sqlite[0-9]*|pem|key|p12|pfx)$|(^|/)id_(rsa|ed25519|ecdsa)$'
allow_re='(^|/)\.env\.example$'
paths=$(sed -n 's|^+++ b/||p; s|^Binary files .* and b/\(.*\) differ$|\1|p' <<<"$diff" | sort -u)
while IFS= read -r f; do
  [[ -z "$f" ]] && continue
  if [[ "$f" =~ $path_re ]] && ! [[ "$f" =~ $allow_re ]]; then
    report "forbidden path: $f"
  fi
done <<<"$paths"

# 2. Secret shapes in added lines.
secret_re='-----BEGIN ([A-Z]+ )?PRIVATE KEY-----|gh[pousr]_[A-Za-z0-9]{36,}|github_pat_[A-Za-z0-9_]{60,}|AKIA[0-9A-Z]{16}|xox[baprs]-[A-Za-z0-9-]{10,}|sk-[A-Za-z0-9_-]{32,}|AIza[0-9A-Za-z_-]{35}|glpat-[A-Za-z0-9_-]{20,}'
added=$(grep -E '^\+' <<<"$diff" | grep -vE '^\+\+\+ ' || true)
hits=$(grep -E -o -e "$secret_re" <<<"$added" | cut -c1-6 | sort -u || true)
if [[ -n "$hits" ]]; then
  while IFS= read -r h; do report "secret-shaped string added (prefix '$h…')"; done <<<"$hits"
fi

if [[ "$fail" -ne 0 ]]; then
  echo "leakguard: refused. Remove the finding; never bypass with --no-verify." >&2
  exit 1
fi
echo "leakguard: clean ($label)"
