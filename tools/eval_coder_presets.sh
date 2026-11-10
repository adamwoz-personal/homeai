#!/usr/bin/env bash
# Copyright (c) 2026 Adam Wosotowsky
# SPDX-License-Identifier: BUSL-1.1
#
# Run tools/eval_local_coder.py once per llm-serve.sh preset (LLM_UPGRADE_PLAN step 3).
#
# Why: comparing coders means serving each one in turn on :8080, where the
# builder agent looks, and the eval takes ~75 min per model. This switches
# ~/.config/llm-preset, restarts llama-server.service, waits for /health, runs
# the eval labelled "<preset>-<date>", and ALWAYS puts the original preset back
# and restarts the coder at the end, even on failure or Ctrl-C. Jarvis is not
# touched; each preset's defaults must already keep Jarvis's model on the GPU
# (tools/bench_kv_configs.sh checks that).
#
# Usage:
#   tools/eval_coder_presets.sh qwen36 qwen        # new candidate, then today's coder
#   EVAL_ARGS="--task planted-bug" tools/eval_coder_presets.sh qwen36
#
# Results: ~/homeai-bench/coder-eval.jsonl (eval_local_coder.py's own file).
set -u
cd "$(dirname "$0")/.."
PY="${PY:-.venv/bin/python}"
PRESET_FILE="${PRESET_FILE:-$HOME/.config/llm-preset}"
URL="${URL:-http://127.0.0.1:8080}"
STAMP="$(date +%Y%m%d)"

[ $# -gt 0 ] || { echo "usage: $0 PRESET [PRESET...]" >&2; exit 2; }
original="$(cat "$PRESET_FILE" 2>/dev/null || echo qwen)"

serve() {
  printf '%s\n' "$1" > "$PRESET_FILE"
  systemctl --user restart llama-server.service
  for _ in $(seq 1 180); do curl -sf "$URL/health" >/dev/null && return 0; sleep 1; done
  return 1
}

restore() {
  echo "restoring coder preset: $original"
  serve "$original" || echo "WARNING: llama-server did not come back with $original: systemctl --user status llama-server" >&2
}
trap restore EXIT
trap 'exit 130' INT TERM

status=0
for preset in "$@"; do
  echo "=== $preset"
  if ! serve "$preset"; then
    echo "  $preset did not start; skipped" >&2
    status=1
    continue
  fi
  # shellcheck disable=SC2086
  $PY tools/eval_local_coder.py --label "$preset-$STAMP" ${EVAL_ARGS:-} || status=1
done
exit $status
