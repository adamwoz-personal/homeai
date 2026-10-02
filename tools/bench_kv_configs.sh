#!/usr/bin/env bash
# Copyright (c) 2026 Adam Wosotowsky
# SPDX-License-Identifier: BUSL-1.1
#
# Compare llama-server KV-cache / MoE-offload configurations for the coder.
#
# Why: ~/models/llm-serve.sh documents q4_0 + all experts on GPU at ~155
# tok/s versus ~74 for q8_0 + 6 MoE layers on CPU -- measured before Jarvis's
# voice model (~7 GB, pinned in Ollama) shared the card. With it resident,
# the q8_0 setup measured 26.5 tok/s (2026-10-02), so the old numbers no
# longer describe this machine.
#
# For each config: stop llama-server.service, start llm-serve.sh with the
# config's environment, wait for /health, run tools/bench_tokps.py, record
# VRAM in use, stop it. The service is ALWAYS restarted at the end, even on
# Ctrl-C or failure.
#
# Usage:
#   tools/bench_kv_configs.sh                       # default config list
#   UNLOAD_VOICE=1 tools/bench_kv_configs.sh ...    # measure with the GPU to itself
#     (unloads Jarvis's Ollama model for the run and reloads it, pinned, after;
#      Jarvis answers slowly or not at all meanwhile)
#   tools/bench_kv_configs.sh "KV=q4_0 NCPUMOE=4"   # just these configs
#
# Results append to ~/homeai-bench/kv.jsonl (label = config string).
set -u
cd "$(dirname "$0")/.."
PY=.venv/bin/python
OUT="${OUT:-$HOME/homeai-bench/kv.jsonl}"
URL=http://127.0.0.1:8080
LOG=$(mktemp /tmp/bench-kv.XXXXXX.log)

if [ $# -gt 0 ]; then
  CONFIGS=("$@")
else
  CONFIGS=("KV=q8_0 NCPUMOE=6" "KV=q4_0 NCPUMOE=0" "KV=q4_0 NCPUMOE=4" "KV=q8_0 NCPUMOE=10")
fi

server_pid=""
VOICE_MODEL="${VOICE_MODEL:-llama31-voice}"
restore() {
  [ -n "$server_pid" ] && kill "$server_pid" 2>/dev/null && wait "$server_pid" 2>/dev/null
  echo "restoring llama-server.service"
  systemctl --user start llama-server.service
  if [ "${UNLOAD_VOICE:-0}" = 1 ]; then
    # Wait for llama-server to finish allocating first. Measured: a reload
    # issued while it was still starting returned without loading anything,
    # leaving Jarvis with no model.
    for _ in $(seq 1 120); do curl -sf "$URL/health" >/dev/null && break; sleep 1; done
    echo "reloading $VOICE_MODEL (pinned)"
    for _ in 1 2 3; do
      curl -s --max-time 120 http://127.0.0.1:11434/api/generate \
        -d "{\"model\": \"$VOICE_MODEL\", \"keep_alive\": -1}" >/dev/null
      ollama ps | grep -q "^$VOICE_MODEL" && break
      sleep 3
    done
    if ollama ps | grep -q "^$VOICE_MODEL"; then
      echo "$VOICE_MODEL loaded"
    else
      echo "WARNING: $VOICE_MODEL is NOT loaded; Jarvis will load it on the next question" >&2
    fi
  fi
}
trap restore EXIT INT TERM

vram_used_gb() {
  rocm-smi --showmeminfo vram 2>/dev/null \
    | awk '/GPU\[0\].*Used Memory/ {printf "%.1f", $NF/1e9}'
}

systemctl --user stop llama-server.service
if [ "${UNLOAD_VOICE:-0}" = 1 ]; then
  echo "unloading $VOICE_MODEL for this run"
  ollama stop "$VOICE_MODEL"
fi
for cfg in "${CONFIGS[@]}"; do
  echo "=== $cfg"
  env $cfg "$HOME/models/llm-serve.sh" qwen >"$LOG" 2>&1 &
  server_pid=$!
  ok=""
  for _ in $(seq 1 120); do
    if ! kill -0 "$server_pid" 2>/dev/null; then break; fi
    if curl -sf "$URL/health" >/dev/null; then ok=1; break; fi
    sleep 1
  done
  if [ -z "$ok" ]; then
    echo "  failed to start; last log lines:"; tail -5 "$LOG" | sed 's/^/  /'
  else
    echo "  VRAM in use: $(vram_used_gb) GB (incl. Ollama voice model)"
    $PY tools/bench_tokps.py --label "$cfg" --out "$OUT" | tail -1 | sed 's/^/  /'
    ollama ps 2>/dev/null | awk 'NR==2 {print "  ollama: " $1 " " $5 " " $6}'
  fi
  kill "$server_pid" 2>/dev/null; wait "$server_pid" 2>/dev/null; server_pid=""
  sleep 2
done
rm -f "$LOG"
