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
#   PRESET=qwen36 tools/bench_kv_configs.sh "KV=q8_0 NCPUMOE=10"   # another llm-serve.sh preset
#
# Every config also reports where Jarvis's model sits: a coder that pushes it
# partly onto the CPU fails the voice-mode requirement, however fast it is.
#
# Results append to ~/homeai-bench/kv.jsonl (label = config string).
set -u
cd "$(dirname "$0")/.."
PY="${PY:-.venv/bin/python}"
OUT="${OUT:-$HOME/homeai-bench/kv.jsonl}"
URL=http://127.0.0.1:8080
LOG=$(mktemp /tmp/bench-kv.XXXXXX.log)

if [ $# -gt 0 ]; then
  CONFIGS=("$@")
else
  CONFIGS=("KV=q8_0 NCPUMOE=6" "KV=q4_0 NCPUMOE=0" "KV=q4_0 NCPUMOE=4" "KV=q8_0 NCPUMOE=10")
fi

server_pid=""
PRESET="${PRESET:-qwen}"
# Jarvis's model from ZeroClaw's config (homeai/voice_model.py), never a copy that could disagree.
VOICE_MODEL="${VOICE_MODEL:-$($PY -m homeai.voice_model 2>/dev/null || echo qwen35-voice)}"
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
  # The biggest card's own counter (homeai/gpu_info.py reads the same files).
  local best=0 used=0 total f
  for f in /sys/class/drm/card*/device/mem_info_vram_total; do
    total=$(cat "$f" 2>/dev/null || echo 0)
    if [ "$total" -gt "$best" ]; then best=$total; used=$(cat "${f%_total}_used" 2>/dev/null || echo 0); fi
  done
  awk -v u="$used" 'BEGIN {printf "%.2f", u/1073741824}'
}

voice_where() {
  ollama ps 2>/dev/null | awk -v m="^$VOICE_MODEL" '$1 ~ m {for (i=1;i<=NF;i++) if ($i ~ /GPU|CPU/) {print $(i-1), $i; exit}}'
}

systemctl --user stop llama-server.service
if [ "${UNLOAD_VOICE:-0}" = 1 ]; then
  echo "unloading $VOICE_MODEL for this run"
  ollama stop "$VOICE_MODEL"
fi
for cfg in "${CONFIGS[@]}"; do
  echo "=== $cfg"
  env $cfg "$HOME/models/llm-serve.sh" "$PRESET" >"$LOG" 2>&1 &
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
    $PY tools/bench_tokps.py --label "$PRESET $cfg" --out "$OUT" | tail -1 | sed 's/^/  /'
    where="$(voice_where)"
    echo "  $VOICE_MODEL: ${where:-not loaded}"
    if [ "${UNLOAD_VOICE:-0}" != 1 ] && [[ "$where" != *"100% GPU"* ]]; then
      echo "  WARNING: $VOICE_MODEL is not wholly on the GPU with this config: fails voice mode"
    fi
  fi
  kill "$server_pid" 2>/dev/null; wait "$server_pid" 2>/dev/null; server_pid=""
  sleep 2
done
rm -f "$LOG"
