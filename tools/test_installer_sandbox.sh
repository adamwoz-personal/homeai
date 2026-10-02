#!/usr/bin/env bash
# End-to-end test of install.sh: a FRESH install of the current working tree
# into a throwaway HOME, then an idempotency re-run, an abort run, and the
# voice-tool security probe against the newly created agent.
#
#   tools/test_installer_sandbox.sh            # full run (~5-10 min first time)
#   tools/test_installer_sandbox.sh --keep     # leave the sandbox for inspection
#
# What is real: the venv, the whisper.cpp clone/build, model downloads, the
# ZeroClaw config written from nothing, a live agent reply through Ollama.
# What is shared with the host: Ollama itself (llama31-voice is reused, not
# recreated, because it already has the right num_ctx) and uv's download cache.
# --no-service is always passed: a fake HOME must not touch the real systemd
# user manager.
#
# Exit status: 0 only if every check passes.
set -euo pipefail

SRC="$(cd "$(dirname "$(readlink -f "$0")")/.." && pwd)"
KEEP=0
[[ "${1:-}" == "--keep" ]] && KEEP=1

REAL_HOME="$HOME"
# Not /tmp: it is often a RAM-backed tmpfs, and a full install is ~7 GB.
mkdir -p "$REAL_HOME/.cache"
SANDBOX="$(mktemp -d "$REAL_HOME/.cache/homeai-install-XXXXXX")"
cleanup() {
    if [[ $KEEP -eq 1 ]]; then echo "sandbox kept: $SANDBOX"; else rm -rf "$SANDBOX"; fi
}
trap cleanup EXIT

FAILS=0
check() {  # check "description" command...
    local desc="$1"; shift
    if "$@" >/dev/null 2>&1; then echo "PASS  $desc"; else echo "FAIL  $desc"; FAILS=$((FAILS + 1)); fi
}

rsync -a --exclude .venv --exclude vendor --exclude .env --exclude .git \
    --exclude __pycache__ --exclude .pytest_cache "$SRC/" "$SANDBOX/repo/"
mkdir -p "$SANDBOX/home"
export HOME="$SANDBOX/home"
export UV_CACHE_DIR="$REAL_HOME/.cache/uv"
cd "$SANDBOX/repo"

echo "== run 1: fresh install (log: $SANDBOX/run1.log)"
set +e
./install.sh --no-service > "$SANDBOX/run1.log" 2>&1
RC1=$?
set -e
tail -25 "$SANDBOX/run1.log"
check "fresh install exits 0" test "$RC1" -eq 0
check "zeroclaw config created" test -f "$HOME/.zeroclaw/config.toml"
check "zeroclaw config is mode 600" test "$(stat -c %a "$HOME/.zeroclaw/config.toml")" = 600
check "persona SOUL.md installed" cmp -s deploy/zeroclaw/SOUL.md "$HOME/.zeroclaw/agents/jarvis/workspace/SOUL.md"
check "persona AGENTS.md installed" cmp -s deploy/zeroclaw/AGENTS.md "$HOME/.zeroclaw/agents/jarvis/workspace/AGENTS.md"
check ".env names the agent" grep -x HOMEAI_AGENT_NAME=jarvis .env
check ".env is mode 600" test "$(stat -c %a .env)" = 600
check "whisper-cli built" test -x vendor/whisper.cpp/build-blas/bin/whisper-cli -o -x vendor/whisper.cpp/build/bin/whisper-cli
check "piper voice downloaded" test -s vendor/piper/en_US-lessac-medium.onnx
check "agent replied during verify" grep -q "agent replied:" "$SANDBOX/run1.log"
check "daemon --check passed" grep -q "check: OK" "$SANDBOX/run1.log"
check "agent answered 7+5 correctly" bash -c "! grep -q 'did not answer 7+5' '$SANDBOX/run1.log'"

echo "== run 2: re-run must change nothing"
cp "$HOME/.zeroclaw/config.toml" "$SANDBOX/config.before"
cp .env "$SANDBOX/env.before"
./install.sh --no-service --skip-verify > "$SANDBOX/run2.log" 2>&1 || true
# The only command a re-run may execute is the (no-op) dependency sync.
CHANGES="$(grep '^   \$ ' "$SANDBOX/run2.log" | grep -v 'pip install' || true)"
[[ -n "$CHANGES" ]] && echo "$CHANGES"
check "re-run executes no changing commands" test -z "$CHANGES"
check "re-run leaves zeroclaw config identical" cmp -s "$SANDBOX/config.before" "$HOME/.zeroclaw/config.toml"
check "re-run leaves .env identical" cmp -s "$SANDBOX/env.before" .env
check "re-run makes no config backup" test -z "$(ls "$HOME/.zeroclaw" | grep '\.bak\.' || true)"

echo "== run 3: an under-powered machine must be refused"
cat > "$SANDBOX/tiny.json" <<'EOF'
{"os": "Linux", "arch": "x86_64", "python": [3, 12], "cpu_cores": 4,
 "ram_total_gb": 7.6, "ram_available_gb": 5.0, "disk_free_gb": 100,
 "gpus": [], "capture_devices": 1, "playback_devices": 1, "systemd_user": true,
 "commands": {"git": true, "curl": true, "cmake": true, "make": true,
              "ollama": true, "zeroclaw": true, "compiler": true}}
EOF
set +e
HOMEAI_ASSESS_FACTS="$SANDBOX/tiny.json" ./install.sh --no-service > "$SANDBOX/run3.log" 2>&1
RC3=$?
set -e
check "under-powered: exits non-zero" test "$RC3" -ne 0
check "under-powered: says INSTALL ABORTED" grep -q "INSTALL ABORTED" "$SANDBOX/run3.log"
check "under-powered: explains resources" grep -q "Not enough resources" "$SANDBOX/run3.log"
check "under-powered: stops before step 2" bash -c "! grep -q '== 2\.' '$SANDBOX/run3.log'"

echo "== security: the new agent must not be able to act on the machine"
set +e
"$SANDBOX/repo/.venv/bin/python" tools/probe_voice_tools.py --agent jarvis \
    --config-dir "$HOME/.zeroclaw" > "$SANDBOX/probe.log" 2>&1
RC4=$?
set -e
tail -8 "$SANDBOX/probe.log"
check "voice tool probe: no side effects got through" test "$RC4" -eq 0

echo
if [[ $FAILS -eq 0 ]]; then echo "ALL CHECKS PASSED"; else echo "$FAILS CHECK(S) FAILED (logs in $SANDBOX; re-run with --keep)"; fi
exit $(( FAILS > 0 ))
