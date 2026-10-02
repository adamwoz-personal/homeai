#!/usr/bin/env bash
# Exercise `homeai-mode memory` against the real zeroclaw binary, on a COPY of
# ~/.zeroclaw in a temp dir. The live config, brain.db and daemon are untouched.
#   tools/test_memory_privacy_sandbox.sh [SOURCE_CONFIG_DIR]
set -euo pipefail
SRC=${1:-$HOME/.zeroclaw}
REPO=$(cd "$(dirname "$0")/.." && pwd)
MODE="$REPO/bin/homeai-mode"
TMP=$(mktemp -d)
trap 'rm -rf "$TMP"' EXIT
pass=0; fail=0
check() { if eval "$2"; then echo "PASS $1"; pass=$((pass+1)); else echo "FAIL $1"; fail=$((fail+1)); fi; }

rsync -a --exclude 'config.toml.bak.*' --exclude '*.wav' "$SRC/" "$TMP/zc/"
ZC="$TMP/zc"; DB="$ZC/data/memory/brain.db"
run() { "$MODE" memory --config-dir "$ZC" "$@"; }
count() { zeroclaw memory stats --config-dir "$ZC" | awk -v c="$1" '$1==c{print $2}'; }
live_before=$(sha256sum "$SRC/config.toml" | cut -d' ' -f1)

conv0=$(count conversation); conv0=${conv0:-0}
core0=$(count core); core0=${core0:-0}
# A saved conversation's key (a UUID) appears nowhere else, so if it is still
# on disk afterwards, the text was not really deleted.
needle=$(sqlite3 "$DB" "select key from memories where category='conversation' limit 1")
echo "sandbox: $conv0 conversation rows, $core0 core rows, needle '${needle:-none}'"

check "status runs" 'run >/dev/null'
check "off" 'run off | grep -q "saving:    off"'
check "off persisted in config" '[ "$(zeroclaw config get --config-dir "$ZC" memory.auto_save)" = false ]'
check "retention 3" 'run retention 3 | grep -q "retention: 3 days"'
check "retention 0 means forever" 'run retention 0 | grep -q "retention: forever"'
check "negative retention rejected" '! run retention -1 >/dev/null 2>&1'
check "on" 'run on | grep -q "saving:    ON"'
check "purge without --yes and no answer deletes nothing" \
  'echo n | run purge >/dev/null && [ "$(count conversation)" = "$conv0" ]'
check "purge --yes" 'run purge --yes >/dev/null'
check "no conversation rows left" '[ -z "$(count conversation)" ]'
check "core rows kept" '[ "${core0}" = "$(count core || true)" ] || [ "$core0" = 0 ]'
if [ -n "${needle:-}" ]; then
  check "conversation text gone from disk" '! grep -rqa -- "$needle" "$ZC/data/memory"'
fi
check "live config untouched" '[ "$(sha256sum "$SRC/config.toml" | cut -d" " -f1)" = "$live_before" ]'
echo "$pass passed, $fail failed"
[ "$fail" = 0 ]
