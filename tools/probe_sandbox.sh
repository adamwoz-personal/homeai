#!/usr/bin/env bash
# Does PrivateUsers=yes make homeai.service's sandbox real without breaking Jarvis? (SANDBOX_PLAN step 1)
#
# Why: homeai.service asks for ProtectSystem=strict and ProtectHome=read-only, but a
# user-manager unit can't build the mount namespace unless it also has PrivateUsers=yes,
# so none of it is applied today (ROUTING_GUIDE 2026-11-07). This runs every check the
# plan lists inside a throwaway transient unit with the installed unit's sandbox
# settings, once as today ("current") and once with PrivateUsers=yes ("private").
# The live homeai.service is never touched: the checks only read, list, open each sound
# card's controls (not the mic: Jarvis holds it, and a recording blocked on it 2026-11-08),
# transcribe a bench clip, and start a no-op transient unit.
#
# Writes ~/homeai-bench/sandbox-<time>/{current,private}.txt and prints both.
#
# Run as root (sudo) it instead tries the fix Adam chose (SANDBOX_PLAN option A): the same
# checks in a transient *system* unit running as the user (User=adam), with the user
# session's bus and runtime dir, writing ~/homeai-bench/sandbox-<time>/system.txt.
#
# Usage: tools/probe_sandbox.sh          (user units: current and PrivateUsers=yes)
#        sudo tools/probe_sandbox.sh     (system unit, User=adam)
set -u
REPO=$(cd "$(dirname "$0")/.." && pwd)
OWNER=$(stat -c %U "$REPO")
OWNER_UID=$(id -u "$OWNER")
HOME=$(getent passwd "$OWNER" | cut -d: -f6)
UNIT=$HOME/.config/systemd/user/homeai.service
RW=$(grep '^ReadWritePaths=' "$UNIT" | cut -d= -f2- | sed "s#%h#$HOME#g")
OUT=$HOME/homeai-bench/sandbox-$(date +%Y%m%d-%H%M%S)
mkdir -p "$OUT"
CLIP=$(ls "$HOME"/homeai-bench/voices/*.wav 2>/dev/null | head -1)
WHISPER="$REPO/vendor/whisper.cpp/build-blas/bin/whisper-cli"
MODEL="$REPO/vendor/whisper.cpp/models/ggml-small.en.bin"

checks=$(cat <<'EOF'
check() { if eval "$2" >/dev/null 2>&1; then echo "ok    $1"; else echo "FAIL  $1"; fi; }
cd "$REPO"
echo "ok    info: uid_map $(tr -s ' ' < /proc/self/uid_map | tr '\n' ';')"
echo "ok    info: user ns $(readlink /proc/self/ns/user), mount ns $(readlink /proc/self/ns/mnt)"
echo "ok    info: $HOME mounted $(findmnt -no OPTIONS --target "$HOME" | cut -c1-40)"
check "home mounted read-only"       'findmnt -no OPTIONS --target "$HOME" | grep -q "^ro"'
check "home is read-only"            '! touch "$HOME/.sandbox-probe"'; rm -f "$HOME/.sandbox-probe"
check "write the repo"               'touch "$REPO/.sandbox-probe" && rm "$REPO/.sandbox-probe"'
check "write ~/.zeroclaw"            'touch "$HOME/.zeroclaw/.sandbox-probe" && rm "$HOME/.zeroclaw/.sandbox-probe"'
check "write ~/.local/share/homeai"  'touch "$HOME/.local/share/homeai/.sandbox-probe" && rm "$HOME/.local/share/homeai/.sandbox-probe"'
check "write ~/Pictures/homeai"      'touch "$HOME/Pictures/homeai/.sandbox-probe" && rm "$HOME/Pictures/homeai/.sandbox-probe"'
check "write /tmp"                   'touch /tmp/.sandbox-probe && rm /tmp/.sandbox-probe'
check "write the event bus dir"      'touch "$XDG_RUNTIME_DIR/homeai/.sandbox-probe" && rm "$XDG_RUNTIME_DIR/homeai/.sandbox-probe"'
check "list sound cards"             'arecord -l | grep -q card'
check "open every sound card"        'for c in /dev/snd/controlC*; do amixer -c "${c##*C}" info || exit 1; done'
check "open the GPU (/dev/kfd)"      'exec 3<>/dev/kfd; exec 3>&-'
check "open the GPU render node"     'exec 3<>/dev/dri/renderD128; exec 3>&-'
check "whisper transcribes a clip"   '[ -n "$CLIP" ] && "$WHISPER" -m "$MODEL" -f "$CLIP" -np -nt | grep -q "[a-z]"'
check "Ollama on localhost"          'curl -sf --max-time 5 http://127.0.0.1:11434/api/ps'
check "Postgres (memory store)"      'pg_isready -q'
check "systemctl --user"             'systemctl --user is-active --quiet homeai.service'
check "systemd-run --user"           'systemd-run --user --collect --quiet --wait --unit=homeai-sandbox-probe-child true'
check "file owner still adam"        '[ "$(stat -c %U "$REPO/README.md")" = "$(id -un)" ]'
EOF
)

run() {   # name, extra properties...
  local name=$1 start; shift
  start=$(date +%s)
  systemd-run --user --wait --quiet --collect --unit="homeai-sandbox-probe-$name" \
    -p RuntimeMaxSec=240 -p ProtectSystem=strict -p ProtectHome=read-only -p NoNewPrivileges=true -p LockPersonality=true \
    -p "ReadWritePaths=$RW" -p "WorkingDirectory=$REPO" "$@" \
    -E "REPO=$REPO" -E "CLIP=$CLIP" -E "WHISPER=$WHISPER" -E "MODEL=$MODEL" \
    bash "$OUT/checks.sh" > /dev/null 2>&1 || true
  # systemd-run --wait doesn't pass stdout through for a service; read it from the journal
  journalctl --user -u "homeai-sandbox-probe-$name" --since "@$start" --no-pager -o cat \
    | grep -E "^(ok|FAIL) " > "$OUT/$name.txt"
  echo "== $name"; cat "$OUT/$name.txt"
}

echo "$checks" > "$OUT/checks.sh"     # run from a file: systemd would expand ${...} on a command line
if [ "$(id -u)" -eq 0 ]; then
  RUN=/run/user/$OWNER_UID
  start=$(date +%s)
  systemd-run --wait --quiet --collect --unit=homeai-sandbox-probe-system --uid="$OWNER" --gid="$OWNER" \
    -p RuntimeMaxSec=240 -p ProtectSystem=strict -p ProtectHome=read-only -p NoNewPrivileges=true \
    -p LockPersonality=true -p "ReadWritePaths=$RW $RUN/homeai" -p "WorkingDirectory=$REPO" \
    -p SupplementaryGroups=audio -p SupplementaryGroups=video -p SupplementaryGroups=render \
    -E "HOME=$HOME" -E "USER=$OWNER" -E "XDG_RUNTIME_DIR=$RUN" -E "DBUS_SESSION_BUS_ADDRESS=unix:path=$RUN/bus" \
    -E "REPO=$REPO" -E "CLIP=$CLIP" -E "WHISPER=$WHISPER" -E "MODEL=$MODEL" \
    bash "$OUT/checks.sh" > /dev/null 2>&1 || true
  journalctl -u homeai-sandbox-probe-system --since "@$start" --no-pager -o cat | grep -E "^(ok|FAIL) " > "$OUT/system.txt"
  chown -R "$OWNER:$OWNER" "$OUT"
  echo "== system (User=$OWNER)"; cat "$OUT/system.txt"
else
  run current
  run private -p PrivateUsers=yes
fi
echo "$OUT"
