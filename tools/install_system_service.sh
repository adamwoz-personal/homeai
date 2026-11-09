#!/usr/bin/env bash
# Move Jarvis (homeai.service) from a user unit to a system unit, or back. (SANDBOX_PLAN step 4)
#
# Why: on this Ubuntu a user unit's ProtectHome is silently skipped, so the
# sandbox did nothing (tools/probe_sandbox.sh shows it). A system unit with
# User=<you> gets it enforced. Only Jarvis moves; zeroclaw, the coder, ComfyUI
# and the picture window stay user units.
#
#   sudo tools/install_system_service.sh            # switch to the system unit
#   sudo tools/install_system_service.sh --undo     # back to the user unit
#   sudo tools/install_system_service.sh --dry-run  # show every step, change nothing
#
# Order matters: the user unit is stopped and its file moved aside before the
# system unit starts, so two Jarvises never share the microphone. If the
# system unit doesn't stay up, the switch is undone on its own.
# Needs the polkit rule first (deploy/polkit/50-homeai.rules.in): mode
# switching restarts Jarvis without a password.
#
# Test hooks: HOMEAI_UNIT_DIR, HOMEAI_POLKIT_RULE, HOMEAI_OWNER_HOME, HOMEAI_SKIP_ROOT_CHECK=1,
# HOMEAI_SETTLE_S (tests/test_install_system_service.py).
set -euo pipefail

REPO="$(cd "$(dirname "$0")/.." && pwd)"
UNIT_DIR="${HOMEAI_UNIT_DIR:-/etc/systemd/system}"
UNIT="$UNIT_DIR/homeai.service"
RULE="${HOMEAI_POLKIT_RULE:-/etc/polkit-1/rules.d/50-homeai.rules}"
SETTLE="${HOMEAI_SETTLE_S:-8}"
DRY=0 UNDO=0
for a in "$@"; do
    case "$a" in
        --dry-run) DRY=1 ;;
        --undo) UNDO=1 ;;
        *) echo "unknown option: $a" >&2; exit 2 ;;
    esac
done

die() { echo "ERROR: $*" >&2; exit 1; }
run() { echo "+ $*"; [[ $DRY -eq 1 ]] || "$@"; }

[[ "${HOMEAI_SKIP_ROOT_CHECK:-0}" == 1 || $EUID -eq 0 ]] || die "run with sudo"
OWNER="${SUDO_USER:-}"
[[ -n "$OWNER" && "$OWNER" != root ]] || die "run with sudo from your own account (SUDO_USER is empty or root)"
OWNER_UID="$(id -u "$OWNER")"
OWNER_HOME="${HOMEAI_OWNER_HOME:-$(getent passwd "$OWNER" | cut -d: -f6)}"
ZC_DIR="$OWNER_HOME/.zeroclaw"
USER_UNIT="$OWNER_HOME/.config/systemd/user/homeai.service"
ASIDE="$USER_UNIT.moved-to-system"

as_owner() {   # systemctl --user for the owner, from root
    run runuser -u "$OWNER" -- env XDG_RUNTIME_DIR="/run/user/$OWNER_UID" \
        DBUS_SESSION_BUS_ADDRESS="unix:path=/run/user/$OWNER_UID/bus" "$@"
}

to_user() {
    run systemctl disable --now homeai.service || true
    run rm -f "$UNIT"
    run systemctl daemon-reload
    if [[ -f "$ASIDE" ]]; then run mv "$ASIDE" "$USER_UNIT"; fi
    as_owner systemctl --user daemon-reload
    as_owner systemctl --user enable --now homeai.service
}

if [[ $UNDO -eq 1 ]]; then
    to_user
    echo "back to the user unit: systemctl --user status homeai"
    exit 0
fi

[[ -f "$RULE" ]] || die "install the polkit rule first:
  sed \"s/@USER@/$OWNER/\" $REPO/deploy/polkit/50-homeai.rules.in | sudo tee $RULE"
[[ "$(loginctl show-user "$OWNER" -p Linger --value 2>/dev/null)" == yes ]] || run loginctl enable-linger "$OWNER"

RENDERED="$(mktemp)"
trap 'rm -f "$RENDERED"' EXIT
sed -e "s#@REPO@#$REPO#g" -e "s#@ZEROCLAW_DIR@#$ZC_DIR#g" -e "s#@HOME@#$OWNER_HOME#g" \
    -e "s#@UID@#$OWNER_UID#g" -e "s#@USER@#$OWNER#g" "$REPO/deploy/systemd/homeai.system.service.in" > "$RENDERED"
! grep -n '@[A-Z_]*@' "$RENDERED" || die "template placeholder left unrendered"

# User unit off and out of the way first: one Jarvis on the mic.
as_owner systemctl --user disable --now homeai.service || true
if [[ -f "$USER_UNIT" ]]; then run mv "$USER_UNIT" "$ASIDE"; fi
as_owner systemctl --user daemon-reload

[[ -f "$UNIT" ]] && ! cmp -s "$RENDERED" "$UNIT" && run cp "$UNIT" "$UNIT.bak.$(date +%s)"
run install -D -m 644 "$RENDERED" "$UNIT"
run systemctl daemon-reload
run systemctl enable --now homeai.service
[[ $DRY -eq 1 ]] && { echo "(dry run: nothing was changed)"; exit 0; }
sleep "$SETTLE"
if ! systemctl is-active --quiet homeai.service; then
    echo "homeai.service did not stay up; going back to the user unit. See: journalctl -u homeai -n 50" >&2
    to_user
    exit 1
fi
echo "Jarvis runs as a system unit. Status: systemctl status homeai   Logs: journalctl -u homeai -f"
