#!/usr/bin/env bash
# Move the memory backup and maintenance units from user units to system units, or back.
# (SANDBOX_PLAN "same review for the memory units", agreed by Adam 2026-11-12.)
#
# Why: tools/probe_unit_sandbox.sh showed their declared ProtectSystem/ProtectHome
# do nothing as user units on this Ubuntu (no mount namespace for user units).
# As system units with User=<you> the sandbox is enforced, as for homeai.service.
#
#   sudo tools/install_memory_system_units.sh            # switch
#   sudo tools/install_memory_system_units.sh --undo     # back to the user units
#   sudo tools/install_memory_system_units.sh --dry-run  # show every step, change nothing
#
# The user timers are stopped and their files moved aside first, so a backup
# never runs twice. Then the system backup runs once as proof (a dump and a
# restore check inside the sandbox); if it fails, the switch is undone.
#
# Test hooks: HOMEAI_UNIT_DIR, HOMEAI_OWNER_HOME, HOMEAI_SKIP_ROOT_CHECK=1
# (tests/test_install_memory_system_units.py).
set -euo pipefail

REPO="$(cd "$(dirname "$0")/.." && pwd)"
UNIT_DIR="${HOMEAI_UNIT_DIR:-/etc/systemd/system}"
NAMES=(homeai-memory-backup homeai-memory-maintain)
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
USER_DIR="$OWNER_HOME/.config/systemd/user"

as_owner() {   # systemctl --user for the owner, from root
    run runuser -u "$OWNER" -- env XDG_RUNTIME_DIR="/run/user/$OWNER_UID" \
        DBUS_SESSION_BUS_ADDRESS="unix:path=/run/user/$OWNER_UID/bus" "$@"
}

timers() { local n out=(); for n in "${NAMES[@]}"; do out+=("$n.timer"); done; echo "${out[@]}"; }

to_user() {
    # shellcheck disable=SC2046
    run systemctl disable --now $(timers) || true
    for n in "${NAMES[@]}"; do run rm -f "$UNIT_DIR/$n.service" "$UNIT_DIR/$n.timer"; done
    run systemctl daemon-reload
    for n in "${NAMES[@]}"; do
        for kind in service timer; do
            f="$USER_DIR/$n.$kind"
            if [[ -f "$f.moved-to-system" ]]; then run mv "$f.moved-to-system" "$f"; fi
        done
    done
    as_owner systemctl --user daemon-reload
    # shellcheck disable=SC2046
    as_owner systemctl --user enable --now $(timers)
}

if [[ $UNDO -eq 1 ]]; then
    to_user
    echo "back to the user units: systemctl --user list-timers 'homeai-memory-*'"
    exit 0
fi

WORK="$(mktemp -d)"
trap 'rm -rf "$WORK"' EXIT
for n in "${NAMES[@]}"; do
    sed -e "s#@REPO@#$REPO#g" -e "s#@HOME@#$OWNER_HOME#g" -e "s#@USER@#$OWNER#g" \
        "$REPO/deploy/systemd/$n.system.service.in" > "$WORK/$n.service"
    cp "$REPO/deploy/systemd/$n.timer" "$WORK/$n.timer"
    ! grep -n '@[A-Z_]*@' "$WORK/$n.service" || die "template placeholder left unrendered in $n.service"
done

# User timers off and their files out of the way first: one backup at a time.
# shellcheck disable=SC2046
as_owner systemctl --user disable --now $(timers) || true
for n in "${NAMES[@]}"; do
    for kind in service timer; do
        f="$USER_DIR/$n.$kind"
        if [[ -f "$f" ]]; then run mv "$f" "$f.moved-to-system"; fi
    done
done
as_owner systemctl --user daemon-reload

for n in "${NAMES[@]}"; do
    for kind in service timer; do
        cmp -s "$WORK/$n.$kind" "$UNIT_DIR/$n.$kind" 2>/dev/null || run install -D -m 644 "$WORK/$n.$kind" "$UNIT_DIR/$n.$kind"
    done
done
run systemctl daemon-reload
# shellcheck disable=SC2046
run systemctl enable --now $(timers)
[[ $DRY -eq 1 ]] && { echo "(dry run: nothing was changed)"; exit 0; }

# Proof: one real backup inside the enforced sandbox (dump + restore check).
if ! run systemctl start homeai-memory-backup.service; then
    echo "the system backup failed; going back to the user units. See: journalctl -u homeai-memory-backup -n 50" >&2
    to_user
    exit 1
fi
echo "Memory backup and maintenance run as system units. Timers: systemctl list-timers 'homeai-memory-*'"
echo "Logs: journalctl -u homeai-memory-backup   journalctl -u homeai-memory-maintain"
