#!/usr/bin/env bash
# Can art mode work from inside homeai.service's sandbox? (AVATAR_PLAN 6a step 8)
#
# Why: the daemon runs with ProtectSystem=strict and ProtectHome=read-only, and
# art mode needs it to run systemctl/systemd-run --user and write to
# ~/Pictures/homeai. This runs those three things under the same sandbox
# settings (read from the installed unit), without touching any real service:
# it only asks whether homeai is active and starts a no-op transient unit.
#
# Usage: tools/art/probe_daemon_sandbox.sh
set -u
UNIT=~/.config/systemd/user/homeai.service
RW=$(grep '^ReadWritePaths=' "$UNIT" | cut -d= -f2- | sed "s#%h#$HOME#g")
props=(-p ProtectSystem=strict -p ProtectHome=read-only -p NoNewPrivileges=true -p "ReadWritePaths=$RW")
systemd-run --user --wait --quiet --collect "${props[@]}" --unit=homeai-sandbox-probe \
  bash -c 'set -e
    touch "$HOME/Pictures/homeai/.sandbox-probe" && rm "$HOME/Pictures/homeai/.sandbox-probe" && echo "write Pictures: ok"
    systemctl --user is-active --quiet homeai.service && echo "systemctl --user: ok"
    systemd-run --user --collect --quiet --wait --unit=homeai-sandbox-probe-child true && echo "systemd-run --user: ok"
    if touch "$HOME/.sandbox-probe" 2>/dev/null; then rm "$HOME/.sandbox-probe"; echo "home read-only: NOT enforced (see ROUTING_GUIDE 2026-11-07)"; else echo "home read-only: ok"; fi'
journalctl --user -u homeai-sandbox-probe --since "-1min" --no-pager -o cat | grep -E "ok|NOT enforced|denied|ailed"
