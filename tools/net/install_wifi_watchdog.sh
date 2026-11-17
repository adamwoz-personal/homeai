#!/bin/sh
# Install (or update) the Wi-Fi watchdog: a root timer that reconnects when the gateway stops answering.
# Why and how: tools/net/wifi_watchdog.py. The script is copied root-owned out of the repo on purpose.
# Usage: sudo tools/net/install_wifi_watchdog.sh        Undo: sudo tools/net/install_wifi_watchdog.sh --remove
set -eu
here=$(dirname "$(readlink -f "$0")")
if [ "${1:-}" = "--remove" ]; then
    systemctl disable --now homeai-wifi-watchdog.timer || true
    rm -f /etc/systemd/system/homeai-wifi-watchdog.service /etc/systemd/system/homeai-wifi-watchdog.timer
    rm -rf /usr/local/lib/homeai/wifi_watchdog.py
    systemctl daemon-reload
    echo "removed (outage history kept in /var/lib/homeai-wifi-watchdog)"
    exit 0
fi
install -d -m 755 /usr/local/lib/homeai /var/lib/homeai-wifi-watchdog
install -m 644 -o root -g root "$here/wifi_watchdog.py" /usr/local/lib/homeai/wifi_watchdog.py
install -m 644 "$here/homeai-wifi-watchdog.service" "$here/homeai-wifi-watchdog.timer" /etc/systemd/system/
systemctl daemon-reload
systemctl enable --now homeai-wifi-watchdog.timer
/usr/bin/python3 -I /usr/local/lib/homeai/wifi_watchdog.py --dry-run
echo "installed; watch it with: journalctl -u homeai-wifi-watchdog -f"
