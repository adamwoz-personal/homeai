#!/usr/bin/env bash
# Installs the nightly art practice timer as a user unit (ART_FEATURES_PLAN F9; Adam chose 01:00-06:30 on
# 2026-11-19). User scope: practice switches the user units llama-server and homeai-comfyui through
# homeai.gpu_mode, and needs no root. Linger is on, so it runs with nobody logged in.
# Usage: tools/art/install_practice_timer.sh [--uninstall]
set -euo pipefail
REPO="$(cd "$(dirname "$0")/../.." && pwd)"
DEST="$HOME/.config/systemd/user"
if [[ "${1:-}" == "--uninstall" ]]; then
    systemctl --user disable --now homeai-art-practice.timer || true
    rm -f "$DEST/homeai-art-practice.service" "$DEST/homeai-art-practice.timer"
    systemctl --user daemon-reload
    exit 0
fi
mkdir -p "$DEST"
sed -e "s#@REPO@#${REPO}#g" "$REPO/deploy/systemd/homeai-art-practice.service.in" > "$DEST/homeai-art-practice.service"
cp "$REPO/deploy/systemd/homeai-art-practice.timer" "$DEST/"
systemctl --user daemon-reload
systemctl --user enable --now homeai-art-practice.timer
systemctl --user list-timers homeai-art-practice.timer --no-pager
