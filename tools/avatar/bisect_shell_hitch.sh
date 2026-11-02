#!/usr/bin/env bash
# Find which GNOME Shell extension causes periodic compositor stalls.
#
# Why: the avatar window probe saw ~110 ms flip stalls every 10 s, caused by
# gnome-shell main-thread CPU bursts. This disables each enabled extension in
# turn (then all of them), runs the window probe, and counts unexplained
# hitches. Every extension is re-enabled on exit, even on Ctrl-C.
#
# Usage: tools/avatar/bisect_shell_hitch.sh [seconds-per-run, default 22]
set -u
cd "$(dirname "$0")/../.."
secs=${1:-22}
mapfile -t exts < <(gnome-extensions list --enabled)
restore() { for e in "${exts[@]}"; do gnome-extensions enable "$e"; done; }
trap restore EXIT

probe() {
    .venv/bin/python tools/avatar/window_probe.py --driver x11 --prime --seconds "$secs" 2>&1 \
        | grep -c "unexplained"
}

echo "baseline (all enabled): $(probe) unexplained hitches"
for e in "${exts[@]}"; do
    gnome-extensions disable "$e"
    sleep 1
    echo "without $e: $(probe) unexplained hitches"
    gnome-extensions enable "$e"
    sleep 1
done
for e in "${exts[@]}"; do gnome-extensions disable "$e"; done
sleep 1
echo "all disabled: $(probe) unexplained hitches"
