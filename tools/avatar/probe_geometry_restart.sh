#!/usr/bin/env bash
# Copyright (c) 2026 Adam Wosotowsky
# SPDX-License-Identifier: BUSL-1.1
# Why: systemd stops homeai-avatar with SIGTERM; the window must exit 0 and save
# its geometry, and a restart must come back at the same place (x11 reports a
# decoration offset, which could make the window creep on every restart).
# Usage: tools/avatar/probe_geometry_restart.sh [runs=3]   (needs a desktop session)
# Prints each run's exit code and saved geometry; same position every run = no creep.
set -u
cd "$(dirname "$0")/../.."
RUNS="${1:-3}"
GEO="$(mktemp -d)/avatar.json"
for run in $(seq 1 "$RUNS"); do
    DRI_PRIME=1 .venv/bin/python -m homeai.avatar.app --socket /tmp/homeai-no-bus.sock \
        --geometry "$GEO" >"${GEO%.json}-$run.log" 2>&1 &
    pid=$!
    sleep 4
    kill -TERM "$pid"
    wait "$pid"
    echo "run $run: exit $? geometry $(cat "$GEO" 2>/dev/null || echo MISSING)"
done
rm -rf "$(dirname "$GEO")"
