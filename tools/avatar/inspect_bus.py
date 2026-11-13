#!/usr/bin/env python3
# Copyright (c) 2026 Adam Wosotowsky
# SPDX-License-Identifier: BUSL-1.1
"""Watch an avatar event bus and report when each event arrived.

Why: Adam saw the sphere stay still for the first half of the first sentence.
For every ``utterance`` this prints how early (positive) or late (negative)
it arrived relative to its scheduled ``start_mono``, plus its duration and
viseme count, so a late or dropped first chunk is visible in numbers.

Usage:
    .venv/bin/python tools/avatar/inspect_bus.py --socket $XDG_RUNTIME_DIR/homeai/demo.sock --seconds 30
    (then run tools/avatar/avatar_demo.py --wait-client 3 in another shell)
Writes ~/homeai-bench/inspect_bus-<stamp>.json.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import datetime
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

BENCH_DIR = Path.home() / "homeai-bench"


def describe(event: dict, arrived: float) -> dict:
    """Pure: one row per event; ``early_ms`` only for utterances."""
    data = event.get("data") or {}
    row = {"type": event.get("type"), "arrived": round(arrived, 4),
           "sent_to_arrival_ms": round((arrived - float(event.get("t_mono") or arrived)) * 1000, 1)}
    if event.get("type") == "utterance":
        row.update(id=data.get("id"), early_ms=round((float(data["start_mono"]) - arrived) * 1000, 1),
                   duration_ms=data.get("duration_ms"), visemes=len(data.get("visemes") or []))
    elif event.get("type") == "state":
        row["state"] = data.get("state")
    elif event.get("type") == "snapshot":     # the latched events a new client starts from
        row["state"] = ",".join(sorted((data.get("events") or {})))
        row["latched"] = {k: (v.get("data") or {}) for k, v in (data.get("events") or {}).items()
                          if k in ("state", "recording", "mode")}
    elif event.get("type") == "recording":
        row["state"] = f"on={data.get('on')} until={data.get('until')}"
    return row


def main() -> int:
    from homeai.avatar.client import BusClient
    from homeai.avatar.bus import default_socket_path

    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--socket", type=Path, default=default_socket_path())
    parser.add_argument("--seconds", type=float, default=30.0)
    args = parser.parse_args()

    rows: list[dict] = []

    def on_event(event: dict) -> None:
        row = describe(event, time.monotonic())
        rows.append(row)
        extra = (f"  {row['id']}  early {row['early_ms']:+.0f} ms  dur {row['duration_ms']} ms"
                 f"  visemes {row['visemes']}" if row["type"] == "utterance"
                 else f"  {row.get('state', '')}")
        print(f"{row['arrived']:.3f} {row['type']:12}{extra}", flush=True)

    client = BusClient(on_event, lambda up: print(f"connected: {up}", flush=True), args.socket)
    client.start()
    try:
        time.sleep(args.seconds)
    except KeyboardInterrupt:
        pass
    client.stop()
    BENCH_DIR.mkdir(exist_ok=True)
    out = BENCH_DIR / f"inspect_bus-{datetime.now():%Y%m%d-%H%M%S}.json"
    out.write_text(json.dumps(rows, indent=2))
    print(f"-> {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
