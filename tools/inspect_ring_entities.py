#!/usr/bin/env python3
"""Every Ring entity in Home Assistant with its state and last change (read-only).

Why: on 2026-11-07 the doorbell event entities still showed 2026-10-15 as the
last ring and motion, so the status update had nothing to report. This shows
whether the Ring integration is still updating.

Usage:
    .venv/bin/python tools/inspect_ring_entities.py
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from homeai.config import load  # noqa: E402
from homeai.home import build_home  # noqa: E402


def main() -> int:
    home = build_home(load().home.env_path)
    if home is None:
        print("Home Assistant isn't configured")
        return 1
    for s in sorted(home.client.states(), key=lambda s: s["entity_id"]):
        attrs = s.get("attributes") or {}
        if "ring" in s["entity_id"] or "front_door" in s["entity_id"] or attrs.get("attribution", "").startswith("Data provided by Ring"):
            print(f"{s['entity_id']:45} {str(s.get('state'))[:32]:34} changed {s.get('last_changed', '')[:19]}"
                  f"  updated {s.get('last_updated', '')[:19]}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
