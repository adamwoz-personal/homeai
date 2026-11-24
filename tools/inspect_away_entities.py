#!/usr/bin/env python3
"""Home Assistant entities an away mode could use: presence, cameras, motion, doors, alarms (read-only).

Why: the away-mode plan (plans/SECURITY_MODE_PLAN.md) needs to know how Jarvis can tell nobody is home (person /
device_tracker), what can see motion (camera, motion sensors, Ring) and what already raises alarms (smoke, water,
door contacts) before it relies on any of them. Prints entity, state and last change; no attributes that could
hold locations or tokens.

Usage:
    .venv/bin/python tools/inspect_away_entities.py
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from homeai.config import load  # noqa: E402
from homeai.home import build_home  # noqa: E402

DOMAINS = ("person.", "device_tracker.", "camera.", "alarm_control_panel.", "lock.")
CLASSES = {"motion", "occupancy", "presence", "door", "window", "opening", "smoke", "carbon_monoxide",
           "moisture", "sound", "tamper", "vibration", "safety", "gas"}


def wanted(state: dict) -> str | None:
    """The group an entity belongs to, or None."""
    eid = state.get("entity_id", "")
    for d in DOMAINS:
        if eid.startswith(d):
            return d.rstrip(".")
    cls = (state.get("attributes") or {}).get("device_class")
    if eid.startswith(("binary_sensor.", "event.")) and cls in CLASSES:
        return f"sensor:{cls}"
    if eid.startswith("event.") and ("motion" in eid or "ding" in eid):
        return "ring-event"
    return None


def main() -> int:
    home = build_home(load().home.env_path)
    if home is None:
        print("Home Assistant isn't configured")
        return 1
    rows = [(wanted(s), s) for s in home.client.states()]
    rows = sorted((r for r in rows if r[0]), key=lambda r: (r[0], r[1]["entity_id"]))
    for group, s in rows:
        print(f"{group:22} {s['entity_id']:48} {str(s.get('state'))[:24]:26} changed {s.get('last_changed', '')[:19]}")
    print(f"{len(rows)} entities")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
