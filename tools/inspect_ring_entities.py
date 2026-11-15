#!/usr/bin/env python3
"""Every Ring entity in Home Assistant with its state and last change (read-only).

Why: on 2026-11-07 the doorbell event entities still showed 2026-10-15 as the
last ring and motion, so the status update had nothing to report. This shows
whether the Ring integration is still updating.

Usage:
    .venv/bin/python tools/inspect_ring_entities.py
    .venv/bin/python tools/inspect_ring_entities.py --history 31   (every activity Ring saw, from the polled sensor)
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from homeai.config import load  # noqa: E402
from homeai.home import build_home  # noqa: E402


def activity_times(raw: list) -> list[str]:
    """Distinct last-activity timestamps from a history response (each one is a ring or motion Ring saw)."""
    seen: list[str] = []
    for rows in raw or []:
        for row in rows:
            value = row.get("state", "")
            if value[:2] == "20" and value not in seen:
                seen.append(value)
    return seen


def history(home, days: float) -> int:
    """Every activity sensor.front_door_last_activity recorded in the last ``days`` (catches what the events missed)."""
    import datetime as dt
    from urllib.parse import quote
    now = dt.datetime.now(dt.timezone.utc)
    start, end = (now - dt.timedelta(days=days)).isoformat(), now.isoformat()   # no end_time = one day only
    raw = home.client._request("GET", f"/api/history/period/{quote(start)}"
                                      f"?end_time={quote(end)}&filter_entity_id=sensor.front_door_last_activity"
                                      f"&minimal_response")
    times = activity_times(raw)
    for t in times:
        print(dt.datetime.fromisoformat(t).astimezone().strftime("%Y-%m-%d %H:%M:%S"))
    print(f"{len(times)} activities in the last {days:g} days")
    return 0


def main() -> int:
    home = build_home(load().home.env_path)
    if home is None:
        print("Home Assistant isn't configured")
        return 1
    if len(sys.argv) > 2 and sys.argv[1] == "--history":
        return history(home, float(sys.argv[2]))
    for s in sorted(home.client.states(), key=lambda s: s["entity_id"]):
        attrs = s.get("attributes") or {}
        if "ring" in s["entity_id"] or "front_door" in s["entity_id"] or attrs.get("attribution", "").startswith("Data provided by Ring"):
            print(f"{s['entity_id']:45} {str(s.get('state'))[:32]:34} changed {s.get('last_changed', '')[:19]}"
                  f"  updated {s.get('last_updated', '')[:19]}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
