"""What the lights did around a moment: did an Alexa routine (e.g. "all clear") actually change them?

Exists because on 2026-11-13 Adam said "all clear" left the lights dim. Jarvis's log showed it sent the
routine to Alexa; this shows whether any light changed afterwards (state, brightness, colour).

Usage: .venv/bin/python tools/ha/inspect_light_history.py "2026-11-11 06:24" [minutes_before] [minutes_after]
Reads Home Assistant history only (no service calls).
"""
from __future__ import annotations

import datetime as dt
import sys

from homeai.ha_client import HAClient, load_settings


def main(argv: list[str]) -> int:
    if not argv:
        print(__doc__)
        return 2
    at = dt.datetime.fromisoformat(argv[0]).astimezone()
    before = float(argv[1]) if len(argv) > 1 else 10
    after = float(argv[2]) if len(argv) > 2 else 10
    client = HAClient(load_settings())
    lights = [s["entity_id"] for s in client.states() if s["entity_id"].startswith("light.")]
    start = (at - dt.timedelta(minutes=before)).isoformat()
    end = (at + dt.timedelta(minutes=after)).isoformat()
    from urllib.parse import quote
    raw = client._request("GET", f"/api/history/period/{quote(start)}?end_time={quote(end)}"
                                 f"&filter_entity_id={','.join(lights)}")
    for rows in raw:
        print(rows[0]["entity_id"])
        for row in rows:
            a = row.get("attributes", {})
            when = dt.datetime.fromisoformat(row["last_changed"]).astimezone().strftime("%m-%d %H:%M:%S")
            print(f"  {when} {row['state']:<4} bri={a.get('brightness')} rgb={a.get('rgb_color')} "
                  f"ct={a.get('color_temp_kelvin')}")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
