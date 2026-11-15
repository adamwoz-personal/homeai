"""Reload the Home Assistant integration that owns an entity (restarts its connections, e.g. Ring's push listener).

Why: on 2026-10-16 10:28 the Ring integration's push listener (firebase_messaging FcmPushClient) hit a malformed
message ("Incorrect padding"), shut itself down and never restarted, so event.front_door_ding / _motion froze at
2026-10-15 while the polled sensor.front_door_last_activity kept updating. Reloading the config entry starts a new
listener without restarting Home Assistant. Needs an admin token (homeassistant.reload_config_entry is admin-only).

Usage: .venv/bin/python tools/ha/reload_integration.py [entity_id]   (default event.front_door_ding)
"""
from __future__ import annotations

import sys

from homeai.ha_client import HAClient, load_settings


def main(argv: list[str]) -> int:
    entity = argv[0] if argv else "event.front_door_ding"
    client = HAClient(load_settings())
    try:
        client.call_service("homeassistant", "reload_config_entry", {"entity_id": entity})
    except Exception as exc:  # noqa: BLE001 - report, never print the token
        print(f"reload refused: {type(exc).__name__}: {str(exc)[:200]}")
        return 1
    print(f"reloaded the integration owning {entity}")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
