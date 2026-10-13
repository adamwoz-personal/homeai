#!/usr/bin/env python3
"""Live, read-only check of the Ring doorbell: discovery, last activity, battery.
`--announce` also speaks the ring announcement on the Echos, as a real ring would."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from homeai.doorbell import discover, door_battery, door_status  # noqa: E402
from homeai.ha_client import HAClient, load_settings  # noqa: E402
from homeai.home import build_home  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--announce", action="store_true")
    args = ap.parse_args()
    home = build_home()
    if home is None:
        print("FAIL: Home Assistant isn't set up")
        return 1
    doors = discover(HAClient(load_settings()).states())
    if not doors:
        print("FAIL: no Ring doorbell entities found (add the Ring integration in HA)")
        return 1
    for door in doors:
        print("door:", door)
    print(door_status(home))
    print(door_battery(home))
    if args.announce:
        print(home.announce("Someone is at the front door.", "everywhere"))
    print("OK")
    return 0


if __name__ == "__main__":
    sys.exit(main())
