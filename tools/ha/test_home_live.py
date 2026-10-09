#!/usr/bin/env python3
"""Functional test of home control against the real Home Assistant.

    .venv/bin/python tools/ha/test_home_live.py            # read-only checks
    .venv/bin/python tools/ha/test_home_live.py --act      # + light, timer (restored)
    .venv/bin/python tools/ha/test_home_live.py --act --announce   # + an announcement

Read-only: credentials, states, device registry over the websocket, Echo
resolution, light-name matching against the real Hue rooms, timer status,
and the MCP server end to end over stdio (tools/list with the home tools).

--act changes things and puts them back: toggles a Hue room and restores its
state, sets and cancels a 1-minute Alexa timer. --announce speaks on every
Echo. Exit status is the number of failed checks.
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from homeai.ha_client import HAClient, load_settings  # noqa: E402
from homeai.home import Home, HomeError  # noqa: E402

failures = 0


def check(name: str, ok: bool, detail: str = "") -> None:
    global failures
    failures += not ok
    print(f"{'PASS' if ok else 'FAIL'}  {name}{'  - ' + detail if detail else ''}")


def mcp_tools() -> list[str]:
    msgs = [{"jsonrpc": "2.0", "id": 1, "method": "initialize", "params": {}},
            {"jsonrpc": "2.0", "id": 2, "method": "tools/list"},
            {"jsonrpc": "2.0", "id": 3, "method": "tools/call",
             "params": {"name": "timer", "arguments": {"action": "status"}}}]
    out = subprocess.run([str(ROOT / "bin/homeai-mcp")], input="\n".join(map(json.dumps, msgs)) + "\n",
                         capture_output=True, text=True, timeout=60).stdout
    replies = {r["id"]: r for r in map(json.loads, out.splitlines())}
    return [t["name"] for t in replies[2]["result"]["tools"]], replies[3]["result"]


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--act", action="store_true", help="toggle a light and set/cancel a timer")
    ap.add_argument("--announce", action="store_true", help="make a whole-house announcement")
    ap.add_argument("--room", default="den lamp", help="Hue room to toggle with --act")
    args = ap.parse_args()

    settings = load_settings()
    client = HAClient(settings)
    home = Home(client)

    states = client.states()
    check("REST states readable", len(states) > 10, f"{len(states)} entities")
    devices = client.devices()
    check("device registry over websocket", len(devices) > 0, f"{len(devices)} devices")

    echos = home.echos()
    check("Echos found", len(echos) > 0, ", ".join(e.name for e in echos))
    for where, want in ((None, "kitchen"), ("everywhere", "everywhere"), ("basement", "basement")):
        try:
            got = home.echo(where).name
            check(f"echo({where!r})", want in got.lower(), got)
        except HomeError as exc:
            check(f"echo({where!r})", False, str(exc))

    lights = home._lights(states)
    for target in ("foyer", "den", "kitchen cabinets", "all"):
        matches, _ = home.match_lights(target, lights)
        check(f"lights match {target!r}", bool(matches), ", ".join(m.name for m in matches[:5]))

    check("timer status answers", bool(home.timer_status()), home.timer_status())

    tools, status = mcp_tools()
    check("MCP lists home tools", {"lights", "timer", "announce", "music"} <= set(tools), ", ".join(tools))
    check("MCP timer status call", not status.get("isError"), status["content"][0]["text"])

    if args.act:
        matches, _ = home.match_lights(args.room, lights)
        room = matches[0]
        before = room.state
        flip = "off" if before == "on" else "on"
        print(f"..    {room.name} is {before}; turning {flip}, then back")
        home.lights(args.room, flip)
        time.sleep(2)
        now = client.state(room.entity_id)["state"]
        check(f"light {room.name} -> {flip}", now == flip, now)
        home.lights(args.room, before if before in ("on", "off") else "off")
        time.sleep(2)
        check(f"light {room.name} restored", client.state(room.entity_id)["state"] == before)

        print("..    setting a 1 minute 'live test' timer on the kitchen Echo")
        check("timer set", "set" in home.timer_set(60, "test"))
        check("timer in ledger", home.timer_status("test").startswith("The test timer has"),
              home.timer_status("test"))
        time.sleep(3)
        check("timer cancel", "cancelled" in home.timer_cancel("test"))

    if args.announce:
        check("announce", home.announce("This is a home control test.") == "Announced.")
        check("quiet window set", home.quiet.remaining() > 0, f"{home.quiet.remaining():.0f}s")

    print(f"\n{failures} failure(s)")
    return failures


if __name__ == "__main__":
    sys.exit(main())
