#!/usr/bin/env python3
"""Probe what Jarvis's HA account can do with the Echos (alexa_devices).

Read-only by default: lists Echo devices with their device_id (needed by
alexa_devices.send_text_command) via the websocket registries, which a
non-admin user may read. Actions make noise, so they are explicit flags:

    .venv/bin/python tools/ha/ha_alexa_probe.py                       # list
    .venv/bin/python tools/ha/ha_alexa_probe.py --speak kitchen "Test"
    .venv/bin/python tools/ha/ha_alexa_probe.py --announce everywhere "Test"
    .venv/bin/python tools/ha/ha_alexa_probe.py --command kitchen "set a timer for 1 minute"
    .venv/bin/python tools/ha/ha_alexa_probe.py --timers

Echo names match on a case-insensitive substring of the device name.
"""
from __future__ import annotations

import argparse
import asyncio
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from ha_inventory import ENV, call, load_env  # noqa: E402

import websockets  # noqa: E402


async def ws_registries(env: dict) -> tuple[list, list]:
    url = env["HA_URL"].replace("http", "ws", 1).rstrip("/") + "/api/websocket"
    async with asyncio.timeout(20), websockets.connect(url, max_size=2**24, open_timeout=10) as ws:
        await ws.recv()
        await ws.send(json.dumps({"type": "auth", "access_token": env["HA_TOKEN"]}))
        auth = json.loads(await ws.recv())
        if auth.get("type") != "auth_ok":
            sys.exit(f"websocket auth failed: {auth}")
        out = []
        for i, kind in enumerate(("config/device_registry/list", "config/entity_registry/list"), 1):
            await ws.send(json.dumps({"id": i, "type": kind}))
            msg = json.loads(await ws.recv())
            if not msg.get("success"):
                sys.exit(f"{kind}: {msg.get('error')}")
            out.append(msg["result"])
        return out[0], out[1]


def echo_devices(env: dict) -> list[dict]:
    devices, entities = asyncio.run(ws_registries(env))
    entries = {e["entry_id"] for e in call(env, "/api/config/config_entries/entry") if e["domain"] == "alexa_devices"}
    by_dev: dict[str, list[str]] = {}
    for e in entities:
        if e.get("device_id"):
            by_dev.setdefault(e["device_id"], []).append(e["entity_id"])
    echos = []
    for d in devices:
        if entries & set(d.get("config_entries", [])):
            echos.append({"device_id": d["id"], "name": d.get("name_by_user") or d.get("name"),
                          "model": d.get("model"), "entities": sorted(by_dev.get(d["id"], []))})
    return sorted(echos, key=lambda d: d["name"].lower())


def pick(echos: list[dict], name: str) -> dict:
    hits = [d for d in echos if name.lower() in d["name"].lower()]
    if len(hits) != 1:
        sys.exit(f"'{name}' matches {[d['name'] for d in hits] or 'nothing'}; be more specific")
    return hits[0]


def entity(dev: dict, suffix: str) -> str:
    for e in dev["entities"]:
        if e.endswith(suffix):
            return e
    sys.exit(f"{dev['name']} has no *{suffix} entity")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--speak", nargs=2, metavar=("ECHO", "TEXT"))
    ap.add_argument("--announce", nargs=2, metavar=("ECHO", "TEXT"))
    ap.add_argument("--command", nargs=2, metavar=("ECHO", "TEXT"), help="say TEXT to Alexa on ECHO")
    ap.add_argument("--timers", action="store_true", help="show next timer per Echo")
    args = ap.parse_args()

    env = load_env(ENV)

    if args.speak or args.announce:
        # notify entities are in /api/states; no registry lookup needed.
        name, text = args.speak or args.announce
        suffix = "_speak" if args.speak else "_announce"
        ids = [s["entity_id"] for s in call(env, "/api/states")
               if s["entity_id"].startswith("notify.") and s["entity_id"].endswith(suffix)
               and name.lower().replace(" ", "_") in s["entity_id"]]
        if len(ids) != 1:
            sys.exit(f"'{name}' matches {ids or 'nothing'}; be more specific")
        target = ids[0]
        call(env, "/api/services/notify/send_message", {"entity_id": target, "message": text})
        print(f"sent to {target}")
        return 0

    echos = echo_devices(env)
    if args.command:
        dev = pick(echos, args.command[0])
        call(env, "/api/services/alexa_devices/send_text_command",
             {"device_id": dev["device_id"], "text_command": args.command[1]})
        print(f"sent to {dev['name']}: {args.command[1]!r}")
    elif args.timers:
        states = {s["entity_id"]: s for s in call(env, "/api/states")}
        for d in echos:
            for e in d["entities"]:
                if e.endswith("_next_timer") and states.get(e, {}).get("state") not in (None, "unavailable", "unknown"):
                    a = states[e]["attributes"]
                    print(f"{d['name']}: {states[e]['state']} {a.get('label') or ''}".rstrip())
    else:
        for d in echos:
            print(f"{d['name']:<24} {d['model'] or '':<28} {d['device_id']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
