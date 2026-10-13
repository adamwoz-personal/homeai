#!/usr/bin/env python3
"""Does this HA offer an integration in the Add-integration list? Usage: ha_integration_lookup.py ring"""
import asyncio, json, sys
from homeai.ha_client import load_settings
import websockets

async def main(term):
    s = load_settings()
    async with websockets.connect(s.url.replace("http", "ws", 1) + "/api/websocket") as ws:
        await ws.recv()
        await ws.send(json.dumps({"type": "auth", "access_token": s.token}))
        await ws.recv()
        await ws.send(json.dumps({"id": 1, "type": "integration/descriptions"}))
        r = json.loads(await ws.recv())
    if not r.get("success"):
        print("failed:", r.get("error")); return 1
    res = r["result"]
    for kind in res:
        for k, v in res[kind].items():
            if term in k or term in json.dumps(v).lower():
                print(kind, k, json.dumps(v)[:200])
asyncio.run(main(sys.argv[1].lower()))
