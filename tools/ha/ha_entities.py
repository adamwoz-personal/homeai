#!/usr/bin/env python3
"""List HA entities by domain / name filter. Usage: ha_entities.py [text] [--domain camera]"""
import argparse, json, sys
import requests
from homeai.ha_client import load_settings

p = argparse.ArgumentParser()
p.add_argument("text", nargs="?", default="")
p.add_argument("--domain")
a = p.parse_args()
s = load_settings()
r = requests.get(s.url + "/api/states", headers={"Authorization": f"Bearer {s.token}"}, timeout=20)
r.raise_for_status()
for e in sorted(r.json(), key=lambda e: e["entity_id"]):
    eid = e["entity_id"]
    name = e["attributes"].get("friendly_name", "")
    if a.domain and not eid.startswith(a.domain + "."):
        continue
    if a.text.lower() in (eid + name).lower():
        print(f"{eid:55} {e['state'][:20]:20} {name}")
