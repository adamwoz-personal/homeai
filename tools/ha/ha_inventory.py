#!/usr/bin/env python3
"""Inventory what Jarvis's Home Assistant account can see.

Uses the non-admin token in ~/.config/homeai/ha.env (HA_TOKEN, HA_URL), so it
shows exactly what the voice tools will be able to reach.

    tools/ha/ha_inventory.py                 # integrations + lights by area
    tools/ha/ha_inventory.py --domain media_player --domain notify
    tools/ha/ha_inventory.py --json          # machine-readable
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import urllib.error
import urllib.request
from collections import defaultdict
from pathlib import Path

ENV = Path(os.environ.get("HOMEAI_HA_ENV", Path.home() / ".config/homeai/ha.env"))


def load_env(path: Path) -> dict[str, str]:
    if not path.exists():
        sys.exit(f"{path} not found; see plans/HOME_ASSISTANT_PLAN.md phase 1")
    out = {}
    for line in path.read_text().splitlines():
        line = line.strip()
        if line and not line.startswith("#") and "=" in line:
            k, v = line.split("=", 1)
            out[k.strip()] = v.strip().strip('"').strip("'")
    if not out.get("HA_TOKEN"):
        sys.exit(f"{path} has no HA_TOKEN= line")
    out.setdefault("HA_URL", "http://127.0.0.1:8123")
    return out


class Forbidden(Exception):
    pass


def call(env: dict, path: str, body: dict | None = None, *, admin_ok: bool = False):
    req = urllib.request.Request(
        env["HA_URL"].rstrip("/") + path,
        data=json.dumps(body).encode() if body is not None else None,
        headers={"Authorization": f"Bearer {env['HA_TOKEN']}", "Content-Type": "application/json"},
        method="POST" if body is not None else "GET",
    )
    try:
        with urllib.request.urlopen(req, timeout=10) as r:
            raw = r.read().decode()
    except urllib.error.HTTPError as e:
        if admin_ok and e.code == 401:
            raise Forbidden(path) from e
        sys.exit(f"{path}: HTTP {e.code} ({'bad token' if e.code == 401 else e.reason})")
    except urllib.error.URLError as e:
        sys.exit(f"{path}: cannot reach HA at {env['HA_URL']} ({e.reason})")
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        return raw


def hue_rooms(states: list[dict]) -> dict[str, str]:
    """Map member light -> Hue room/zone name, from Hue group entities."""
    rooms = {}
    for s in states:
        a = s["attributes"]
        if a.get("is_hue_group"):
            for member in a.get("entity_id", []):
                rooms.setdefault(member, a.get("friendly_name", s["entity_id"]))
    return rooms


def areas_for(env: dict, entity_ids: list[str], states: list[dict]) -> dict[str, str]:
    """HA areas via the template API (admin-only); fall back to Hue rooms."""
    if not entity_ids:
        return {}
    try:
        return _template_areas(env, entity_ids)
    except Forbidden:
        return hue_rooms(states)


def _template_areas(env: dict, entity_ids: list[str]) -> dict[str, str]:
    # One template call instead of one request per entity.
    tpl = "{% set ns = namespace(d={}) %}{% for e in ids %}{% set ns.d = dict(ns.d, **{e: area_name(e) or ''}) %}{% endfor %}{{ ns.d | tojson }}"
    res = call(env, "/api/template", {"template": tpl, "variables": {"ids": entity_ids}}, admin_ok=True)
    return json.loads(res) if isinstance(res, str) else res


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--domain", action="append", help="entity domains to list (default: light)")
    ap.add_argument("--json", action="store_true")
    args = ap.parse_args()
    domains = args.domain or ["light"]

    env = load_env(ENV)
    cfg = call(env, "/api/config")
    entries = call(env, "/api/config/config_entries/entry")
    states = call(env, "/api/states")
    wanted = [s for s in states if s["entity_id"].split(".")[0] in domains]
    area = areas_for(env, [s["entity_id"] for s in wanted], states)

    if args.json:
        print(json.dumps({"version": cfg.get("version"), "integrations": sorted({e["domain"] for e in entries}),
                          "entities": [{**s, "area": area.get(s["entity_id"], "")} for s in wanted]}, indent=2))
        return 0

    print(f"HA {cfg.get('version')} at {env['HA_URL']}")
    print("integrations:", ", ".join(sorted({f"{e['domain']}({e['state']})" for e in entries})))
    by_area: dict[str, list] = defaultdict(list)
    for s in wanted:
        by_area[area.get(s["entity_id"]) or "(no area)"].append(s)
    print(f"\n{len(wanted)} entities in {', '.join(domains)}:")
    for a in sorted(by_area):
        print(f"  {a}")
        for s in sorted(by_area[a], key=lambda s: s["entity_id"]):
            name = s["attributes"].get("friendly_name", "")
            print(f"    {s['entity_id']:<45} {s['state']:<12} {name}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
