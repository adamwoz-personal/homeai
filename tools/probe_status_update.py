#!/usr/bin/env python3
"""What would "Hey Jarvis, status update" say right now? (asked for by Adam 2026-11-07)

Why: the unit tests fake Home Assistant and the weather. This builds the real
status update (voice registry, Ring history from Home Assistant, Open-Meteo
weather) and prints it, read-only: a scratch ledger, so Adam's real "last
update" time doesn't move, and nothing is spoken or announced.

Writes ~/homeai-bench/status-update-<time>/result.json.

Usage:
    .venv/bin/python tools/probe_status_update.py [--as Adam]
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import datetime
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from homeai import status_update as S  # noqa: E402
from homeai.config import load  # noqa: E402
from homeai.doorbell import door_alerts  # noqa: E402
from homeai.home import build_home  # noqa: E402
from homeai.mcp_server import DEFAULT_LOCATION  # noqa: E402
from homeai.speaker import SpeakerRegistry  # noqa: E402
from homeai.weather import get_weather  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--as", dest="who", default="Adam")
    ap.add_argument("--days", type=float, default=1.0, help="how far back to count doorbell history (check only)")
    args = ap.parse_args()
    cfg = load()
    out = Path.home() / "homeai-bench" / f"status-update-{datetime.now():%Y%m%d-%H%M%S}"
    out.mkdir(parents=True)
    registry = SpeakerRegistry(cfg.speaker.registry_path, "wespeaker_en_voxceleb_resnet34_LM")
    home = build_home(cfg.home.env_path, default_echo=cfg.home.default_echo)
    doors = (lambda since, until: door_alerts(home, since, until)) if home else None
    alerts = door_alerts(home, time.time() - args.days * 86400, time.time()) if home else {}
    st = S.StatusUpdate(registry.profiles, doors, lambda: get_weather(DEFAULT_LOCATION).spoken(include_place=False),
                        admins=cfg.status.admins, ledger=S.Ledger(out / "ledger.json"))
    start = time.monotonic()
    reply = st.handle("status update", args.who)
    seconds = round(time.monotonic() - start, 2)
    result = {"as": args.who, "reply": reply, "seconds": seconds, "words": len(reply.split()),
              "profiles": {p.name: {"created_at": p.created_at, "source": p.source} for p in registry.profiles()},
              f"alerts_{args.days:g}d": {f"{d}/{k}": len(v) for (d, k), v in alerts.items()}}
    if home:                                      # what the doorbell lookup saw, for when it says nothing
        from homeai.doorbell import discover
        doors = discover(home.client.states())
        ids = [e for d in doors for e in (d.ding, d.motion, d.activity) if e]
        start = datetime.fromtimestamp(time.time() - args.days * 86400).astimezone().isoformat()
        raw = home.client.history(ids, start, datetime.now().astimezone().isoformat())
        result["doors"] = [d.name for d in doors]
        result["history_rows"] = {eid: len(v) for eid, v in raw.items()}
        result["history_sample"] = {eid: [r["state"] for r in v[-3:]] for eid, v in raw.items()}
        result["current"] = {eid: home.client.state(eid).get("state") for eid in ids}
    (out / "result.json").write_text(json.dumps(result, indent=1))
    print(json.dumps(result, indent=1))
    print(out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
