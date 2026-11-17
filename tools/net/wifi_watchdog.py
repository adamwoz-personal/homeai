"""Bring the Wi-Fi back when it drops and NetworkManager doesn't retry.

Why: on 2026-11-17 at 04:31:57 the box lost its access point (MT7925, driver mt7925e, SSID "bednarz").
NetworkManager failed the connection with ssid-not-found and then logged nothing for 5 h 40 min. Jarvis's
routines, Home Assistant's Alexa buttons and DNS were all dead until Adam rebooted (docs/RUNBOOK.md
"Routines fail and DNS server refused query").

Run by a root systemd timer every minute (tools/net/homeai-wifi-watchdog.{service,timer}). Each run pings the
default gateway once. It does nothing while the network is up, and nothing before the network has been down
for DOWN_S. Then it escalates, logging every step to the journal:

  1. reconnect:     nmcli device wifi rescan; nmcli connection up <connection>
  2. reload driver: modprobe -r <driver>; modprobe <driver>; then reconnect   (if 1 didn't help in DOWN_S)
  then reconnect again every RETRY_S and reload the driver at most once per RELOAD_S, until it's back.

State is kept in /run (cleared by a reboot). Only `decide` holds the policy, so tests drive it with a fake clock.
Each outage (start, end, actions) is appended to EVENTS, which Jarvis's status update reads
(homeai/netwatch.py), so Jarvis can say what went wrong with her own network and what fixed it.

Stdlib only: the installer copies it to /usr/local/lib/homeai (root-owned) and the unit runs it with the system
python3, so a root timer never runs a file from the repo, which the builder agent can write to.

Usage (root): python3 tools/net/wifi_watchdog.py [--connection bednarz] [--driver mt7925e] [--dry-run]
Install:      sudo tools/net/install_wifi_watchdog.sh
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

STATE = Path("/run/homeai-wifi-watchdog.json")
EVENTS = Path("/var/lib/homeai-wifi-watchdog/events.jsonl")   # world-readable; one line per outage
DOWN_S = 180          # how long the gateway must be silent before acting (a router reboot takes ~2 min)
RETRY_S = 600         # then reconnect again this often
RELOAD_S = 3600       # and reload the driver at most this often


def decide(state: dict, up: bool, now: float) -> tuple[str | None, dict]:
    """(action or None, new state). Actions: "reconnect", "reload", "recovered"."""
    if up:
        return ("recovered" if state.get("acted") else None), {}
    state = dict(state)
    since = state.setdefault("down_since", now)
    if now - since < DOWN_S:
        return None, state
    last = state.get("last_action_at")
    if last is not None and now - last < (DOWN_S if state.get("acted") == 1 else RETRY_S):
        return None, state
    reload_ok = state.get("acted", 0) >= 1 and now - state.get("last_reload_at", -RELOAD_S) >= RELOAD_S
    action = "reload" if reload_ok else "reconnect"
    state["acted"] = state.get("acted", 0) + 1
    state["last_action_at"] = now
    if action == "reload":
        state["last_reload_at"] = now
    return action, state


def gateway() -> str | None:
    out = subprocess.run(["ip", "-4", "route", "show", "default"], capture_output=True, text=True).stdout.split()
    return out[out.index("via") + 1] if "via" in out else None


def is_up(gw: str | None) -> bool:
    if not gw:
        return False
    return subprocess.run(["ping", "-c", "1", "-W", "2", gw], capture_output=True).returncode == 0


def commands(action: str, connection: str, driver: str) -> list[list[str]]:
    reconnect = [["nmcli", "device", "wifi", "rescan"], ["nmcli", "connection", "up", connection]]
    if action == "reload":
        return [["modprobe", "-r", driver], ["modprobe", driver], ["sleep", "10"], *reconnect]
    return reconnect if action == "reconnect" else []


def outage(state: dict, now: float) -> dict:
    """The EVENTS line for an outage that just ended (state as it was while down)."""
    return {"down_since": state.get("down_since"), "back_at": now, "actions": state.get("log", [])}


def record(path: Path, event: dict) -> None:
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("a") as f:
            f.write(json.dumps(event) + "\n")
    except OSError as exc:
        print(f"wifi-watchdog: couldn't record the outage: {exc}", flush=True)


def load_state(path: Path) -> dict:
    try:
        return json.loads(path.read_text())
    except (OSError, ValueError):
        return {}


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n", 1)[0])
    ap.add_argument("--connection", default="bednarz")
    ap.add_argument("--driver", default="mt7925e")
    ap.add_argument("--state", type=Path, default=STATE)
    ap.add_argument("--events", type=Path, default=EVENTS)
    ap.add_argument("--dry-run", action="store_true", help="say what it would do, change nothing")
    args = ap.parse_args(argv)
    state = load_state(args.state)
    gw = gateway() or state.get("gateway")
    now = time.time()
    action, new = decide(state, is_up(gw), now)
    if gw:
        new["gateway"] = gw
    if action == "recovered":
        print(f"wifi-watchdog: gateway {gw} answers again after {state.get('acted')} action(s)", flush=True)
        if not args.dry_run:
            record(args.events, outage(state, now))
    elif action:
        print(f"wifi-watchdog: gateway {gw} silent for {int(now - new['down_since'])}s: {action}", flush=True)
        ok = True
        for cmd in commands(action, args.connection, args.driver):
            if args.dry_run:
                print("  would run: " + " ".join(cmd), flush=True)
                continue
            try:
                proc = subprocess.run(cmd, capture_output=True, text=True, timeout=60)
                rc, said = proc.returncode, (proc.stderr or proc.stdout).strip()[:200]
            except (OSError, subprocess.TimeoutExpired) as exc:
                rc, said = -1, str(exc)[:200]
            ok = ok and rc == 0
            print(f"  {' '.join(cmd)} -> {rc} {said}", flush=True)
        new["log"] = [*new.get("log", []), {"at": now, "action": action, "ok": ok}]
    if not args.dry_run:
        args.state.write_text(json.dumps(new))
    return 0


if __name__ == "__main__":
    sys.exit(main())
