#!/usr/bin/env python3
"""Quiet Home Assistant's known-harmless log noise (idempotent, backs up first).

    tools/ha/ha_quiet_logs.py            # apply (stops HA ~30 s, edits, restarts)
    tools/ha/ha_quiet_logs.py --dry-run  # show what would change
    tools/ha/ha_quiet_logs.py --scan     # count WARNING/ERROR by source, last 6 h

Two sources, both measured 2026-10-09 (HA 2026.10):
1. habluetooth: "Failed to force stop scanner ... 'NoneType' has no attribute
   'send'" tracebacks every 10 min. The container has no working BlueZ bus
   even with /run/dbus mounted, and nothing here uses Bluetooth. The auto-
   created bluetooth config entry is disabled (disabled_by=user), which
   survives restarts and re-discovery.
2. aioamazondevices: "Failed to refresh communications settings for device X,
   used cached values", one per Echo per cycle. Amazon rate limiting; cached
   values are fine. Logger level for that package goes to error.

HA's config files are root-owned (written by the container), so the edit runs
inside a throwaway container of the same image, with HA stopped (HA rewrites
core.config_entries on shutdown). Backups: <file>.homeai-bak next to the file.
"""
from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
from collections import Counter
from pathlib import Path

HERE = Path(__file__).resolve().parent
CONFIG = Path.home() / "homeassistant-config"
IMAGE = "ghcr.io/home-assistant/home-assistant:stable"
LOGGER_BLOCK = "\nlogger:\n  logs:\n    aioamazondevices: error\n"
DISABLE_DOMAINS = ("bluetooth",)


def apply(config: Path, dry_run: bool = False) -> list[str]:
    """Edit config in place. Runs as root inside the container."""
    changes = []
    entries_path = config / ".storage" / "core.config_entries"
    data = json.loads(entries_path.read_text())
    for entry in data["data"]["entries"]:
        if entry["domain"] in DISABLE_DOMAINS and not entry.get("disabled_by"):
            entry["disabled_by"] = "user"
            changes.append(f"disabled config entry {entry['domain']} ({entry['title']})")
    if changes and not dry_run:
        backup(entries_path)
        entries_path.write_text(json.dumps(data, indent=2))

    yaml_path = config / "configuration.yaml"
    text = yaml_path.read_text()
    if not re.search(r"^logger:", text, re.M):
        changes.append("added logger: aioamazondevices: error")
        if not dry_run:
            backup(yaml_path)
            yaml_path.write_text(text.rstrip("\n") + "\n" + LOGGER_BLOCK)
    elif "aioamazondevices" not in text:
        changes.append("MANUAL: configuration.yaml already has a logger: block; add "
                       "`aioamazondevices: error` under logs:")
    return changes


def backup(path: Path) -> None:
    bak = path.with_name(path.name + ".homeai-bak")
    if not bak.exists():  # keep the first, pristine copy
        bak.write_bytes(path.read_bytes())


def docker(*args: str) -> subprocess.CompletedProcess:
    cmd = ["docker", *args]
    probe = subprocess.run(["docker", "info"], capture_output=True)
    if probe.returncode != 0:
        cmd = ["sg", "docker", "-c", " ".join(map(_q, cmd))]
    return subprocess.run(cmd, capture_output=True, text=True)


def _q(s: str) -> str:
    import shlex
    return shlex.quote(s)


def scan(hours: int = 6) -> int:
    out = docker("logs", "--since", f"{hours}h", "homeassistant")
    counts: Counter[tuple[str, str]] = Counter()
    for line in (out.stdout + out.stderr).splitlines():
        m = re.match(r"\S+ \S+ (WARNING|ERROR|CRITICAL) \(\S+\) \[([^\]]+)\] (.{0,70})", line)
        if m:
            msg = re.sub(r"\d+|\([0-9A-F:]{17}\)", "#", m.group(3))
            counts[(m.group(1), f"{m.group(2)}: {msg}")] += 1
    if not counts:
        print(f"no warnings or errors in the last {hours} h")
    for (level, what), n in counts.most_common(15):
        print(f"{n:5d}  {level:8s} {what}")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--scan", action="store_true")
    ap.add_argument("--hours", type=int, default=6)
    ap.add_argument("--inside", metavar="DIR", help=argparse.SUPPRESS)
    args = ap.parse_args()

    if args.inside:  # running as root in the throwaway container
        for change in apply(Path(args.inside), args.dry_run):
            print("  " + change)
        return 0
    if args.scan:
        return scan(args.hours)

    run = lambda *a: docker("run", "--rm", "-v", f"{CONFIG}:/config",
                            "-v", f"{Path(__file__).resolve()}:/ha_quiet_logs.py:ro",
                            "--entrypoint", "python3", IMAGE, "/ha_quiet_logs.py", "--inside", "/config", *a)
    if args.dry_run:
        res = run("--dry-run")
        print(res.stdout or "nothing to change", res.stderr[-300:] if res.returncode else "")
        return res.returncode
    print("stopping Home Assistant...")
    docker("stop", "homeassistant")
    try:
        res = run()
        print(res.stdout.rstrip() or "  nothing to change")
        if res.returncode:
            print(res.stderr[-500:], file=sys.stderr)
            return res.returncode
    finally:
        print("starting Home Assistant...")
        subprocess.run([str(HERE / "ha_container.sh"), "start"])
    return 0


if __name__ == "__main__":
    sys.exit(main())
