#!/usr/bin/env python3
"""Check the avatar window against its resource budget (AVATAR_PLAN §4.3 / §10 Phase 2).

Why: the window shares the CPU with the voice pipeline (and the GPU memory
with the desktop), so it must stay cheap: CPU < 5% idle and < 15% speaking (one core =
100%), GPU memory < 300 MB. It launches a real window on a private event bus
(the live daemon and avatar are untouched) on the iGPU like homeai-avatar.service
(DRI_PRIME=1; ``--dgpu`` for the RX 7900 XT), drives idle, then synthetic
speech, then followup, samples /proc/<pid>/stat and the DRM fdinfo, writes
~/homeai-bench/avatar_profile-*.json and exits 1 if a budget is broken.
"0 fps when minimised" is not checked here (no way to minimise from a script
on this box); check it by eye.

Usage:
    .venv/bin/python tools/avatar/avatar_profile.py [--seconds 15] [--look face] [--points 12000]
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import tempfile
import time
from datetime import datetime
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

BENCH_DIR = Path.home() / "homeai-bench"
BUDGET = {"idle_cpu_pct": 5.0, "speaking_cpu_pct": 15.0, "followup_cpu_pct": 15.0, "gpu_mb": 300.0}
GPU_KEYS = ("drm-resident-vram", "drm-resident-gtt")


def cpu_ticks(stat_text: str) -> int:
    """Pure: utime + stime from /proc/<pid>/stat (the comm field may contain spaces)."""
    fields = stat_text.rsplit(")", 1)[1].split()
    return int(fields[11]) + int(fields[12])


def cpu_pct(ticks_a: int, ticks_b: int, seconds: float, hz: int) -> float:
    if seconds <= 0 or hz <= 0:
        raise ValueError("seconds and hz must be positive")
    return 100.0 * (ticks_b - ticks_a) / hz / seconds


def gpu_kib(fdinfos: list[str]) -> int:
    """Pure: resident VRAM + GTT over unique DRM clients (many fds share one client)."""
    clients: dict[tuple[str, str], int] = {}
    for text in fdinfos:
        kv = {}
        for line in text.splitlines():
            key, _, value = line.partition(":")
            kv[key.strip()] = value.strip()
        if "drm-client-id" not in kv:
            continue
        total = 0
        for key in GPU_KEYS:
            parts = kv.get(key, "0").split()
            total += int(parts[0]) if parts and parts[0].isdigit() else 0
        clients[(kv.get("drm-pdev", ""), kv["drm-client-id"])] = total
    return sum(clients.values())


def verdict(measured: dict) -> list[str]:
    """Pure: the budgets ``measured`` breaks (missing measurements are not failures)."""
    return [f"{k} {measured[k]:.1f} > {limit}" for k, limit in BUDGET.items()
            if k in measured and measured[k] > limit]


def read_fdinfos(pid: int) -> list[str]:
    out = []
    for f in Path(f"/proc/{pid}/fdinfo").glob("*"):
        try:
            out.append(f.read_text())
        except OSError:
            continue
    return out


def main() -> int:
    from homeai.avatar.bus import EventBus
    from homeai.avatar.tour import HEARD, SAID, speech

    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--seconds", type=float, default=15.0, help="per phase")
    parser.add_argument("--points", type=int, default=12000, help="sphere points")
    parser.add_argument("--look", choices=("auto", "face", "sphere"), default="auto",
                        help="like homeai-avatar: auto shows the chosen face if there is one")
    parser.add_argument("--dgpu", action="store_true",
                        help="render on the RX 7900 XT (default: the iGPU, like the service)")
    args = parser.parse_args()

    env = {**os.environ, "SDL_VIDEODRIVER": "x11", "DRI_PRIME": "0" if args.dgpu else "1"}

    hz = os.sysconf("SC_CLK_TCK")
    tmp = Path(tempfile.mkdtemp(prefix="avatar-profile-"))
    sock = tmp / "bus.sock"
    bus = EventBus(sock, source="profile")
    bus.start()
    proc = subprocess.Popen([sys.executable, "-m", "homeai.avatar.app", "--socket", str(sock),
                             "--look", args.look,
                             "--points", str(args.points), "--geometry", str(tmp / "geo.json")],
                            cwd=REPO, env=env, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    measured: dict = {}
    try:
        end = time.monotonic() + 20
        while bus.client_count == 0 and time.monotonic() < end and proc.poll() is None:
            time.sleep(0.1)
        if bus.client_count == 0:
            print("window never connected to the bus")
            return 2
        stat = Path(f"/proc/{proc.pid}/stat")
        for phase in ("idle", "speaking", "followup"):
            bus.publish("state", {"state": phase})
            if phase == "speaking":
                bus.publish("caption", {"who": "user", "text": HEARD})
                bus.publish("caption", {"who": "jarvis", "text": SAID})
                for ev in speech(time.monotonic() + 0.2, seconds=args.seconds + 1):
                    bus.publish(ev["type"], ev["data"])
            time.sleep(2.0)                  # let easing and caches settle
            a, t = cpu_ticks(stat.read_text()), time.monotonic()
            time.sleep(args.seconds)
            measured[f"{phase}_cpu_pct"] = round(
                cpu_pct(a, cpu_ticks(stat.read_text()), time.monotonic() - t, hz), 2)
        measured["gpu_mb"] = round(gpu_kib(read_fdinfos(proc.pid)) / 1024, 1)
    finally:
        proc.terminate()
        try:
            proc.wait(timeout=5)
        except subprocess.TimeoutExpired:
            proc.kill()
        bus.stop()
    broken = verdict(measured)
    report = {"when": datetime.now().isoformat(timespec="seconds"), "look": args.look, "points": args.points,
              "gpu": "dgpu" if args.dgpu else "igpu",
              "seconds_per_phase": args.seconds, **measured, "budget": BUDGET, "broken": broken}
    BENCH_DIR.mkdir(exist_ok=True)
    out = BENCH_DIR / f"avatar_profile-{datetime.now():%Y%m%d-%H%M%S}.json"
    out.write_text(json.dumps(report, indent=1))
    print(json.dumps(report))
    print(f"-> {out}")
    return 1 if broken else 0


if __name__ == "__main__":
    raise SystemExit(main())
