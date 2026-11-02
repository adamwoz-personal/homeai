#!/usr/bin/env python3
"""Detect system-wide scheduling stalls with a plain timer loop (no window, no GPU).

Why: the avatar window probe showed ~110 ms frame hitches every 10 s. If this
loop sees the same gaps, the stall is system-wide (CPU/kernel/power); if not,
it is in the display path (compositor / GL flip).

Usage:
    .venv/bin/python tools/avatar/probe_stalls.py [--seconds 30] [--tick-ms 2] [--gap-ms 20]
"""
from __future__ import annotations

import argparse
import os
import time


def find_gaps(stamps: list[float], expected_s: float, gap_s: float) -> list[tuple[float, float]]:
    """Return (t_since_start_s, gap_ms) for intervals longer than expected + gap_s."""
    if len(stamps) < 2:
        return []
    out = []
    for prev, cur in zip(stamps, stamps[1:]):
        if cur - prev > expected_s + gap_s:
            out.append((prev - stamps[0], (cur - prev) * 1000))
    return out


def find_bursts(stamps: list[float], cpu_s: list[float], window_s: float,
                min_share: float) -> list[tuple[float, float]]:
    """Return (t_since_start_s, cpu_ms) for windows where the process used > min_share of a core.

    ``cpu_s`` is cumulative process CPU time sampled at each stamp. Windows are
    non-overlapping, ``window_s`` long.
    """
    if len(stamps) != len(cpu_s) or len(stamps) < 2:
        return []
    out = []
    i = 0
    for j in range(1, len(stamps)):
        if stamps[j] - stamps[i] >= window_s:
            used = cpu_s[j] - cpu_s[i]
            if used > min_share * (stamps[j] - stamps[i]):
                out.append((stamps[i] - stamps[0], used * 1000))
            i = j
    return out


def stat_path(target: str) -> str:
    """'PID' -> whole process, 'PID/TID' -> one thread."""
    pid, _, tid = target.partition("/")
    if not pid.isdigit() or (tid and not tid.isdigit()):
        raise ValueError(f"bad target {target!r}, want PID or PID/TID")
    return f"/proc/{pid}/task/{tid}/stat" if tid else f"/proc/{pid}/stat"


def proc_cpu_s(path: str) -> float:
    """Cumulative user+system CPU seconds from a /proc stat file."""
    with open(path) as f:
        fields = f.read().rsplit(")", 1)[1].split()
    return (int(fields[11]) + int(fields[12])) / os.sysconf("SC_CLK_TCK")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--seconds", type=float, default=30.0)
    parser.add_argument("--tick-ms", type=float, default=2.0)
    parser.add_argument("--gap-ms", type=float, default=20.0)
    parser.add_argument("--pid", action="append", default=[],
                        help="also report CPU bursts of PID or PID/TID (repeatable)")
    parser.add_argument("--burst-share", type=float, default=0.5,
                        help="CPU share of one core in a 50 ms window counted as a burst")
    args = parser.parse_args()
    if args.seconds <= 0 or args.tick_ms <= 0:
        parser.error("--seconds and --tick-ms must be positive")
    try:
        paths = {t: stat_path(t) for t in args.pid}
        cpu = {t: [proc_cpu_s(p)] for t, p in paths.items()}
    except (OSError, ValueError) as exc:
        parser.error(f"cannot read process: {exc}")
    tick = args.tick_ms / 1000
    stamps = [time.monotonic()]
    while stamps[-1] - stamps[0] < args.seconds:
        time.sleep(tick)
        stamps.append(time.monotonic())
        for pid, samples in cpu.items():
            try:
                samples.append(proc_cpu_s(paths[pid]))
            except OSError:
                samples.append(samples[-1])
    gaps = find_gaps(stamps, tick, args.gap_ms / 1000)
    print(f"start_mono {stamps[0]:.3f}  ticks {len(stamps)}  gaps>{args.gap_ms:g} ms: {len(gaps)}")
    for t, ms in gaps:
        print(f"    gap {ms:.0f} ms at {t:.2f} s")
    for pid, samples in cpu.items():
        bursts = find_bursts(stamps, samples, 0.05, args.burst_share)
        print(f"pid {pid}: {len(bursts)} CPU bursts")
        for t, ms in bursts:
            print(f"    burst {ms:.0f} ms CPU at {t:.2f} s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
