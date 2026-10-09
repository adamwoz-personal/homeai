#!/usr/bin/env python3
"""Experiment: which Alexa text command reliably cancels a timer, and how soon?

Seen live 2026-10-09: "set a test timer for 1 minute", then 3 s later
"cancel the test timer". HA accepted both, but the timer still rang.

Each trial sets a 10-minute timer (so a failed cancel never rings mid-test),
waits DELAY seconds after the set command, sends a cancel phrase, then polls the Echo's
next_timer sensor (which lags about 90 s) to see whether the timer is gone.
Every run ends with "cancel all timers" as cleanup.

    .venv/bin/python tools/ha/timer_cancel_experiment.py
    .venv/bin/python tools/ha/timer_cancel_experiment.py --trial 15 "cancel the {label} timer"
"""
from __future__ import annotations

import argparse
import sys
import time
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from homeai.home import build_home  # noqa: E402

DEFAULT_TRIALS = [
    (3, "cancel the {label} timer"),
    (8, "cancel the {label} timer"),
    (15, "cancel the {label} timer"),
]


def sensor(home, echo) -> str:
    for s in home.client.states():
        if s["entity_id"].endswith("_next_timer") and echo.name.lower().split()[0] in s["entity_id"]:
            return s["state"]
    return "?"


def running(state: str) -> bool:
    """The sensor keeps a rung timer's past timestamp; only a future one counts."""
    try:
        return datetime.fromisoformat(state).timestamp() > time.time()
    except (TypeError, ValueError):
        return False


def wait_for(home, echo, want_timer: bool, limit: float = 180) -> float | None:
    start = time.time()
    while time.time() - start < limit:
        state = sensor(home, echo)
        if running(state) == want_timer:
            return time.time() - start
        time.sleep(5)
    return None


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--trial", nargs=2, action="append", metavar=("DELAY", "PHRASE"))
    ap.add_argument("--echo", default=None)
    args = ap.parse_args()
    trials = [(float(d), p) for d, p in args.trial] if args.trial else DEFAULT_TRIALS

    home = build_home()
    echo = home.echo(args.echo)
    print(f"echo: {echo.name}; sensor now {sensor(home, echo)}", flush=True)
    results = []
    try:
        for i, (delay, phrase) in enumerate(trials):
            label = ["apple", "banana", "cherry", "grape", "lemon"][i % 5]
            # DELAY counts from the set command, like the daemon's own
            # set-then-cancel; waiting for the laggy sensor first hid the bug.
            home._alexa(echo, f"set a {label} timer for 10 minutes")
            time.sleep(delay)
            home._alexa(echo, phrase.format(label=label))
            print(f"trial {i}: {label} set, cancel sent {delay}s later", flush=True)
            # The sensor lags ~90 s; an early "no timer" would be a false pass.
            time.sleep(150)
            state = sensor(home, echo)
            ok = not running(state)
            print(f"trial {i}: sensor {state}", flush=True)
            gone = 150 if ok else None
            results.append((delay, phrase, ok, gone))
            print(f"trial {i}: '{phrase}' after {delay}s -> {'CANCELLED' if ok else 'STILL RUNNING'}"
                  f"{f' (sensor cleared after {gone:.0f}s)' if ok else ''}", flush=True)
            if not ok:
                home._alexa(echo, "cancel all timers")
                wait_for(home, echo, False)
    finally:
        home._alexa(echo, "cancel all timers")
    print("\nsummary:")
    for delay, phrase, ok, _ in results:
        print(f"  {'OK  ' if ok else 'FAIL'} delay={delay:>4}s  {phrase}")
    return sum(not r[2] for r in results)


if __name__ == "__main__":
    sys.exit(main())
