"""Summarise the listen probe (interactive mode step 0): how much speech, and could transcription keep up?

Reads ~/homeai-bench/interactive-listen-*.jsonl (written by tools/bench_interactive_listen.py) and prints hours of
speech, segments, and lag / real-time factor (p50, p95, max) overall, while an art job ran, and while Jarvis
was talking. The go/no-go (plans/INTERACTIVE_MODE_PLAN.md step 0): lag p95 under 2 s with an art job running.

Usage: .venv/bin/python tools/report_interactive_listen.py [jsonl ...]
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

BENCH = Path.home() / "homeai-bench"


def pct(values: list[float], p: float) -> float | None:
    if not values:
        return None
    values = sorted(values)
    return values[min(len(values) - 1, int(round(p / 100 * (len(values) - 1))))]


def summary(rows: list[dict]) -> dict:
    ok = [r for r in rows if r.get("ok")]
    noise = [r for r in rows if not r.get("ok") and "minimum length" in (r.get("error") or "")]
    lag = [r["lag_s"] for r in ok]
    rtf = [r["rtf"] for r in ok]
    # Noise (whisper heard no words: a door, the fridge) is not a failure, and not speech: only clips with
    # words count toward the ~10 hours step 0 needs.
    return {"segments": len(rows), "noise": len(noise), "failed": len(rows) - len(ok) - len(noise),
            "speech_h": round(sum(float(r.get("duration") or 0) for r in ok) / 3600, 2),
            "lag_p50": pct(lag, 50), "lag_p95": pct(lag, 95), "lag_max": max(lag, default=None),
            "rtf_p50": pct(rtf, 50), "rtf_p95": pct(rtf, 95)}


def report(rows: list[dict]) -> str:
    groups = {"all": rows,
              "art job running": [r for r in rows if r.get("comfy_busy")],
              "art mode on": [r for r in rows if r.get("art_active")],
              "Jarvis talking": [r for r in rows if r.get("jarvis_speaking")]}
    lines = []
    for name, group in groups.items():
        s = summary(group)
        lines.append(f"{name:16} " + " ".join(f"{k}={v}" for k, v in s.items()))
    art = summary(groups["art job running"])
    verdict = ("no art-job data yet" if not art["lag_p95"] and art["lag_p95"] != 0 else
               "GO" if art["lag_p95"] < 2.0 else "NO-GO")
    lines.append(f"step 0 (lag p95 < 2 s with an art job running): {verdict}")
    return "\n".join(lines)


def load_rows(paths: list[Path]) -> list[dict]:
    rows = []
    for path in paths:
        for line in path.read_text().splitlines():
            try:
                rows.append(json.loads(line))
            except ValueError:
                continue
    return rows


def main(argv: list[str]) -> int:
    paths = [Path(a) for a in argv] or sorted(BENCH.glob("interactive-listen-*.jsonl"))
    print(report(load_rows(paths)))
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
