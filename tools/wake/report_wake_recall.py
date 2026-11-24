#!/usr/bin/env python3
"""Break a wake bench on synthetic phrases down by noise, speed, voice and sound-alike (plans/WAKE_WORD_PLAN.md).

Why: bench_wake_word.py gives one recall number; to see *why* a model misses (noise? fast speech? one voice?) and
which sound-alikes wake it, this joins its result.json with make_test_phrases.py's index.json.

Usage:
    .venv/bin/python tools/wake/report_wake_recall.py BENCH_DIR PHRASES_DIR
"""
from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path


def table(index: list[dict], hit: set[str], kind: str, key: str) -> dict[str, str]:
    groups = defaultdict(lambda: [0, 0])
    for item in index:
        if item["kind"] != kind:
            continue
        g = groups[str(item[key])]
        g[0] += item["file"] in hit
        g[1] += 1
    return {k: f"{a}/{b}" for k, (a, b) in sorted(groups.items())}


def report(result: dict, index: list[dict]) -> dict:
    out = {"threshold": result["threshold"], "models": []}
    for m in result["models"]:
        woke = set(f["clip"] for f in m.get("false", []))                  # negatives that fired
        missed = set(m.get("recall", {}).get("missed", []))
        hit = {i["file"] for i in index if i["kind"] == "positive"} - missed
        out["models"].append({
            "model": m["model"],
            "recall": m.get("recall", {}).get("rate"),
            "recall_by_noise": table(index, hit, "positive", "noise_db_below"),
            "recall_by_speed": table(index, hit, "positive", "speed"),
            "recall_by_voice": table(index, hit, "positive", "voice"),
            "recall_by_text": table(index, hit, "positive", "text"),
            "sound_alike_wakes": table(index, woke, "negative", "text"),
        })
    return out


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("bench", type=Path)
    ap.add_argument("phrases", type=Path)
    args = ap.parse_args(argv)
    out = report(json.loads((args.bench / "result.json").read_text()),
                 json.loads((args.phrases / "index.json").read_text()))
    print(json.dumps(out, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
