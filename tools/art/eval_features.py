#!/usr/bin/env python3
"""How well does the voice model keep the lettered features? (ART_FEATURES_PLAN F1)

Why: features only help if "make the dragon red" changes the dragon's letter
instead of adding a second dragon, complaints become plain fixes, and rambling
descriptions come out clean. This runs fixed instructions through
homeai/art/patcher.features_patch_for against the real voice model (direct
Ollama call, no tools: nothing in the house can act), checks each result, and
reports pass rate and time.

Writes ~/homeai-bench/features-<time>/result.json.

Usage:
    .venv/bin/python tools/art/eval_features.py [--repeats 2] [--only NAME]
"""
from __future__ import annotations

import argparse
import json
import statistics
import sys
from datetime import datetime
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

from homeai.art import features as F  # noqa: E402
from homeai.art import patcher as P  # noqa: E402

DRAGON = {"features": [{"id": "A", "kind": "subject", "text": "a single green dragon with one head and four legs"},
                       {"id": "B", "kind": "element", "text": "one samurai in full armor sitting on the dragon's back"},
                       {"id": "C", "kind": "setting", "text": "a neon city street at night"},
                       {"id": "D", "kind": "style", "text": "cyberpunk"}], "aspect": "landscape"}


def text(spec, letter):
    return next((f["text"].lower() for f in spec.get("features", []) if f["id"] == letter), "")


def count(spec, kind=None):
    return len([f for f in spec.get("features", []) if kind is None or f["kind"] == kind])


def has_word(spec, *words):
    return any(w in " ".join(f["text"].lower() for f in spec.get("features", [])) for w in words)


CASES = [
    ("new rider", {}, "draw a dragon with a samurai riding on its back",
     [lambda s, c: count(s, "subject") == 1 or "not one subject",
      lambda s, c: has_word(s, "samurai") or "no samurai",
      lambda s, c: not has_word(s, "neon", "cyberpunk", "lighting") or "invented a look"]),
    ("red merges", DRAGON, "make the dragon red",
     [lambda s, c: "red" in text(s, "A") or f"A is {text(s, 'A')!r}",
      lambda s, c: count(s) == 4 or f"{count(s)} features (added one?)"]),
    ("by letter", DRAGON, "change B to a knight",
     [lambda s, c: "knight" in text(s, "B") or f"B is {text(s, 'B')!r}", lambda s, c: count(s) == 4 or "count"]),
    ("remove letter", DRAGON, "remove C", [lambda s, c: text(s, "C") == "" and count(s) == 3 or "C still there"]),
    ("complaint", DRAGON, "we ended up with a second head and a second rider this time",
     [lambda s, c: "two" not in " ".join(f["text"] for f in s["features"]).lower() or "says two",
      lambda s, c: count(s, "subject") == 1 or "subjects"]),
    ("leg merged", DRAGON, "the samurai's leg turned into one of the dragon's legs",
     [lambda s, c: count(s) <= 5 or "too many", lambda s, c: "dragon" in text(s, "A") or "lost the dragon"]),
    ("ramble", {}, "um, a dragon, no wait, a wyvern, a green one, flying over, uh, mountains I guess",
     [lambda s, c: has_word(s, "wyvern") or "no wyvern", lambda s, c: not has_word(s, "dragon") or "kept the dragon",
      lambda s, c: has_word(s, "mountain") or "no mountains"]),
    ("add thing", DRAGON, "add a full moon behind them",
     [lambda s, c: has_word(s, "moon") or "no moon", lambda s, c: count(s) == 5 or f"{count(s)} features"]),
    ("style swap", DRAGON, "make it a watercolour instead of cyberpunk",
     [lambda s, c: "watercolo" in text(s, "D") or f"D is {text(s, 'D')!r}", lambda s, c: count(s) == 4 or "count"]),
]


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--repeats", type=int, default=2)
    ap.add_argument("--only", default="")
    args = ap.parse_args()
    out = Path.home() / "homeai-bench" / f"features-{datetime.now():%Y%m%d-%H%M%S}"
    out.mkdir(parents=True)
    rows, times = [], []
    for name, start, said, checks in CASES:
        if args.only and args.only not in name:
            continue
        for n in range(args.repeats):
            result = P.features_patch_for(start, said)
            problems = [r for c in checks for r in [c(result.spec, result.changed)] if r is not True] if result.ok \
                else [f"not caught: {result.detail}"]
            times.append(result.seconds)
            rows.append({"case": name, "run": n, "ok": not problems, "problems": problems, "ops": result.patch,
                         "features": result.spec.get("features") if result.ok else None, "seconds": result.seconds})
            print(f"{name:14} {n} {'PASS' if not problems else 'FAIL'} {result.seconds:5.2f}s "
                  f"{json.dumps(result.patch.get('ops'))[:160] if result.ok else ''} {'; '.join(problems)}", flush=True)
    passed = sum(r["ok"] for r in rows)
    summary = {"passed": passed, "runs": len(rows), "median_s": statistics.median(times), "max_s": max(times)}
    (out / "result.json").write_text(json.dumps({"summary": summary, "rows": rows}, indent=1))
    print(json.dumps(summary))
    print(out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
