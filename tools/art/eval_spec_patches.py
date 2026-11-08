#!/usr/bin/env python3
"""How well does the voice model turn spoken picture changes into PromptSpec patches? (AVATAR_PLAN 6a step 6)

Why: "make it bluer" only works if the model puts blue in `palette` and leaves
the subject alone. This runs a fixed set of instructions through
homeai/art/patcher.py against the real voice model (direct Ollama call, no
tools, so nothing in the house can act), checks each result, repeats each case,
and reports pass rate and latency. Jarvis keeps running; the model stays pinned.

Writes ~/homeai-bench/spec-patches-<time>/result.json.

Usage:
    .venv/bin/python tools/art/eval_spec_patches.py [--repeats 3]
"""
from __future__ import annotations

import argparse
import json
import statistics
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

from homeai.art import patcher as P  # noqa: E402
from homeai.art import prompt_spec as PS  # noqa: E402
from homeai.benchio import bench_path  # noqa: E402

FOX = {"subject": "a red fox", "setting": "sitting in the snow"}


def has(field: str, *words: str):
    return lambda before, after: any(w in str(after.get(field, "")).lower() for w in words) or \
        f"{field} lacks {'/'.join(words)}: {after.get(field)!r}"


def any_of(fields: tuple[str, ...], *words: str):
    return lambda before, after: any(w in str(after.get(f, "")).lower() for f in fields for w in words) or \
        f"none of {fields} has {'/'.join(words)}"


def same(field: str):
    return lambda before, after: before.get(field) == after.get(field) or \
        f"{field} changed: {before.get(field)!r} -> {after.get(field)!r}"


def cleared(field: str):
    return lambda before, after: not after.get(field) or f"{field} still {after.get(field)!r}"


def equals(field: str, value):
    return lambda before, after: after.get(field) == value or f"{field} is {after.get(field)!r}, wanted {value!r}"


# Adam's first live session (2026-11-07): the voice model filled in everything, and "both cyberpunk" wasn't caught.
DRAGON = {"style": "anime", "aspect": "landscape", "subject": "dragon", "details": "samurai riding on its back",
          "setting": "medieval Japan", "palette": "fire, water, earth", "lighting": "dramatic, with a sunset in the background",
          "camera": "low-angle, looking up at the dragon", "composition": "dynamic, with the dragon and samurai in motion",
          "negative": "none"}

CASES = [
    ("new picture", {}, "draw a red fox sitting in the snow", [has("subject", "fox"), any_of(("subject", "setting"), "snow")]),
    ("bluer", FOX, "make it bluer", [has("palette", "blue"), same("subject")]),
    ("night", FOX, "make it night time", [any_of(("setting", "lighting"), "night"), same("subject")]),
    ("fog", FOX, "add some fog", [any_of(("details", "setting", "lighting"), "fog", "mist"), same("subject")]),
    ("watercolour", FOX, "make it a watercolour", [has("style", "watercolo"), same("subject")]),
    ("wide", FOX, "make it a wide picture", [equals("aspect", "wide"), same("subject")]),
    ("new subject", FOX, "make it a wolf instead", [has("subject", "wolf")]),
    ("remove colour", {**FOX, "palette": "blue"}, "get rid of the blue", [cleared("palette"), same("subject")]),
    ("camera", FOX, "show it from a low angle", [has("camera", "low"), same("subject")]),
    ("leave out", FOX, "no people in it", [has("negative", "people"), same("subject")]),
    ("reference", FOX, "use Mira", [lambda b, a: "Mira" in a.get("references", []) or f"references {a.get('references')}",
                                    same("subject")]),
    ("both cyberpunk", DRAGON, "make the dragon and the samurai both cyberpunk",
     [any_of(("subject", "details", "style"), "cyberpunk"), has("subject", "dragon")]),
    ("correction", DRAGON, "that should have been a dragon with a samurai riding on its back",
     [has("subject", "dragon"), any_of(("subject", "details"), "samurai")]),
    ("lets make it", DRAGON, "let's make it a cyberpunk dragon and a cyberpunk samurai",
     [any_of(("subject", "details", "style"), "cyberpunk")]),
    ("complaint heads", {**DRAGON, "details": "being ridden by a cyberpunk samurai, with a more skeletal appearance"},
     "we ended up with a second head and a second rider this time",
     [lambda b, a: "two" not in str(a.get("details", "")).lower() or f"details still say two: {a.get('details')!r}",
      any_of(("details", "negative", "composition"), "one", "single", "second head", "extra")]),
    ("complaint hand", DRAGON, "I don't know what that samurai has in his right hand but it needs to go",
     [lambda b, a: "right hand" not in str(a.get("details", "")).lower() or "kept the hand thing in details",
      has("subject", "dragon")]),
    ("no invented refs", DRAGON, "make it an undead dragon",
     [has("subject", "undead"), lambda b, a: not a.get("references") or f"references {a.get('references')}"]),
    ("no invented style", {}, "draw a dragon with a samurai riding on its back",
     [has("subject", "dragon"), lambda b, a: not a.get("style") or f"style invented: {a.get('style')!r}",
      lambda b, a: not a.get("palette") or f"palette invented: {a.get('palette')!r}"]),
    ("composition", FOX, "put the fox small in the corner", [any_of(("composition", "details"), "corner"),
                                                             has("subject", "fox")]),
]


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--repeats", type=int, default=3)
    ap.add_argument("--only", default="", help="run only cases whose name contains this")
    args = ap.parse_args()
    out = bench_path("spec-patches")
    out.mkdir(parents=True)
    rows, times = [], []
    for name, start, instruction, checks in CASES:
        if args.only and args.only not in name:
            continue
        before = PS.validate(start) if start else {}
        for n in range(args.repeats):
            result = P.patch_for(before, instruction)
            problems = [r for c in checks for r in [c(before, result.spec)] if r is not True] if result.ok \
                else [f"not caught: {result.message} {result.detail}"]
            times.append(result.seconds)
            rows.append({"case": name, "run": n, "ok": not problems, "problems": problems, "patch": result.patch,
                         "seconds": result.seconds})
            print(f"{name:14} {n} {'PASS' if not problems else 'FAIL'} {result.seconds:5.2f}s {result.patch} "
                  f"{'; '.join(problems)}", flush=True)
    passed = sum(r["ok"] for r in rows)
    summary = {"passed": passed, "runs": len(rows), "median_s": statistics.median(times), "max_s": max(times),
               "model": P.MODEL}
    (out / "result.json").write_text(json.dumps({"summary": summary, "rows": rows}, indent=1))
    print(json.dumps(summary))
    print(out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
