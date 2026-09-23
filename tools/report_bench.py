#!/usr/bin/env python3
# Copyright (c) 2026 Adam Wosotowsky
# SPDX-License-Identifier: BUSL-1.1
# Licensed under the Business Source License 1.1; see LICENSE at the
# repository root. Converts to Apache-2.0 on 2030-09-15.
"""Summarise a bench_llm.py JSON run.

Why this is separate from bench_llm.py
--------------------------------------
A benchmark run takes minutes and holds a model in VRAM. Re-running it just to
re-read the numbers is wasteful and, on this machine, risky -- loading models
while llama-server is resident has OOM'd the host. The raw responses are saved
to JSON precisely so analysis can be repeated for free.

It also answers the question bench_llm.py deliberately refuses to: *which
prompts* fail, and *how consistently*. A model that fails one prompt every
time is a different problem from one that fails three prompts at random.

Usage
-----
    report_bench.py ~/bench-voice-20260923.json
    report_bench.py ~/bench-voice-20260923.json --prompt list_trap --show-text
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
from collections import defaultdict
from pathlib import Path


def median(values: list[float]) -> float:
    return statistics.median(values) if values else float("nan")


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("path", help="JSON written by bench_llm.py --out")
    ap.add_argument("--prompt", help="restrict to a single prompt name")
    ap.add_argument("--show-text", action="store_true", help="print responses")
    ap.add_argument(
        "--max-chars", type=int, default=400, help="truncate shown text"
    )
    ap.add_argument(
        "--speech-check",
        action="store_true",
        help="re-run each response through homeai.speech.normalise_for_speech "
             "and report what markdown survives. Answers the question the raw "
             "flags cannot: does the TTS sanitiser actually cover what these "
             "models emit, or would it reach Piper?",
    )
    args = ap.parse_args(argv)

    try:
        with open(args.path, encoding="utf-8") as fh:
            rows = json.load(fh)
    except OSError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2

    if args.prompt:
        rows = [r for r in rows if r["prompt"] == args.prompt]
        if not rows:
            print(f"error: no rows for prompt {args.prompt!r}", file=sys.stderr)
            return 2

    endpoints = sorted({r["endpoint"] for r in rows})

    # ---- per-endpoint headline ------------------------------------------
    print(f"{'endpoint':<10} {'runs':>5} {'clean':>7} {'unspeak':>8} "
          f"{'hedged':>7} {'ttft med':>9} {'total med':>10} {'chars med':>10}")
    print("-" * 78)
    for ep in endpoints:
        er = [r for r in rows if r["endpoint"] == ep]
        ok = [r for r in er if r["error"] is None]
        clean = sum(
            1 for r in ok
            if not r["unspeakable"] and not r["missing_facts"]
            and not r["too_long"] and not r["too_short"]
        )
        print(
            f"{ep:<10} {len(er):>5} {clean:>4}/{len(er):<2} "
            f"{sum(1 for r in ok if r['unspeakable']):>8} "
            f"{sum(1 for r in ok if r['hedges']):>7} "
            f"{median([r['ttft_ms'] for r in ok if r['ttft_ms']]):>7.0f}ms "
            f"{median([r['total_ms'] for r in ok if r['total_ms']]):>8.0f}ms "
            f"{median([float(r['chars']) for r in ok]):>10.0f}"
        )

    # ---- per-prompt failure consistency ---------------------------------
    print("\nper-prompt failures (fails/runs; blank means always clean)")
    prompts = sorted({r["prompt"] for r in rows})
    header = f"{'prompt':<16}" + "".join(f"{ep:>14}" for ep in endpoints)
    print(header)
    print("-" * len(header))
    for p in prompts:
        line = f"{p:<16}"
        for ep in endpoints:
            pr = [r for r in rows if r["prompt"] == p and r["endpoint"] == ep]
            ok = [r for r in pr if r["error"] is None]
            bad = defaultdict(int)
            for r in ok:
                for flag in r["unspeakable"]:
                    bad[flag] += 1
                for flag in r["hedges"]:
                    bad["hedge"] += 1
                if r["missing_facts"]:
                    bad["facts"] += 1
                if r["too_long"]:
                    bad["long"] += 1
                if r["too_short"]:
                    bad["short"] += 1
            nbad = sum(
                1 for r in ok
                if r["unspeakable"] or r["hedges"] or r["missing_facts"]
                or r["too_long"] or r["too_short"]
            )
            cell = f"{nbad}/{len(ok)}" if nbad else "."
            line += f"{cell:>14}"
        print(line)

    # ---- worst offenders -------------------------------------------------
    print("\nmost common defects")
    for ep in endpoints:
        counts: defaultdict[str, int] = defaultdict(int)
        for r in rows:
            if r["endpoint"] != ep or r["error"] is not None:
                continue
            for flag in r["unspeakable"]:
                counts[flag] += 1
            for flag in r["hedges"]:
                counts[f"hedge:{flag}"] += 1
        top = sorted(counts.items(), key=lambda kv: -kv[1])[:6]
        summary = ", ".join(f"{k}={v}" for k, v in top) or "none"
        print(f"  {ep:<10} {summary}")

    if args.speech_check:
        sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
        from homeai.speech import flatten_markdown, normalise_for_speech
        from homeai.safety import sanitise_for_speech
        from bench_llm import find_unspeakable

        def spoken(text: str) -> str:
            """Exactly the pipeline in daemon.py; order is load-bearing.

            flatten_markdown needs the line breaks that sanitise_for_speech
            collapses, and normalise_for_speech expects the safety guard to
            have already run. Checking any single stage on its own gives a
            misleading answer.
            """
            return normalise_for_speech(sanitise_for_speech(flatten_markdown(text)))

        print("\nspeech pipeline check (defects BEFORE -> AFTER "
              "flatten_markdown + sanitise_for_speech + normalise_for_speech)")
        for ep in endpoints:
            before = after = 0
            residue: defaultdict[str, int] = defaultdict(int)
            examples: dict[str, str] = {}
            for r in rows:
                if r["endpoint"] != ep or r["error"] is not None:
                    continue
                if r["unspeakable"]:
                    before += 1
                left = find_unspeakable(spoken(r["text"]))
                if left:
                    after += 1
                    for flag in left:
                        residue[flag] += 1
                        examples.setdefault(flag, f"{r['prompt']}#{r['rep']+1}")
            detail = ", ".join(
                f"{k}={v} (e.g. {examples[k]})"
                for k, v in sorted(residue.items(), key=lambda kv: -kv[1])
            )
            print(f"  {ep:<10} {before:>2} -> {after:<2} responses with defects"
                  f"{('  residue: ' + detail) if detail else '  (all cleaned)'}")

    if args.show_text:
        for r in rows:
            if r["error"]:
                print(f"\n--- {r['endpoint']} {r['prompt']}#{r['rep']+1} ERROR {r['error']}")
                continue
            flags = r["unspeakable"] + [f"hedge:{h}" for h in r["hedges"]]
            print(f"\n--- {r['endpoint']} {r['prompt']}#{r['rep']+1} "
                  f"[{','.join(flags) or 'ok'}]")
            print(r["text"][: args.max_chars])

    return 0


if __name__ == "__main__":
    sys.exit(main())
