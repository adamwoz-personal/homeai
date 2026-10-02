#!/usr/bin/env python3
# Copyright (c) 2026 Adam Wosotowsky
# SPDX-License-Identifier: BUSL-1.1
# Licensed under the Business Source License 1.1; see LICENSE at the
# repository root. Converts to Apache-2.0 on 2030-09-15.
"""Measure raw llama-server throughput: prompt processing and generation.

Why this exists
---------------
``bench_llm.py`` judges answer *quality*. Choosing a KV-cache type or a
CPU/GPU layer split is a pure speed question, and for a coding agent two
speeds matter separately:

* **prompt processing** (tokens/s read) -- a coding agent resends a large
  context (files, tool output) on every step, so this dominates wall time;
* **generation** (tokens/s written).

llama-server reports both in the ``timings`` block of ``/completion``, so this
reads them from the server rather than timing HTTP round trips.

Usage
-----
    bench_tokps.py                         # localhost:8080, 3 reps
    bench_tokps.py --url http://127.0.0.1:8080 --ctx-tokens 16000 --reps 5
    bench_tokps.py --label q4_0 --out ~/homeai-bench/kv.jsonl   # append result
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
import urllib.request
from pathlib import Path

# Varied code-like filler, so prompt processing is not measured on a
# degenerate repeated token that a cache could shortcut.
_FILLER = """
def handle_{n}(request, cache):
    key = request.path + ":" + str(request.user_id)
    if key in cache and cache[key].fresh({n}):
        return cache[key].value
    result = compute_{n}(request.args, limit={n})
    cache[key] = Entry(result, ttl={n} * 3)
    return result
"""


def build_prompt(target_tokens: int) -> str:
    # ~60 tokens per block for this tokenizer family.
    blocks = max(1, target_tokens // 60)
    body = "".join(_FILLER.format(n=i) for i in range(blocks))
    return (
        "Here is a Python module.\n```python\n" + body + "```\n"
        "Write a detailed explanation of what this module does, then suggest "
        "three improvements with code.\n"
    )


def completion(url: str, prompt: str, n_predict: int, timeout: float) -> dict:
    payload = json.dumps({
        "prompt": prompt,
        "n_predict": n_predict,
        "temperature": 0.0,
        "cache_prompt": False,  # measure real prompt processing every rep
        "ignore_eos": True,     # fixed-length generation for a fair rate
    }).encode()
    req = urllib.request.Request(
        url.rstrip("/") + "/completion", data=payload,
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return json.loads(resp.read())


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--url", default="http://127.0.0.1:8080")
    ap.add_argument("--ctx-tokens", type=int, default=8000,
                    help="approximate prompt length")
    ap.add_argument("--n-predict", type=int, default=400)
    ap.add_argument("--reps", type=int, default=3)
    ap.add_argument("--timeout", type=float, default=600)
    ap.add_argument("--label", default="")
    ap.add_argument("--out", help="append a JSON line with the result")
    args = ap.parse_args(argv)

    prompt = build_prompt(args.ctx_tokens)
    pp, tg, n_prompt = [], [], 0
    for rep in range(args.reps):
        started = time.monotonic()
        try:
            data = completion(args.url, prompt, args.n_predict, args.timeout)
        except OSError as exc:
            print(f"error: {exc}", file=sys.stderr)
            return 1
        t = data.get("timings", {})
        n_prompt = t.get("prompt_n", 0)
        pp.append(t.get("prompt_per_second", 0.0))
        tg.append(t.get("predicted_per_second", 0.0))
        print(f"rep {rep + 1}: prompt {n_prompt} tok @ {pp[-1]:7.1f} tok/s, "
              f"gen {t.get('predicted_n', 0)} tok @ {tg[-1]:6.1f} tok/s, "
              f"wall {time.monotonic() - started:5.1f}s")

    result = {
        "label": args.label, "prompt_tokens": n_prompt,
        "pp_median": round(statistics.median(pp), 1),
        "tg_median": round(statistics.median(tg), 1),
        "reps": args.reps, "at": time.strftime("%Y-%m-%dT%H:%M:%S"),
    }
    print(f"\n{args.label or 'result'}: prompt {result['pp_median']} tok/s, "
          f"generation {result['tg_median']} tok/s (median of {args.reps})")
    if args.out:
        out = Path(args.out).expanduser()
        out.parent.mkdir(parents=True, exist_ok=True)
        with out.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(result) + "\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
