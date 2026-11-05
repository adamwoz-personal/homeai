#!/usr/bin/env python3
"""How much does image generation slow the voice model? (AVATAR_PLAN Phase 0, 7.9)

Why: Jarvis has to stay usable while a picture is drawing. This times the voice
model's first token and token rate with the GPU quiet, then during a generation.
It asks Ollama directly with a fixed prompt: no agent and no tools, so nothing can
act on the house. Results go to ~/homeai-bench/contention-<time>.json.

Usage:
    .venv/bin/python tools/art/contention_bench.py [--model zimage] [--rounds 5]
"""
from __future__ import annotations

import argparse
import json
import statistics
import sys
import threading
import time
import urllib.request
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from comfy_common import MODELS, OLLAMA, Comfy, CoderStopped, workflow  # noqa: E402

VOICE = "llama31-voice"
QUESTION = "In two sentences, why is the sky blue?"
LOOP_PROMPT = "a misty mountain valley at dawn, oil painting"


def ask() -> dict:
    """First-token seconds and tokens per second for one short reply."""
    body = json.dumps({"model": VOICE, "prompt": QUESTION, "stream": True, "keep_alive": -1,
                       "options": {"num_predict": 60, "seed": 1}}).encode()
    start, first, last = time.monotonic(), None, {}
    request = urllib.request.Request(f"{OLLAMA}/api/generate", data=body)
    with urllib.request.urlopen(request, timeout=300) as reply:
        for line in reply:
            last = json.loads(line)
            if first is None and last.get("response"):
                first = time.monotonic() - start
    rate = last.get("eval_count", 0) / max(last.get("eval_duration", 1) / 1e9, 1e-6)
    return {"first_token_s": round(first or 0.0, 3), "tokens_per_s": round(rate, 1)}


def summary(rows: list[dict]) -> dict:
    return {"first_token_s_median": statistics.median(r["first_token_s"] for r in rows),
            "first_token_s_worst": max(r["first_token_s"] for r in rows),
            "tokens_per_s_median": statistics.median(r["tokens_per_s"] for r in rows), "rounds": len(rows)}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--model", action="append", choices=MODELS)
    ap.add_argument("--rounds", type=int, default=5)
    args = ap.parse_args()
    out = Path.home() / "homeai-bench" / f"contention-{datetime.now():%Y%m%d-%H%M%S}.json"
    results = {}
    ask()                                               # warm
    with CoderStopped(), Comfy(out.with_suffix(".log")) as comfy:
        results["quiet"] = summary([ask() for _ in range(args.rounds)])
        print("quiet", results["quiet"], flush=True)
        for model in args.model or MODELS:
            comfy.run(workflow(model, LOOP_PROMPT, seed=1))                 # loaded before timing
            stop = threading.Event()

            def draw() -> None:
                seed = 2
                while not stop.is_set():
                    comfy.run(workflow(model, LOOP_PROMPT, seed=seed))
                    seed += 1

            worker = threading.Thread(target=draw, daemon=True)
            worker.start()
            time.sleep(2.0)
            rows = [ask() for _ in range(args.rounds)]
            stop.set()
            worker.join()
            results[model] = summary(rows)
            print(model, results[model], flush=True)
            out.write_text(json.dumps(results, indent=1))
    print(out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
