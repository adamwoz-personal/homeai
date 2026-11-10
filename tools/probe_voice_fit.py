#!/usr/bin/env python3
# Copyright (c) 2026 Adam Wosotowsky
# SPDX-License-Identifier: BUSL-1.1
# Licensed under the Business Source License 1.1; see LICENSE at the
# repository root. Converts to Apache-2.0 on 2030-09-15.
"""Does a voice-model candidate fit on the card beside the coder at Jarvis's context? (LLM_UPGRADE_PLAN step 1b)

Why: eval_tool_misfire.py's default route left the context at Ollama's default,
so it said nothing about memory. Jarvis runs its model at num_ctx 16384
(deploy/ollama/llama31-voice.Modelfile) with the coder (llama-server) on the
same card; a model that spills onto the CPU answers several times slower.

For each model and rep: unload whatever Ollama holds, load the model with
Jarvis's real system prompt and tools at num_ctx 16384 (one cold question),
time a few warm questions, then read Ollama's /api/ps (size, size_vram,
context) and the card's VRAM use. The card's own reading counts: on
2026-11-10 /api/ps reported only gemma4's 0.31 GB speculative drafter
("gemma4-assistant"), not the 6 GB model; ``card_model_gb`` is the card's
use with the model loaded minus its use with nothing loaded. Talks only to Ollama: tool calls in the
replies are never run. Jarvis is paused for the run and comes back with its
own model reloaded wholly on the GPU (``gpu_mode.ModeSwitcher.jarvis_paused``).

    .venv/bin/python tools/probe_voice_fit.py --prompt ~/homeai-bench/zc-prompt-<date>.json \\
        --model llama31-voice --model qwen3.5:9b --model gemma4:e4b [--reps 2]

Writes ~/homeai-bench/voicefit-<time>/result.json and prints a markdown table.
"""
from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "tools"))

import eval_tool_misfire as M  # noqa: E402
from homeai import gpu_info  # noqa: E402
from homeai.benchio import bench_path  # noqa: E402
from homeai.httpjson import OLLAMA, post_json  # noqa: E402

CTX = 16384     # must match request_for's num_ctx and Jarvis's Modelfile
COLD_QUESTION = "Tell me a joke."
WARM_QUESTIONS = ["Why is the sky blue?", "What's the capital of France?", "Tell me about yourself."]
GB = 1024 ** 3


def loaded(post=post_json) -> list[dict]:
    return post(f"{OLLAMA}/api/ps").get("models", [])


def unload_all(post=post_json, sleep=time.sleep) -> None:
    for m in loaded(post):
        post(f"{OLLAMA}/api/generate", {"model": m["name"], "keep_alive": 0}, 60)
    sleep(3)


def ps_entry(model: str, post=post_json) -> dict:
    for m in loaded(post):
        if m.get("name") in (model, f"{model}:latest"):
            return m
    return {}


def card() -> tuple[float, float]:
    c = gpu_info.biggest()
    return (round(c.used / GB, 2), round(c.total / GB, 2)) if c else (0.0, 0.0)


def measure(capture: dict, model: str, post=post_json, card=card, sleep=time.sleep) -> dict:
    """One cold load plus warm questions; the model stays loaded for the reading."""
    unload_all(post, sleep)
    empty, _ = card()
    state = {"endpoint": "native"}
    cold = M.ask(capture, model, COLD_QUESTION, state, post)
    warm = [M.ask(capture, model, q, state, post) for q in WARM_QUESTIONS]
    entry = ps_entry(model, post)
    size, vram = entry.get("size", 0), entry.get("size_vram", 0)
    used, total = card()
    return {"cold_s": cold["seconds"], "warm_s": [w["seconds"] for w in warm],
            "size_gb": round(size / GB, 2), "vram_gb": round(vram / GB, 2),
            "gpu_percent": round(100 * vram / size, 1) if size else 0.0,
            "context": entry.get("context_length"), "card_model_gb": round(used - empty, 2), "card_used_gb": used, "card_total_gb": total,
            "headroom_gb": round(total - used, 2)}


def verdict(rows: list[dict]) -> str:
    if not rows or any(r["gpu_percent"] < 100 for r in rows):
        return "spills"
    if any(r["context"] not in (None, CTX) for r in rows):
        return "wrong context"
    return "fits"


def run(capture: dict, models: list[str], reps: int, out: Path, post=post_json, card=card,
        sleep=time.sleep, log=print) -> dict:
    data: dict = {"num_ctx": CTX, "models": {}}
    try:
        for model in models:
            rows = []
            for rep in range(reps):
                row = measure(capture, model, post, card, sleep)
                rows.append(row)
                log(f"{model} rep {rep}: {row}")
            data["models"][model] = {"verdict": verdict(rows), "rows": rows}
            (out / "result.json").write_text(json.dumps(data, indent=1))
    finally:
        unload_all(post, sleep)    # leave the card to jarvis_paused's reload
    return data


def table(data: dict) -> str:
    lines = ["| model | verdict | size GB | on GPU % | context | on card GB | card used GB | headroom GB | cold s | warm median s |",
             "|---" * 10 + "|"]
    for model, d in data["models"].items():
        rows = d["rows"]
        warm = [s for r in rows for s in r["warm_s"]]
        lines.append(f"| {model} | {d['verdict']} | {max(r['size_gb'] for r in rows)} | "
                     f"{min(r['gpu_percent'] for r in rows)} | {rows[0]['context']} | {max(r['card_model_gb'] for r in rows)} | "
                     f"{max(r['card_used_gb'] for r in rows)} | {min(r['headroom_gb'] for r in rows)} | "
                     f"{max(r['cold_s'] for r in rows)} | {statistics.median(warm) if warm else 0} |")
    return "\n".join(lines)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--prompt", required=True, help="a capture from probe_agent_prompt.py --save")
    ap.add_argument("--model", action="append", required=True)
    ap.add_argument("--reps", type=int, default=2)
    args = ap.parse_args()
    capture = json.loads(Path(args.prompt).expanduser().read_text())[0]
    out = bench_path("voicefit")
    out.mkdir(parents=True)
    from homeai import gpu_mode as G
    with G.ModeSwitcher(G.System(), log=lambda line: print("[mode]", line, flush=True)).jarvis_paused():
        data = run(capture, args.model, args.reps, out)
    print(table(data))
    print(out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
