#!/usr/bin/env python3
# Copyright (c) 2026 Adam Wosotowsky
# SPDX-License-Identifier: BUSL-1.1
# Licensed under the Business Source License 1.1; see LICENSE at the
# repository root. Converts to Apache-2.0 on 2030-09-15.
"""Does a voice model call tools when it shouldn't (or miss them when it should)? (LLM_UPGRADE_PLAN step 1)

Why: llama31-voice called a house tool on "tell me about yourself" (2026-11-05),
and Adam dislikes Jarvis looking up trivial things. This asks each candidate a
fixed set of everyday questions with Jarvis's real system prompt and tool list
(captured by ``tools/probe_agent_prompt.py --save``), and scores each answer:
ok / misfire (a tool where none was wanted) / miss (no tool where one was) /
wrong (another tool). Also: words per spoken answer, seconds per answer, and
how much of the model sits on the GPU beside the coder.

By default the request goes to Ollama's OpenAI-compatible endpoint exactly as
ZeroClaw sends it (``--endpoint native`` uses /api/chat with thinking off; on
2026-11-09 both routes gave the same picture). Checked against real ZeroClaw
with ``probe_agent_prompt.py --real-reply``: llama31-voice calls a tool for
"What's the capital of France?" there too.

Never touches the house: it calls Ollama directly, not ZeroClaw,
so a tool call is only a JSON field in the reply; nothing runs it.

Jarvis is stopped for the run (each candidate takes the voice model's place on
the card) and comes back afterwards with its model reloaded wholly on the GPU,
whatever happens (``gpu_mode.ModeSwitcher.jarvis_paused``). The coder stays up.

    .venv/bin/python tools/probe_agent_prompt.py --agent local --save ~/homeai-bench/zc-prompt-<date>.json "hi"
    .venv/bin/python tools/eval_tool_misfire.py --prompt ~/homeai-bench/zc-prompt-<date>.json \\
        --model llama31-voice --model qwen3.5:9b --model gemma4:e4b [--reps 3]

Writes ~/homeai-bench/misfire-<time>/result.json and prints a markdown table.
"""
from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
import urllib.error
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from homeai.benchio import bench_path  # noqa: E402
from homeai.httpjson import OLLAMA, post_json  # noqa: E402

NONE = frozenset({None})
CALC_OK = frozenset({None, "calculator"})   # simple arithmetic: answering directly is fine too

# (question, acceptable tools). None = answer in words.
QUESTIONS: list[tuple[str, frozenset]] = [
    ("Tell me about yourself.", NONE),
    ("What can you do?", NONE),
    ("How are you today?", NONE),
    ("Tell me a joke.", NONE),
    ("What's the capital of France?", NONE),
    ("Who wrote Pride and Prejudice?", NONE),
    ("How many legs does a spider have?", NONE),
    ("Why is the sky blue?", NONE),
    ("What's a good name for a puppy?", NONE),
    ("Thank you, Jarvis.", NONE),
    ("Good night, Jarvis.", NONE),
    ("What do you think about free will?", NONE),
    ("Explain photosynthesis simply.", NONE),
    ("What's the difference between a frog and a toad?", NONE),
    ("How far away is the moon?", NONE),
    ("What's your favorite color?", NONE),
    ("What is seventeen times twenty-three?", CALC_OK),
    ("What's fifteen percent of eighty?", CALC_OK),
    ("What's the weather like right now?", frozenset({"homeai__weather"})),
    ("Do I need a jacket today?", frozenset({"homeai__weather"})),
    ("How hot is it going to get today?", frozenset({"homeai__weather"})),
    ("Who was at the front door?", frozenset({"homeai__door"})),
    ("Did anyone ring the doorbell?", frozenset({"homeai__door"})),
    ("Set a timer for ten minutes.", frozenset({"homeai__timer"})),
    ("How much time is left on the timer?", frozenset({"homeai__timer"})),
    ("Cancel the pasta timer.", frozenset({"homeai__timer"})),
    ("What's the latest news about NASA's Artemis mission?", frozenset({"homeai__research"})),
    ("Who won the Falcons game last night?", frozenset({"homeai__research"})),
    ("What's the price of Bitcoin right now?", frozenset({"homeai__research"})),
]


def classify(called: str | None, wanted: frozenset) -> str:
    if called in wanted:
        return "ok"
    if called is None:
        return "miss"
    return "misfire" if None in wanted else "wrong"


def first_tool(message: dict) -> str | None:
    calls = message.get("tool_calls") or []
    return calls[0].get("function", {}).get("name") if calls else None


def request_for(capture: dict, model: str, question: str, think: bool | None) -> dict:
    """Jarvis's captured request with the model and question swapped; Ollama's native /api/chat shape."""
    body = capture["body"]
    system = [m for m in body["messages"] if m["role"] == "system"]
    payload = {"model": model, "stream": False, "keep_alive": "10m",
               "messages": system + [{"role": "user", "content": question}],
               "tools": body.get("tools", []),
               "options": {"temperature": body.get("temperature", 0.45), "num_ctx": 16384}}
    if think is not None:
        payload["think"] = think
    return payload


def openai_request_for(capture: dict, model: str, question: str, extra: dict | None = None) -> dict:
    """The captured body exactly as ZeroClaw sends it (OpenAI-compatible endpoint), model and question swapped.

    ``extra`` is merged at the top level, as ZeroClaw's ``provider_extra`` / ``think`` would be.
    """
    body = dict(capture["body"])
    body["messages"] = [m for m in body["messages"] if m["role"] == "system"] + [{"role": "user", "content": question}]
    return {**body, "model": model, "stream": False, **(extra or {})}


def ask_openai(capture: dict, model: str, question: str, post=post_json, extra: dict | None = None) -> dict:
    """Jarvis's real route: no think flag, whatever the model does by default is what Jarvis would get.

    Records how much the model thought: on 2026-11-10 both candidates turned out to think on this route.
    """
    start = time.monotonic()
    reply = post(f"{OLLAMA}/v1/chat/completions", openai_request_for(capture, model, question, extra), 300)
    message = (reply.get("choices") or [{}])[0].get("message", {})
    return {"tool": first_tool(message), "text": message.get("content") or "",
            "reasoning_chars": len(message.get("reasoning") or message.get("reasoning_content") or ""),
            "seconds": round(time.monotonic() - start, 2)}


def ask(capture: dict, model: str, question: str, state: dict, post=post_json) -> dict:
    """One answer. Thinking is turned off where the model has it; models without it reject the flag, so drop it."""
    if state.get("endpoint") == "openai":
        return ask_openai(capture, model, question, post, state.get("extra"))
    think = state.get(model, False)
    start = time.monotonic()
    try:
        reply = post(f"{OLLAMA}/api/chat", request_for(capture, model, question, think), 300)
    except urllib.error.HTTPError as exc:
        if think is None or exc.code != 400:
            raise
        state[model] = None
        return ask(capture, model, question, state, post)
    message = reply.get("message", {})
    return {"tool": first_tool(message), "text": message.get("content", ""),
            "seconds": round(time.monotonic() - start, 2)}


def on_gpu(model: str, post=post_json) -> float:
    for m in post(f"{OLLAMA}/api/ps").get("models", []):
        if m.get("name") == model or m.get("name") == f"{model}:latest":
            return round(100 * m.get("size_vram", 0) / max(m.get("size", 1), 1), 1)
    return 0.0


def summary(rows: list[dict]) -> dict:
    count = {k: sum(1 for r in rows if r["verdict"] == k) for k in ("ok", "misfire", "miss", "wrong")}
    spoken = [len(r["text"].split()) for r in rows if r["tool"] is None and r["text"].strip()]
    warm = sorted(r["seconds"] for r in rows[1:]) or [0.0]
    thought = [r.get("reasoning_chars", 0) for r in rows]
    return {**count, "n": len(rows), "median_words": statistics.median(spoken) if spoken else 0,
            "median_s": warm[len(warm) // 2], "median_reasoning": statistics.median(thought) if thought else 0}


def run(capture: dict, models: list[str], reps: int, out: Path, post=post_json, log=print,
        endpoint: str = "native", extra: dict | None = None) -> dict:
    data: dict = {"questions": [[q, sorted(str(t) for t in w)] for q, w in QUESTIONS], "models": {},
                  "endpoint": endpoint, "extra": extra or {}}
    state: dict = {"endpoint": endpoint, "extra": extra}
    for model in models:
        rows = []
        for rep in range(reps):
            for question, wanted in QUESTIONS:
                answer = ask(capture, model, question, state, post)
                row = {"question": question, "rep": rep, **answer, "verdict": classify(answer["tool"], wanted)}
                rows.append(row)
                log(f"{model} | {row['verdict']:7} | {row['tool'] or '-':17} | {question} | {row['text'][:70]!r}")
        data["models"][model] = {"gpu_percent": on_gpu(model, post), "summary": summary(rows), "rows": rows}
        (out / "result.json").write_text(json.dumps(data, indent=1))
    return data


def table(data: dict) -> str:
    lines = ["| model | ok | misfire | miss | wrong | median words | median s | on GPU % | median reasoning chars |",
             "|---" * 9 + "|"]
    for model, d in data["models"].items():
        s = d["summary"]
        lines.append(f"| {model} | {s['ok']}/{s['n']} | {s['misfire']} | {s['miss']} | {s['wrong']} | "
                     f"{s['median_words']} | {s['median_s']} | {d['gpu_percent']} | {s.get('median_reasoning', 0)} |")
    return "\n".join(lines)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--prompt", required=True, help="a capture from probe_agent_prompt.py --save")
    ap.add_argument("--model", action="append", required=True)
    ap.add_argument("--reps", type=int, default=3)
    ap.add_argument("--no-pause", action="store_true", help="leave Jarvis running (timings will be skewed)")
    ap.add_argument("--endpoint", choices=("native", "openai"), default="openai",
                    help="openai = /v1/chat/completions, the route ZeroClaw really uses (default); "
                         "native = /api/chat with thinking turned off")
    ap.add_argument("--extra", type=json.loads, default=None,
                    help='JSON merged into each openai request, e.g. \'{"reasoning_effort": "none"}\'')
    args = ap.parse_args()
    capture = json.loads(Path(args.prompt).expanduser().read_text())[0]
    out = bench_path("misfire")
    out.mkdir(parents=True)
    if args.no_pause:
        data = run(capture, args.model, args.reps, out, endpoint=args.endpoint, extra=args.extra)
    else:
        from homeai import gpu_mode as G
        with G.ModeSwitcher(G.System(), log=lambda line: print("[mode]", line, flush=True)).jarvis_paused():
            data = run(capture, args.model, args.reps, out, endpoint=args.endpoint, extra=args.extra)
    print(table(data))
    print(out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
