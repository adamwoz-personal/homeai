#!/usr/bin/env python3
# Copyright (c) 2026 Adam Wosotowsky
# SPDX-License-Identifier: BUSL-1.1
# Licensed under the Business Source License 1.1; see LICENSE at the
# repository root. Converts to Apache-2.0 on 2030-09-15.
"""Blind A/B listening test of two voice models in Jarvis's voice (LLM_UPGRADE_PLAN step 2).

Why: the misfire eval and the VRAM probe (steps 1, 1b) can't tell whether an
answer is worth hearing. Adam judges that by ear. bench_conversation.py can't
be used: it drives the real ZeroClaw agent, where tool calls run (the
"say hello" probe played jazz on 2026-11-04), and it only uses the live model.

``generate``: each model answers the same conversations with Jarvis's
captured system prompt and tools, sent to Ollama's OpenAI endpoint the way
ZeroClaw sends them (multi-turn, the model's own earlier answers in the
history). Talks only to Ollama: a tool call stays a JSON field and is heard as
"(would look this up)". Jarvis is paused meanwhile and comes back with its
own model on the GPU. Each answer goes through the daemon's own speech chain
(markdown flattening, sanitising, dropping repeated sentences, the spoken
budget plus "There's more to it..."), so Adam hears what Jarvis would say.
Per conversation, the models are shuffled into A and B; the key is kept in
key.json, out of sight until Adam has picked.

``play``: speaks every conversation through Jarvis's Piper voice and speaker,
A's whole conversation then B's. Jarvis is stopped while it plays (an answer
saying "Jarvis" would wake it) and started again afterwards.

``reveal``: prints which model was A and B, with the timings.

    .venv/bin/python tools/listen_voice_models.py generate --prompt ~/homeai-bench/zc-prompt-<date>.json \\
        --model qwen35-voice --model gemma4-voice
    .venv/bin/python tools/listen_voice_models.py play ~/homeai-bench/listen-<time> [--only 2]
    .venv/bin/python tools/listen_voice_models.py reveal ~/homeai-bench/listen-<time>
"""
from __future__ import annotations

import argparse
import contextlib
import json
import random
import subprocess
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from homeai.benchio import bench_path  # noqa: E402
from homeai.dialogue import CONTINUE_PROMPT, drop_repeated_sentences, split_for_budget, unwrap_quotes  # noqa: E402
from homeai.httpjson import OLLAMA, post_json  # noqa: E402
from homeai.safety import sanitise_for_speech  # noqa: E402
from homeai.speech import flatten_markdown, normalise_for_speech  # noqa: E402

# (title, turns). Singles are what people ask a home assistant; the two conversations test follow-ups
# that lean on the previous answer (repetition across turns was llama31's fault).
CONVERSATIONS: list[tuple[str, list[str]]] = [
    ("About Jarvis", ["Tell me about yourself."]),
    ("A joke", ["Tell me a joke."]),
    ("The sky", ["Why is the sky blue?"]),
    ("Frogs and toads", ["What's the difference between a frog and a toad?"]),
    ("Free will", ["Do you think free will is real?", "Why do you think that?",
                   "But doesn't neuroscience show our decisions are made before we're aware of them?",
                   "So where does that leave moral responsibility?"]),
    ("Eggs", ["How long should I boil an egg for a soft yolk?", "And for hard boiled?"]),
]
LOOKUP = "(Jarvis would look this up.)"


def spoken_form(text: str, earlier: list[str], budget: int) -> tuple[str, int]:
    """The daemon's chain (daemon.py _answer and _deliver): what Jarvis would actually say, and words held back."""
    spoken = normalise_for_speech(sanitise_for_speech(flatten_markdown(unwrap_quotes(text))))
    spoken, _ = drop_repeated_sentences(spoken, earlier)
    head, rest = split_for_budget(spoken, budget)
    return (f"{head} {CONTINUE_PROMPT}" if rest else head), len(rest.split())


def converse(capture: dict, model: str, turns: list[str], budget: int, post=post_json) -> list[dict]:
    body = capture["body"]
    messages = [m for m in body["messages"] if m["role"] == "system"]
    earlier: list[str] = []
    out = []
    for question in turns:
        messages.append({"role": "user", "content": question})
        start = time.monotonic()
        reply = post(f"{OLLAMA}/v1/chat/completions", {**body, "model": model, "stream": False,
                                                       "messages": list(messages)}, 300)
        seconds = round(time.monotonic() - start, 2)
        message = (reply.get("choices") or [{}])[0].get("message", {})
        calls = message.get("tool_calls") or []
        text = message.get("content") or ""
        tool = calls[0].get("function", {}).get("name") if calls else None
        spoken, held = (LOOKUP, 0) if tool else spoken_form(text, earlier, budget)
        out.append({"question": question, "text": text, "spoken": spoken, "held_words": held, "tool": tool,
                    "words": len(text.split()), "seconds": seconds,
                    "reasoning_chars": len(message.get("reasoning") or message.get("reasoning_content") or "")})
        messages.append({"role": "assistant", "content": text})
        earlier.append(text)
    return out


def generate(capture: dict, models: list[str], out: Path, budget: int, post=post_json,
             rng: random.Random | None = None, log=print) -> dict:
    if len(models) != 2:
        raise ValueError("a blind A/B test takes exactly two models")
    rng = rng or random.Random()
    answers = {m: [converse(capture, m, turns, budget, post) for _, turns in CONVERSATIONS] for m in models}
    blind, key = [], []
    for i, (title, _) in enumerate(CONVERSATIONS):
        order = list(models)
        rng.shuffle(order)
        key.append(dict(zip("AB", order)))
        blind.append({"title": title, "A": answers[order[0]][i], "B": answers[order[1]][i]})
        log(f"{i + 1}. {title}: done")
    (out / "blind.json").write_text(json.dumps(blind, indent=1))
    (out / "key.json").write_text(json.dumps({"models": models, "key": key, "answers": answers}, indent=1))
    return {"blind": blind, "key": key}


def script_for(item: dict, number: int) -> list[str]:
    """The lines spoken for one conversation: A's turns, then B's."""
    lines = [f"Conversation {number}: {item['title']}."]
    for label in "AB":
        lines.append(f"Answer {label}.")
        for turn in item[label]:
            if len(item[label]) > 1:
                lines.append(f"You asked: {turn['question']}")
            lines.append(turn["spoken"])
    return lines


def play(folder: Path, only: int | None, say, pause=contextlib.nullcontext, gap=time.sleep) -> int:
    blind = json.loads((folder / "blind.json").read_text())
    picks = [(n, item) for n, item in enumerate(blind, 1) if only in (None, n)]
    if not picks:
        raise SystemExit(f"no conversation {only}; have 1-{len(blind)}")
    with pause():
        for number, item in picks:
            for line in script_for(item, number):
                say(line)
                gap(0.6)
    return len(picks)


def reveal(folder: Path) -> str:
    data = json.loads((folder / "key.json").read_text())
    lines = []
    for (title, _), k in zip(CONVERSATIONS, data["key"]):
        cells = []
        for label in "AB":
            model = k[label]
            i = [t for t, _ in CONVERSATIONS].index(title)
            turns = data["answers"][model][i]
            cells.append(f"{label} = {model} ({sum(t['words'] for t in turns)} words, "
                         f"{max(t['seconds'] for t in turns)} s max)")
        lines.append(f"{title}: " + "; ".join(cells))
    return "\n".join(lines)


def _jarvis_stopped(run=subprocess.run):
    """Jarvis off while the speaker talks (it would hear "Jarvis"), back on afterwards whatever happens.

    service_scope.systemctl only builds the argv; on 2026-11-10 the first version called it without running it,
    and Jarvis stayed on.
    """
    from homeai import service_scope

    @contextlib.contextmanager
    def ctx():
        run(service_scope.systemctl("stop", "homeai.service"), check=True)
        try:
            yield
        finally:
            run(service_scope.systemctl("start", "homeai.service"), check=True)
    return ctx


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    g = sub.add_parser("generate")
    g.add_argument("--prompt", required=True, help="a capture from probe_agent_prompt.py --save")
    g.add_argument("--model", action="append", required=True)
    p = sub.add_parser("play")
    p.add_argument("folder")
    p.add_argument("--only", type=int)
    r = sub.add_parser("reveal")
    r.add_argument("folder")
    args = ap.parse_args()

    from homeai.config import Config
    cfg = Config()
    if args.cmd == "generate":
        capture = json.loads(Path(args.prompt).expanduser().read_text())[0]
        out = bench_path("listen")
        out.mkdir(parents=True)
        from homeai import gpu_mode as G
        with G.ModeSwitcher(G.System(), log=lambda line: print("[mode]", line, flush=True)).jarvis_paused():
            generate(capture, args.model, out, cfg.tts.spoken_budget_words)
        print(out)
    elif args.cmd == "play":
        from homeai.tts import Speaker
        speaker = Speaker(cfg.tts, cfg.audio.output_device)

        def say(line: str) -> None:
            result = speaker.say(line)
            if not result.ok:
                raise SystemExit(f"speech failed: {result.error}")
        play(Path(args.folder).expanduser(), args.only, say, _jarvis_stopped())
    else:
        print(reveal(Path(args.folder).expanduser()))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
