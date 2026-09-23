#!/usr/bin/env python3
# Copyright (c) 2026 Adam Wosotowsky
# SPDX-License-Identifier: BUSL-1.1
# Licensed under the Business Source License 1.1; see LICENSE at the
# repository root. Converts to Apache-2.0 on 2030-09-15.
"""Ask one ad-hoc question of a model using the deployed SOUL.md as the prompt.

This is the quick prober; ``bench_llm.py`` is the measurement tool. Use this to
eyeball a single response or to check that a SOUL.md edit took effect. Do not
draw conclusions from it: responses vary run to run, and a single sample of a
model "having an opinion" misled this project once already. Use
``bench_llm.py --soul --reps 5`` for anything you intend to act on.

It talks to the OpenAI-compatible endpoint directly, so the result is not
coloured by ZeroClaw's own system prompt or attached tools. Comparing this
against ``zeroclaw agent --agent local`` is how you tell a model problem from
a harness problem.

Usage
-----
    probe_soul.py llama31-voice
    probe_soul.py llama31-voice "Do you think free will is real?"
    probe_soul.py mistral-nemo-voice --endpoint http://127.0.0.1:11434/v1
"""
import argparse
import json
import sys
import time
import urllib.request

DEFAULT_SOUL = "/home/adam/.zeroclaw/agents/local/workspace/SOUL.md"
DEFAULT_ENDPOINT = "http://127.0.0.1:11434/v1"
DEFAULT_QUESTION = "What do you think about death? Do you actually have a view?"


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("model", help="model name, e.g. llama31-voice")
    ap.add_argument("question", nargs="?", default=DEFAULT_QUESTION)
    ap.add_argument("--endpoint", default=DEFAULT_ENDPOINT)
    ap.add_argument("--soul", default=DEFAULT_SOUL)
    ap.add_argument("--temperature", type=float, default=0.45)
    ap.add_argument(
        "--no-soul",
        action="store_true",
        help="send no system prompt, to isolate persona effects from the model",
    )
    args = ap.parse_args(argv)

    messages = []
    if not args.no_soul:
        try:
            with open(args.soul, encoding="utf-8") as fh:
                messages.append({"role": "system", "content": fh.read()})
        except OSError as exc:
            print(f"error: cannot read {args.soul}: {exc}", file=sys.stderr)
            return 2
    messages.append({"role": "user", "content": args.question})

    body = json.dumps(
        {
            "model": args.model,
            "messages": messages,
            "temperature": args.temperature,
        }
    ).encode()
    req = urllib.request.Request(
        args.endpoint.rstrip("/") + "/chat/completions",
        body,
        {"Content-Type": "application/json"},
    )

    started = time.monotonic()
    try:
        with urllib.request.urlopen(req, timeout=300) as resp:
            payload = json.load(resp)
    except Exception as exc:
        print(f"error: request failed: {exc}", file=sys.stderr)
        return 1

    text = payload["choices"][0]["message"].get("content") or ""
    print(
        "=== %s | %.1fs | %d words | soul=%s"
        % (
            args.model,
            time.monotonic() - started,
            len(text.split()),
            "no" if args.no_soul else "yes",
        )
    )
    print(text)
    return 0


if __name__ == "__main__":
    sys.exit(main())
