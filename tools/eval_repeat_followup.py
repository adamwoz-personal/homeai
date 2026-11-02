#!/usr/bin/env python3
# Copyright (c) 2026 Adam Wosotowsky
# SPDX-License-Identifier: BUSL-1.1
# Licensed under the Business Source License 1.1; see LICENSE at the
# repository root. Converts to Apache-2.0 on 2030-09-15.
"""Replay a follow-up conversation and measure how much the model repeats itself.

Why this exists
---------------
Live on 2026-11-02, "tell me about the moon and what created it" was followed
by "is there a secondary hypothesis?" and "...I was just wondering if there are
secondary theories". Both follow-up replies re-explained the giant-impact
theory, despite three prompt rules against restating. ``drop_repeated_sentences``
now removes such sentences; this checks how often the model repeats and what
the listener actually hears after the filter.

Runs the production client (real SOUL, memory window, tools) against the live
agent, so it costs local model time only.

    .venv/bin/python tools/eval_repeat_followup.py [--rep 3]
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from homeai.agent_cli import build_agent_client  # noqa: E402
from homeai.config import load  # noqa: E402
from homeai.dialogue import drop_repeated_sentences, unwrap_quotes  # noqa: E402
from homeai.memory import ConversationMemory  # noqa: E402
from homeai.safety import sanitise_for_speech  # noqa: E402
from homeai.speech import flatten_markdown, normalise_for_speech, split_sentences  # noqa: E402

CONVERSATION = [
    "tell me about the moon and what created it.",
    "is there a secondary hypothesis?",
    "yeah, I mean I totally believe that's how the moon was created. "
    "I was just wondering if there are secondary theories out there.",
]
OUT_DIR = Path.home() / "homeai-bench"


def run_once(cfg) -> list[dict]:
    """One full conversation; per-turn raw reply, spoken text and drops."""
    memory = ConversationMemory()
    client = build_agent_client(cfg.agent, memory=memory)
    turns = []
    for utterance in CONVERSATION:
        earlier = [e.assistant for e in memory.recent()]
        reply = client.ask(utterance)
        raw = reply.text if reply.ok else f"<error: {reply.error}>"
        spoken = normalise_for_speech(sanitise_for_speech(flatten_markdown(unwrap_quotes(raw))))
        heard, dropped = drop_repeated_sentences(spoken, earlier)
        turns.append({"user": utterance, "raw": raw, "heard": heard, "dropped": dropped,
                      "sentences": len(split_sentences(spoken))})
    return turns


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--rep", type=int, default=3)
    args = ap.parse_args()

    cfg = load()
    runs = []
    for i in range(args.rep):
        turns = run_once(cfg)
        runs.append(turns)
        print(f"\n=== run {i + 1} ===")
        for t in turns:
            print(f"USER : {t['user']}\nHEARD: {t['heard']}\n"
                  f"       ({t['dropped']}/{t['sentences']} sentences dropped)")
    follow = [t for turns in runs for t in turns[1:]]
    repeating = sum(1 for t in follow if t["dropped"])
    print(f"\nfollow-up replies with a repeated sentence: {repeating}/{len(follow)}")
    OUT_DIR.mkdir(exist_ok=True)
    out = OUT_DIR / f"eval_repeat_followup-{time.strftime('%Y%m%d-%H%M%S')}.json"
    out.write_text(json.dumps(runs, indent=2))
    print(f"-> {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
