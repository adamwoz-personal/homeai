#!/usr/bin/env python3
# Copyright (c) 2026 Adam Wosotowsky
# SPDX-License-Identifier: BUSL-1.1
# Licensed under the Business Source License 1.1; see LICENSE at the
# repository root. Converts to Apache-2.0 on 2030-09-15.
"""Check how the model answers a mid-conversation "thank you".

Why this exists
---------------
Live on 2026-10-04, "thanks a lot" after a question about France got
"Expressing gratitude can strengthen relationships..." -- the 8B model took
the thanks as a topic. This runs the production client (real memory, SOUL,
tools) with one prior exchange, then each closing with ``CLOSING_HINT`` (or
a candidate hint from ``--hint``), and prints the raw and spoken replies.

    .venv/bin/python tools/eval_closing_hint.py [--rep 2] [--hint "..."]
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from homeai.agent_cli import build_agent_client  # noqa: E402
from homeai.config import load  # noqa: E402
from homeai.daemon import CLOSING_HINT  # noqa: E402
from homeai.dialogue import one_sentence  # noqa: E402
from homeai.memory import ConversationMemory  # noqa: E402
from homeai.safety import sanitise_for_speech  # noqa: E402
from homeai.speech import normalise_for_speech  # noqa: E402

PRIOR = ("what is the capital of France?", "The capital of France is Paris.")
CLOSINGS = ["thank you.", "thanks a lot.", "thanks Jarvis.", "bye.", "goodnight."]
# Phrases suggesting the model talked *about* thanks instead of accepting it.
OFF_TOPIC = ("gratitude", "thankful", "expressing", "relationship", "paris", "france")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--rep", type=int, default=1)
    ap.add_argument("--hint", default=CLOSING_HINT)
    args = ap.parse_args()

    cfg = load()
    bad = total = 0
    for closing in CLOSINGS:
        for _ in range(args.rep):
            memory = ConversationMemory()
            memory.add(*PRIOR)
            client = build_agent_client(cfg.agent, memory=memory)
            reply = client.ask(closing, hint=args.hint)
            spoken = one_sentence(normalise_for_speech(sanitise_for_speech(reply.text))) \
                if reply.ok else ""
            low = spoken.lower()
            off = not spoken or any(w in low for w in OFF_TOPIC)
            # "bye" -> "You're welcome." means the model ignored what was said.
            off = off or (not closing.startswith("thank") and "welcome" in low)
            flag = "OFF" if off else "ok"
            bad += flag != "ok"
            total += 1
            print(f"[{flag:3}] {closing!r:18} -> {spoken!r}   (raw {len(reply.text)} chars)")
    print(f"\n{total - bad}/{total} acceptable")
    return 0 if bad == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
