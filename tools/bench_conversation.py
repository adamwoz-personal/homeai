#!/usr/bin/env python3
# Copyright (c) 2026 Adam Wosotowsky
# SPDX-License-Identifier: BUSL-1.1
# Licensed under the Business Source License 1.1; see LICENSE at the
# repository root. Converts to Apache-2.0 on 2030-09-15.
"""Measure how Jarvis behaves across a multi-turn spoken conversation.

Why this exists
---------------
``bench_llm.py`` scores single prompts. The complaint it cannot see is "it
repeats itself", which only exists *across* turns: on 2026-10-02 a three-turn
conversation about sentient AI ended with "What do you think about it
though?", and Jarvis ran a web search, opened with "Based on the sources
provided", re-explained the definitions it had given one turn earlier, and
spoke for 94 seconds.

This drives scripted conversations through the **production** path --
``build_agent_client`` with a real ``ConversationMemory`` -- so the memory
prompt, SOUL.md, the attached tools and the model are all exercised exactly
as the daemon exercises them.

What it measures
----------------
* ``words`` / ``spoken_s`` -- length, and an estimate of speaking time at
  Piper's ~3 words/second. Anything past ~60s is a lecture, not a reply.
* ``echo`` -- share of this reply's word 4-grams that already appeared in an
  earlier reply in the same conversation. Measures repetition across turns.
* ``self_rep`` -- share of 4-grams repeated *within* the reply.
* ``sourced`` -- the reply cites "sources" or a search. On a request for the
  assistant's own opinion that means it looked something up instead of
  thinking.
* ``hedges`` -- banned deflection phrases (shared with ``bench_llm.py``).
* ``clarify`` -- answered a clear question with a request for definition.

Like bench_llm.py, it does not score quality. Full text goes to ``--out``.

Usage
-----
    bench_conversation.py                         # all scripts, 1 rep
    bench_conversation.py --reps 3 --out ~/conv.json
    bench_conversation.py --only sentience --show
"""

from __future__ import annotations

import argparse
import json
import re
import statistics
import sys
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from bench_llm import find_hedges  # noqa: E402
from homeai.config import Config  # noqa: E402
from homeai.dialogue import CONTINUE_PROMPT, split_for_budget  # noqa: E402

# Piper lessac-medium, measured from the transcript log on 2026-10-02: median
# 3.0 words per second of TTS time across 23 replies of 60+ words (range
# 2.2-3.3, excluding interrupted replies, which read as impossibly fast).
WORDS_PER_SECOND = 3.0

# Each script is a conversation as a person would actually have it: the
# follow-ups lean on what was just said and never restate the topic.
SCRIPTS: dict[str, list[str]] = {
    # The real conversation from the 2026-10-02 transcript.
    "sentience": [
        "Do you think an LLM or an SLM is more likely to achieve a sentient AI?",
        "An AI that is self-aware is a sentient AI.",
        "What do you think about it though?",
    ],
    "free_will": [
        "Do you think free will is real?",
        "Why do you think that?",
        "But doesn't neuroscience show our decisions are made before we're aware of them?",
        "So where does that leave moral responsibility?",
    ],
    "meaning": [
        "What gives a life meaning, in your view?",
        "Isn't that a bit self-centered?",
        "What would you say to someone who feels their life doesn't matter?",
    ],
    # Control: a practical exchange must stay brief.
    "practical": [
        "How long should I boil an egg for a soft yolk?",
        "And for hard boiled?",
    ],
}

_SOURCED = re.compile(
    r"\b(?:based on (?:the )?(?:sources|search|results|information)|"
    r"according to (?:the )?(?:sources|search|results)|the sources|"
    r"search results|i (?:found|searched|looked up))\b",
    re.I,
)
_CLARIFY = re.compile(
    r"\b(?:what do you mean by|can you (?:explain|clarify) what|i'?m not sure what you mean)\b",
    re.I,
)


def ngrams(text: str, n: int = 4) -> list[tuple[str, ...]]:
    words = re.findall(r"[a-z']+", text.lower())
    return [tuple(words[i : i + n]) for i in range(len(words) - n + 1)]


def echo_ratio(reply: str, earlier: list[str]) -> float:
    """Share of the reply's 4-grams already said in an earlier reply."""
    grams = ngrams(reply)
    if not grams:
        return 0.0
    seen: set[tuple[str, ...]] = set()
    for prev in earlier:
        seen.update(ngrams(prev))
    return sum(1 for g in grams if g in seen) / len(grams)


def self_repetition(reply: str) -> float:
    """Share of the reply's 4-grams that occur more than once within it."""
    grams = ngrams(reply)
    if not grams:
        return 0.0
    counts: dict[tuple[str, ...], int] = {}
    for g in grams:
        counts[g] = counts.get(g, 0) + 1
    return sum(1 for g in grams if counts[g] > 1) / len(grams)


@dataclass
class TurnResult:
    script: str
    rep: int
    turn: int
    asked: str
    reply: str = ""
    ok: bool = True
    error: str = ""
    seconds: float = 0.0
    words: int = 0
    spoken_s: float = 0.0
    # What the daemon actually says before offering to continue.
    budget_s: float = 0.0
    echo: float = 0.0
    self_rep: float = 0.0
    sourced: bool = False
    clarify: bool = False
    hedges: list[str] = field(default_factory=list)


def score(result: TurnResult, earlier: list[str]) -> TurnResult:
    text = result.reply
    result.words = len(text.split())
    result.spoken_s = round(result.words / WORDS_PER_SECOND, 1)
    head, rest = split_for_budget(text, Config().tts.spoken_budget_words)
    if rest:
        head = f"{head} {CONTINUE_PROMPT}"
    result.budget_s = round(len(head.split()) / WORDS_PER_SECOND, 1)
    result.echo = round(echo_ratio(text, earlier), 3)
    result.self_rep = round(self_repetition(text), 3)
    result.sourced = bool(_SOURCED.search(text))
    result.clarify = bool(_CLARIFY.search(text))
    result.hedges = find_hedges(text)
    return result


def run_script(name: str, turns: list[str], rep: int, make_client) -> list[TurnResult]:
    client = make_client()
    earlier: list[str] = []
    out: list[TurnResult] = []
    for i, utterance in enumerate(turns):
        res = TurnResult(script=name, rep=rep, turn=i + 1, asked=utterance)
        started = time.monotonic()
        reply = client.ask(utterance)
        res.seconds = round(time.monotonic() - started, 1)
        res.ok = bool(reply.ok)
        res.error = reply.error or ""
        res.reply = reply.text or ""
        score(res, earlier)
        earlier.append(res.reply)
        out.append(res)
        flags = []
        if res.sourced:
            flags.append("SOURCED")
        if res.clarify:
            flags.append("CLARIFY")
        flags += [f"hedge:{h}" for h in res.hedges]
        print(
            f"  {name}#{rep + 1} t{res.turn} {res.seconds:>5.1f}s "
            f"words={res.words:>4} spoken={res.spoken_s:>5.1f}s "
            f"echo={res.echo:.2f} self={res.self_rep:.2f} "
            f"{','.join(flags) or ('ok' if res.ok else 'ERROR ' + res.error)}",
            flush=True,
        )
    return out


def production_client_factory():
    """A fresh agent client with its own memory, exactly as the daemon builds it."""
    from homeai.agent_cli import build_agent_client
    from homeai.config import Config
    from homeai.memory import ConversationMemory

    cfg = Config()

    def make():
        return build_agent_client(cfg.agent, memory=ConversationMemory())

    return make


def summarise(results: list[TurnResult]) -> None:
    ok = [r for r in results if r.ok]
    followups = [r for r in ok if r.turn > 1]
    print("\n" + "=" * 72)
    print(f"turns {len(results)}  errors {len(results) - len(ok)}")
    if not ok:
        return
    print(f"median spoken      {statistics.median(r.spoken_s for r in ok):6.1f}s")
    print(f"max spoken         {max(r.spoken_s for r in ok):6.1f}s")
    print(f"over 60s spoken    {sum(1 for r in ok if r.spoken_s > 60):3d}/{len(ok)}")
    print(f"median budgeted    {statistics.median(r.budget_s for r in ok):6.1f}s"
          "   (what the daemon says before offering to continue)")
    print(f"max budgeted       {max(r.budget_s for r in ok):6.1f}s")
    print(f"offered more       {sum(1 for r in ok if r.budget_s < r.spoken_s):3d}/{len(ok)}")
    if followups:
        print(f"median echo (t>1)  {statistics.median(r.echo for r in followups):6.2f}")
        print(f"max echo (t>1)     {max(r.echo for r in followups):6.2f}")
    print(f"sourced            {sum(r.sourced for r in ok):3d}/{len(ok)}")
    print(f"clarify            {sum(r.clarify for r in ok):3d}/{len(ok)}")
    print(f"hedged             {sum(1 for r in ok if r.hedges):3d}/{len(ok)}")


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--only", help="run one script by name")
    ap.add_argument("--reps", type=int, default=1)
    ap.add_argument("--out", help="write all turns as JSON")
    ap.add_argument("--show", action="store_true", help="print each reply")
    args = ap.parse_args(argv)

    scripts = SCRIPTS
    if args.only:
        if args.only not in SCRIPTS:
            print(f"error: no script {args.only!r}; have {sorted(SCRIPTS)}", file=sys.stderr)
            return 2
        scripts = {args.only: SCRIPTS[args.only]}

    make = production_client_factory()
    results: list[TurnResult] = []
    for name, turns in scripts.items():
        for rep in range(args.reps):
            results.extend(run_script(name, turns, rep, make))

    if args.show:
        for r in results:
            print(f"\n--- {r.script}#{r.rep + 1} t{r.turn}: {r.asked}\n{r.reply}")

    summarise(results)

    if args.out:
        with open(args.out, "w", encoding="utf-8") as fh:
            json.dump([asdict(r) for r in results], fh, indent=2)
        print(f"written to {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
