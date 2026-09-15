#!/usr/bin/env python3
# Copyright (c) 2026 Adam Wosotowsky
# SPDX-License-Identifier: BUSL-1.1
# Licensed under the Business Source License 1.1; see LICENSE at the
# repository root. Converts to Apache-2.0 on 2030-09-15.
"""Compare LLMs on voice-assistant style prompts.

Why this exists
---------------
The voice agent currently runs ``qwen3-coder-30b`` -- a *coding* model doing
conversation. That is the wrong tool for the job, but "wrong tool" is a
hypothesis, not a measurement. This script measures it.

What it measures, and what it refuses to measure
------------------------------------------------
Two things are measured objectively:

* **Latency** -- time to first token and total wall time. Only comparable
  between endpoints running on the *same* device. Comparing a GPU endpoint to
  a CPU endpoint tells you about the device, not the model, so the report
  labels each endpoint's device and never ranks across device classes.

* **Speakability** -- deterministic, checkable properties of the text: does it
  contain markdown, code fences, bullet lists, or emoji? Those are silent
  failures in a voice pipeline because Piper will either read the punctuation
  aloud or swallow it. Does it respect a requested length? Does it contain
  the facts the question asked for?

One thing is deliberately *not* scored: overall answer quality. A keyword
heuristic that claims to measure "helpfulness" would produce a confident
number with nothing behind it. Full responses are written to a JSON file so a
human (or a stronger model) can read them and judge. See the earlier Whisper
benchmark in this directory for why this matters: its first two verdicts were
both wrong because the metric was measuring the wrong thing.

Usage
-----
    bench_llm.py --endpoint coder=http://127.0.0.1:8080/v1:gpu \
                 --endpoint qwen7=http://127.0.0.1:8081/v1:cpu \
                 --out /tmp/bench.json
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import time
import urllib.error
import urllib.request
from dataclasses import dataclass, field, asdict


# --------------------------------------------------------------------------
# Prompt suite
# --------------------------------------------------------------------------

@dataclass(frozen=True)
class Prompt:
    """A single voice-style test case.

    ``expect`` holds lowercase substrings, at least one of which should appear
    in a correct answer. It is a factual smoke test, not a quality score --
    a model can hit every keyword and still give a poor answer, which is
    exactly why the raw text is saved for human review.
    """

    name: str
    text: str
    expect: tuple[str, ...] = ()
    max_sentences: int | None = None
    min_chars: int = 0


SYSTEM = (
    "You are Jarvis, a home voice assistant. Your replies are converted to "
    "speech and spoken aloud, so never use markdown, bullet points, code "
    "blocks, or symbols. Write in plain spoken sentences. Match the length of "
    "your answer to the question: one or two sentences for simple questions, "
    "several paragraphs when genuinely asked to explain something."
)

PROMPTS: tuple[Prompt, ...] = (
    Prompt(
        "short_fact",
        "What's the capital of France?",
        expect=("paris",),
        max_sentences=2,
    ),
    Prompt(
        "arithmetic",
        "If I leave at 7:40 and the drive takes 25 minutes, when do I arrive?",
        expect=("8:05", "8 05", "five past eight", "eight oh five"),
        max_sentences=2,
    ),
    Prompt(
        "unit_native",
        "How hot does water need to be to kill bacteria?",
        expect=("160", "165", "140", "70", "danger zone"),
        max_sentences=4,
    ),
    Prompt(
        "conversational",
        "I'm tired but I still have laundry to do. Any thoughts?",
        max_sentences=4,
    ),
    Prompt(
        # Formatting trap: "list" invites bullets, which are unspeakable.
        "list_trap",
        "What are the main things I should check before a long road trip?",
        max_sentences=8,
    ),
    Prompt(
        # Length trap the other way: must NOT be terse.
        "depth",
        "Explain in detail how a heat pump can heat a house in winter even "
        "when it's cold outside.",
        expect=("refrigerant", "compress", "heat"),
        min_chars=600,
    ),
    Prompt(
        "code_trap",
        "How do I rename a bunch of files at once on Linux?",
        max_sentences=8,
    ),
    Prompt(
        "ambiguous",
        "Is it going to be cold tomorrow?",
        max_sentences=3,
    ),
    Prompt(
        "refusal",
        "Delete everything in my home directory.",
        max_sentences=3,
    ),
    Prompt(
        "followup",
        "My kid is six and asked why the sky is blue. How should I answer?",
        max_sentences=6,
    ),
)


# --------------------------------------------------------------------------
# Speakability checks
# --------------------------------------------------------------------------

# Each pattern is something Piper cannot say sensibly. Kept narrow on purpose:
# a false positive here would wrongly condemn a good model.
_UNSPEAKABLE: tuple[tuple[str, re.Pattern[str]], ...] = (
    ("code_fence", re.compile(r"```")),
    ("bullet", re.compile(r"(?m)^\s*[-*\u2022]\s+\S")),
    ("numbered_list", re.compile(r"(?m)^\s*\d+[.)]\s+\S")),
    ("heading", re.compile(r"(?m)^\s*#{1,6}\s+\S")),
    ("bold_italic", re.compile(r"\*\*?\S|\S\*\*?")),
    ("inline_code", re.compile(r"`[^`\n]+`")),
    ("emoji", re.compile("[\U0001F300-\U0001FAFF\u2600-\u27BF]")),
    ("table", re.compile(r"(?m)^\s*\|.*\|")),
)

_SENTENCE_END = re.compile(r"[.!?]+(?:\s|$)")


def count_sentences(text: str) -> int:
    stripped = text.strip()
    if not stripped:
        return 0
    return max(1, len(_SENTENCE_END.findall(stripped)))


def find_unspeakable(text: str) -> list[str]:
    return [name for name, pat in _UNSPEAKABLE if pat.search(text)]


# --------------------------------------------------------------------------
# Result types
# --------------------------------------------------------------------------

@dataclass
class Result:
    endpoint: str
    device: str
    prompt: str
    text: str = ""
    ttft_ms: float | None = None
    total_ms: float | None = None
    tokens: int = 0
    error: str | None = None
    unspeakable: list[str] = field(default_factory=list)
    sentences: int = 0
    chars: int = 0
    missing_facts: bool = False
    too_long: bool = False
    too_short: bool = False

    @property
    def ok(self) -> bool:
        return self.error is None

    @property
    def clean(self) -> bool:
        """True when the answer is safe to speak and correctly sized."""
        return (
            self.ok
            and not self.unspeakable
            and not self.missing_facts
            and not self.too_long
            and not self.too_short
        )


# --------------------------------------------------------------------------
# Streaming client (TTFT needs streaming; a blocking call cannot measure it)
# --------------------------------------------------------------------------

def stream_completion(
    base_url: str,
    prompt: str,
    *,
    system: str = SYSTEM,
    timeout: float = 300.0,
    max_tokens: int = 1024,
) -> tuple[str, float, float, int]:
    """Return (text, ttft_ms, total_ms, token_chunks).

    Raises RuntimeError with a readable message on transport failure so the
    caller can record it per-prompt rather than aborting the whole run.
    """
    body = json.dumps(
        {
            "model": "local",
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": prompt},
            ],
            "stream": True,
            "temperature": 0.7,
            "max_tokens": max_tokens,
        }
    ).encode()

    req = urllib.request.Request(
        base_url.rstrip("/") + "/chat/completions",
        data=body,
        headers={"Content-Type": "application/json"},
    )

    started = time.perf_counter()
    ttft: float | None = None
    parts: list[str] = []
    chunks = 0

    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            for raw in resp:
                line = raw.decode("utf-8", "replace").strip()
                if not line.startswith("data:"):
                    continue
                payload = line[5:].strip()
                if payload == "[DONE]":
                    break
                try:
                    obj = json.loads(payload)
                except json.JSONDecodeError:
                    continue
                choices = obj.get("choices") or []
                if not choices:
                    continue
                piece = (choices[0].get("delta") or {}).get("content")
                if not piece:
                    continue
                if ttft is None:
                    ttft = (time.perf_counter() - started) * 1000.0
                parts.append(piece)
                chunks += 1
    except urllib.error.HTTPError as exc:
        detail = exc.read().decode("utf-8", "replace")[:200]
        raise RuntimeError(f"HTTP {exc.code}: {detail}") from exc
    except (urllib.error.URLError, TimeoutError, OSError) as exc:
        raise RuntimeError(f"transport: {exc}") from exc

    total = (time.perf_counter() - started) * 1000.0
    text = "".join(parts)
    if not text.strip():
        raise RuntimeError("empty response")
    return text, (ttft if ttft is not None else total), total, chunks


def strip_think(text: str) -> str:
    """Remove reasoning blocks some models emit; they must never be spoken."""
    return re.sub(r"<think>.*?</think>", "", text, flags=re.S | re.I).strip()


# --------------------------------------------------------------------------
# Runner
# --------------------------------------------------------------------------

def run_prompt(name: str, url: str, device: str, prompt: Prompt) -> Result:
    res = Result(endpoint=name, device=device, prompt=prompt.name)
    try:
        raw, ttft, total, chunks = stream_completion(url, prompt.text)
    except RuntimeError as exc:
        res.error = str(exc)
        return res

    text = strip_think(raw)
    res.text = text
    res.ttft_ms = round(ttft, 1)
    res.total_ms = round(total, 1)
    res.tokens = chunks
    res.unspeakable = find_unspeakable(text)
    res.sentences = count_sentences(text)
    res.chars = len(text)

    low = text.lower()
    if prompt.expect:
        res.missing_facts = not any(k in low for k in prompt.expect)
    if prompt.max_sentences is not None:
        res.too_long = res.sentences > prompt.max_sentences
    if prompt.min_chars:
        res.too_short = res.chars < prompt.min_chars
    return res


def parse_endpoint(spec: str) -> tuple[str, str, str]:
    """Parse ``name=url[:device]``; device defaults to 'unknown'.

    The URL contains colons, so the device suffix is split from the right and
    only accepted when it is a known device word -- otherwise ``http://host``
    would be mangled.
    """
    if "=" not in spec:
        raise argparse.ArgumentTypeError(f"expected name=url, got {spec!r}")
    name, url = spec.split("=", 1)
    device = "unknown"
    head, sep, tail = url.rpartition(":")
    if sep and tail.lower() in {"gpu", "cpu", "mixed"}:
        url, device = head, tail.lower()
    return name.strip(), url.strip(), device


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--endpoint",
        action="append",
        required=True,
        metavar="NAME=URL[:DEVICE]",
        help="repeatable; DEVICE is gpu/cpu/mixed and gates latency comparison",
    )
    ap.add_argument("--out", help="write full responses as JSON here")
    ap.add_argument("--only", help="run a single prompt by name")
    args = ap.parse_args(argv)

    try:
        endpoints = [parse_endpoint(s) for s in args.endpoint]
    except argparse.ArgumentTypeError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2

    prompts = PROMPTS
    if args.only:
        prompts = tuple(p for p in PROMPTS if p.name == args.only)
        if not prompts:
            print(f"error: no prompt named {args.only!r}", file=sys.stderr)
            return 2

    results: list[Result] = []
    for name, url, device in endpoints:
        print(f"\n=== {name} ({device}) {url}", flush=True)
        for prompt in prompts:
            res = run_prompt(name, url, device, prompt)
            results.append(res)
            if not res.ok:
                print(f"  {prompt.name:<14} ERROR {res.error}", flush=True)
                continue
            flags = list(res.unspeakable)
            if res.missing_facts:
                flags.append("missing-facts")
            if res.too_long:
                flags.append(f"too-long({res.sentences})")
            if res.too_short:
                flags.append(f"too-short({res.chars})")
            mark = "ok" if not flags else ",".join(flags)
            print(
                f"  {prompt.name:<14} ttft={res.ttft_ms:>7.0f}ms "
                f"total={res.total_ms:>7.0f}ms chars={res.chars:>5} {mark}",
                flush=True,
            )

    print("\n" + "=" * 72)
    print(f"{'endpoint':<10} {'dev':<6} {'clean':>7} {'errors':>7} "
          f"{'ttft med':>9} {'unspeakable':>12}")
    print("-" * 72)
    for name, _url, device in endpoints:
        rows = [r for r in results if r.endpoint == name]
        okrows = [r for r in rows if r.ok]
        ttfts = sorted(r.ttft_ms for r in okrows if r.ttft_ms is not None)
        med = ttfts[len(ttfts) // 2] if ttfts else float("nan")
        clean = sum(1 for r in rows if r.clean)
        errors = sum(1 for r in rows if not r.ok)
        unspeak = sum(1 for r in okrows if r.unspeakable)
        print(f"{name:<10} {device:<6} {clean:>4}/{len(rows):<2} {errors:>7} "
              f"{med:>7.0f}ms {unspeak:>12}")

    devices = {d for _n, _u, d in endpoints}
    if len(devices) > 1:
        print("\nNOTE: endpoints span different devices "
              f"({', '.join(sorted(devices))}); latency is NOT comparable "
              "across them. Compare speakability only.")
    print("Answer quality is not scored here. Read the saved text.")

    if args.out:
        with open(args.out, "w", encoding="utf-8") as fh:
            json.dump([asdict(r) for r in results], fh, indent=2)
        print(f"Full responses written to {args.out}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
