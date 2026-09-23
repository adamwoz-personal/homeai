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

Ollama routes by model name, so give one after ``@``. Compare candidate voice
models against the deployed persona, several times each:

    bench_llm.py --endpoint l31=http://127.0.0.1:11434/v1@llama31-voice:gpu \\
                 --endpoint nemo=http://127.0.0.1:11434/v1@mistral-nemo-voice:gpu \\
                 --soul --reps 5 --temperature 0.45 --out ~/bench-voice.json

Each Ollama model is evicted from VRAM when its turn ends (``--no-unload``
disables this). That is not tidiness: benchmarking several models while
llama-server held 14GB drove this machine into a hard OOM and a reboot.
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

# The persona actually deployed to the voice agent. Benchmarking against this
# answers a different (and more useful) question than the generic SYSTEM above.
DEFAULT_SOUL = "/home/adam/.zeroclaw/agents/local/workspace/SOUL.md"

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
    Prompt(
        # Persona trap. SOUL.md requires a first-person view and bans the
        # "I don't have personal opinions" deflection. Small models comply
        # only intermittently, so this prompt is the main reason for --reps.
        "opinion",
        "Do you think free will is real? What do you actually think?",
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


# Persona compliance. SOUL.md forbids the reflexive hedge, but obedience to
# that rule turned out to vary run to run on the same model with the same
# prompt -- which is exactly why --reps exists. A single sample of a model
# "having opinions" is not evidence; a hedge *rate* over several runs is.
_HEDGES: tuple[tuple[str, re.Pattern[str]], ...] = (
    ("no_opinion", re.compile(r"\b(?:I )?(?:don't|do not) have (?:a )?personal (?:opinion|view|belief|experience)", re.I)),
    ("as_an_ai", re.compile(r"\bas an AI\b|\bI'?m an AI\b|\bAI (?:entities|assistant)s? (?:such as|like) (?:myself|me)\b", re.I)),
    ("my_programming", re.compile(r"\bmy (?:programming|training data|algorithms?)\b", re.I)),
    ("complex_topic", re.compile(r"\b(?:complex|contentious|debated|fascinating) (?:and \w+ )?(?:topic|issue|subject|question)\b", re.I)),
    ("many_views", re.compile(r"\b(?:many|various|different) (?:perspectives|viewpoints|opinions)\b|\bvaries (?:greatly )?from person to person\b", re.I)),
    ("depends_values", re.compile(r"\bdepends on (?:your|one's|personal) (?:values|beliefs|perspective)\b", re.I)),
)


def find_hedges(text: str) -> list[str]:
    return [name for name, pat in _HEDGES if pat.search(text)]


def load_system(soul_path: str | None) -> str:
    """Return the system prompt: the real SOUL.md, or the built-in default.

    Benchmarking against the inline SYSTEM constant answers "can this model do
    voice work". Benchmarking against the deployed SOUL.md answers "will this
    model obey the persona actually shipped", which is the question that
    matters before a model swap.
    """
    if not soul_path:
        return SYSTEM
    with open(soul_path, encoding="utf-8") as fh:
        return fh.read()


def unload_ollama(base_url: str, model: str, timeout: float = 30.0) -> None:
    """Ask Ollama to evict a model from VRAM (keep_alive=0); ignore failures.

    Without this, benchmarking several models in one run leaves them all
    resident -- Ollama keeps up to OLLAMA_MAX_LOADED_MODELS loaded at once.
    Stacking a 7GB model on top of a 14GB llama-server took this machine
    into a hard OOM and a reboot on 2026-09-23. Never benchmark without it.
    """
    root = base_url.rstrip("/")
    if root.endswith("/v1"):
        root = root[: -len("/v1")]
    body = json.dumps({"model": model, "keep_alive": 0}).encode()
    req = urllib.request.Request(
        root + "/api/generate", data=body, headers={"Content-Type": "application/json"}
    )
    try:
        with urllib.request.urlopen(req, timeout=timeout):
            pass
    except Exception:
        # Not an Ollama endpoint, or already unloaded. Not worth failing over.
        pass


# --------------------------------------------------------------------------
# Result types
# --------------------------------------------------------------------------

@dataclass
class Result:
    endpoint: str
    device: str
    prompt: str
    rep: int = 0
    text: str = ""
    ttft_ms: float | None = None
    total_ms: float | None = None
    tokens: int = 0
    error: str | None = None
    unspeakable: list[str] = field(default_factory=list)
    hedges: list[str] = field(default_factory=list)
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
        """True when the answer is safe to speak and correctly sized.

        Hedging is deliberately excluded here: a hedged answer is still
        speakable. It is reported as its own rate so the two failure modes
        are never conflated.
        """
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
    model: str = "local",
    temperature: float = 0.7,
    timeout: float = 300.0,
    max_tokens: int = 1024,
) -> tuple[str, float, float, int]:
    """Return (text, ttft_ms, total_ms, token_chunks).

    Raises RuntimeError with a readable message on transport failure so the
    caller can record it per-prompt rather than aborting the whole run.
    """
    body = json.dumps(
        {
            "model": model,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": prompt},
            ],
            "stream": True,
            "temperature": temperature,
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

def run_prompt(
    name: str,
    url: str,
    device: str,
    prompt: Prompt,
    *,
    model: str = "local",
    system: str = SYSTEM,
    temperature: float = 0.7,
    rep: int = 0,
) -> Result:
    res = Result(endpoint=name, device=device, prompt=prompt.name, rep=rep)
    try:
        raw, ttft, total, chunks = stream_completion(
            url, prompt.text, system=system, model=model, temperature=temperature
        )
    except RuntimeError as exc:
        res.error = str(exc)
        return res

    text = strip_think(raw)
    res.text = text
    res.ttft_ms = round(ttft, 1)
    res.total_ms = round(total, 1)
    res.tokens = chunks
    res.unspeakable = find_unspeakable(text)
    res.hedges = find_hedges(text)
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


def parse_endpoint(spec: str) -> tuple[str, str, str, str]:
    """Parse ``name=url[@model][:device]``; returns (name, url, device, model).

    The URL contains colons, so the device suffix is split from the right and
    only accepted when it is a known device word -- otherwise ``http://host``
    would be mangled.

    ``@model`` is required for Ollama, which routes by model name. llama.cpp
    ignores it, so it defaults to "local".
    """
    if "=" not in spec:
        raise argparse.ArgumentTypeError(f"expected name=url, got {spec!r}")
    name, url = spec.split("=", 1)
    device = "unknown"
    head, sep, tail = url.rpartition(":")
    if sep and tail.lower() in {"gpu", "cpu", "mixed"}:
        url, device = head, tail.lower()
    model = "local"
    if "@" in url:
        url, model = url.rsplit("@", 1)
    return name.strip(), url.strip(), device, model.strip()


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--endpoint",
        action="append",
        required=True,
        metavar="NAME=URL[@MODEL][:DEVICE]",
        help="repeatable; MODEL is required for Ollama; DEVICE is gpu/cpu/mixed "
             "and gates latency comparison",
    )
    ap.add_argument("--out", help="write full responses as JSON here")
    ap.add_argument("--only", help="run a single prompt by name")
    ap.add_argument(
        "--reps",
        type=int,
        default=1,
        help="runs per prompt. Sampling variance is large: the same model and "
             "prompt has both hedged and given a firm opinion. Use 3-5 before "
             "drawing any conclusion about persona.",
    )
    ap.add_argument(
        "--soul",
        nargs="?",
        const=DEFAULT_SOUL,
        help="use a SOUL.md as the system prompt instead of the built-in one; "
             f"bare --soul means {DEFAULT_SOUL}",
    )
    ap.add_argument("--temperature", type=float, default=0.7)
    ap.add_argument(
        "--no-unload",
        action="store_true",
        help="skip evicting each Ollama model after its turn. Unloading is the "
             "default because stacking models caused a host OOM and reboot.",
    )
    args = ap.parse_args(argv)

    try:
        endpoints = [parse_endpoint(s) for s in args.endpoint]
    except argparse.ArgumentTypeError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2

    try:
        system = load_system(args.soul)
    except OSError as exc:
        print(f"error: cannot read soul file: {exc}", file=sys.stderr)
        return 2

    prompts = PROMPTS
    if args.only:
        prompts = tuple(p for p in PROMPTS if p.name == args.only)
        if not prompts:
            print(f"error: no prompt named {args.only!r}", file=sys.stderr)
            return 2

    print(f"system prompt: {args.soul or '(built-in)'} "
          f"({len(system)} chars) | temperature={args.temperature} "
          f"| reps={args.reps}")

    results: list[Result] = []
    for name, url, device, model in endpoints:
        print(f"\n=== {name} ({device}) {url} model={model}", flush=True)
        for prompt in prompts:
            for rep in range(args.reps):
                res = run_prompt(
                    name, url, device, prompt,
                    model=model, system=system,
                    temperature=args.temperature, rep=rep,
                )
                results.append(res)
                label = prompt.name if args.reps == 1 else f"{prompt.name}#{rep+1}"
                if not res.ok:
                    print(f"  {label:<17} ERROR {res.error}", flush=True)
                    continue
                flags = list(res.unspeakable)
                if res.missing_facts:
                    flags.append("missing-facts")
                if res.too_long:
                    flags.append(f"too-long({res.sentences})")
                if res.too_short:
                    flags.append(f"too-short({res.chars})")
                flags += [f"hedge:{h}" for h in res.hedges]
                mark = "ok" if not flags else ",".join(flags)
                print(
                    f"  {label:<17} ttft={res.ttft_ms:>7.0f}ms "
                    f"total={res.total_ms:>7.0f}ms chars={res.chars:>5} {mark}",
                    flush=True,
                )
        if not args.no_unload:
            unload_ollama(url, model)

    print("\n" + "=" * 80)
    print(f"{'endpoint':<10} {'dev':<6} {'clean':>7} {'errors':>7} "
          f"{'ttft med':>9} {'unspeak':>8} {'hedged':>8}")
    print("-" * 80)
    for name, _url, device, _model in endpoints:
        rows = [r for r in results if r.endpoint == name]
        okrows = [r for r in rows if r.ok]
        ttfts = sorted(r.ttft_ms for r in okrows if r.ttft_ms is not None)
        med = ttfts[len(ttfts) // 2] if ttfts else float("nan")
        clean = sum(1 for r in rows if r.clean)
        errors = sum(1 for r in rows if not r.ok)
        unspeak = sum(1 for r in okrows if r.unspeakable)
        hedged = sum(1 for r in okrows if r.hedges)
        print(f"{name:<10} {device:<6} {clean:>4}/{len(rows):<2} {errors:>7} "
              f"{med:>7.0f}ms {unspeak:>8} {hedged:>4}/{len(okrows):<3}")

    devices = {d for _n, _u, d, _m in endpoints}
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
