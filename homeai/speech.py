# Copyright (c) 2026 Adam Wosotowsky
# SPDX-License-Identifier: BUSL-1.1
# Licensed under the Business Source License 1.1; see LICENSE at the
# repository root. Converts to Apache-2.0 on 2030-09-15.
"""Normalise agent text into something a speech synthesiser reads correctly.

This is deliberately *code*, not prompt instructions. The voice agent was told
plainly in SOUL.md to speak American units and to avoid symbols, and it still
answered "28 degrees Celsius ... humidity is at 50% ... 6 kilometers per hour".
A local model relaying tool output tends to echo that output's formatting no
matter what the system prompt says.

Anything that must be reliable belongs here, where it is deterministic and
testable. The prompt handles *style*; this handles *correctness*.

Separate from ``safety.py`` on purpose: that module is the security boundary
and should not accumulate presentation logic.
"""

from __future__ import annotations

import re

# ---------------------------------------------------------------------------
# Unit conversion (metric -> US customary)
# ---------------------------------------------------------------------------

# Matches "28 degrees Celsius", "28°C", "28 C", "-5 degrees celsius".
_CELSIUS = re.compile(
    r"(-?\d+(?:\.\d+)?)\s*(?:°\s*C\b|degrees?\s+(?:c|celsius)\b|\bC\b(?=\s|$|[.,!?]))",
    re.IGNORECASE,
)
_KMH = re.compile(
    r"(\d+(?:\.\d+)?)\s*(?:km/?h\b|kph\b|kilometers?\s+per\s+hour\b)",
    re.IGNORECASE,
)
_KM = re.compile(r"(\d+(?:\.\d+)?)\s*(?:kilometers?\b|kms?\b)", re.IGNORECASE)
_CM = re.compile(r"(\d+(?:\.\d+)?)\s*(?:centimet(?:er|re)s?\b)", re.IGNORECASE)
_KG = re.compile(r"(\d+(?:\.\d+)?)\s*(?:kilograms?\b|kgs?\b)", re.IGNORECASE)
_MM = re.compile(r"(\d+(?:\.\d+)?)\s*(?:millimet(?:er|re)s?\b|mm\b)", re.IGNORECASE)


def _fmt(value: float) -> str:
    """Format a converted quantity for the ear, not for a display.

    Spoken decimals are noise: "eighty two point four degrees" is worse than
    "eighty two degrees", and no one needs tenths of a degree by voice. Values
    of ten or more are rounded to whole numbers. Below ten a single decimal is
    kept, because there the fraction carries real information ("six point two
    miles" is meaningfully different from "six miles").
    """
    if abs(value) >= 10:
        return str(int(round(value)))
    text = f"{value:.1f}"
    return text[:-2] if text.endswith(".0") else text


def convert_units(text: str) -> str:
    """Rewrite metric quantities as US customary ones.

    Conversions are applied to the *number as well as the unit*, so the
    sentence stays true. Temperature is handled first because "C" is the most
    ambiguous token and the others cannot produce a stray "C".
    """
    if not text:
        return ""

    def celsius(m: re.Match[str]) -> str:
        c = float(m.group(1))
        return f"{_fmt(c * 9 / 5 + 32)} degrees"

    def kmh(m: re.Match[str]) -> str:
        return f"{_fmt(float(m.group(1)) * 0.621371)} miles per hour"

    def km(m: re.Match[str]) -> str:
        return f"{_fmt(float(m.group(1)) * 0.621371)} miles"

    def cm(m: re.Match[str]) -> str:
        return f"{_fmt(float(m.group(1)) / 2.54)} inches"

    def kg(m: re.Match[str]) -> str:
        return f"{_fmt(float(m.group(1)) * 2.20462)} pounds"

    def mm(m: re.Match[str]) -> str:
        return f"{_fmt(float(m.group(1)) / 25.4)} inches"

    text = _CELSIUS.sub(celsius, text)
    text = _KMH.sub(kmh, text)
    text = _KM.sub(km, text)
    text = _CM.sub(cm, text)
    text = _MM.sub(mm, text)
    text = _KG.sub(kg, text)
    return text


# ---------------------------------------------------------------------------
# Symbol expansion
# ---------------------------------------------------------------------------

_SYMBOLS: tuple[tuple[re.Pattern[str], str], ...] = (
    (re.compile(r"(\d)\s*%"), r"\1 percent"),
    (re.compile(r"%"), " percent"),
    (re.compile(r"°\s*F\b", re.IGNORECASE), " degrees"),
    (re.compile(r"°"), " degrees"),
    (re.compile(r"(\d)\s*&\s*(\d)"), r"\1 and \2"),
    (re.compile(r"\s&\s"), " and "),
    # Piper reads a bare URL character by character; it is never useful aloud.
    (re.compile(r"https?://\S+"), " a link "),
    (re.compile(r"\*+"), " "),
    (re.compile(r"#{1,6}\s"), " "),
    (re.compile(r"_{2,}"), " "),
)


# ---------------------------------------------------------------------------
# Markdown structure
# ---------------------------------------------------------------------------
#
# These must run while the text still has line breaks. ``sanitise_for_speech``
# collapses all whitespace, so by the time unit conversion happens the fact
# that a line *started* with "- " is gone and the marker is indistinguishable
# from a hyphen mid-sentence. Hence flattening happens first in the chain.

_LIST_MARKER = re.compile(r"(?m)^[ \t]*(?:[-*\u2022\u2013]|\d{1,2}[.)])[ \t]+")
_HEADING = re.compile(r"(?m)^[ \t]*#{1,6}[ \t]+(.*)$")
_TABLE_ROW = re.compile(r"(?m)^[ \t]*\|(.+)\|[ \t]*$")
_TABLE_RULE = re.compile(r"(?m)^[ \t]*\|[\s|:-]+\|[ \t]*$")
_BLOCKQUOTE = re.compile(r"(?m)^[ \t]*>[ \t]?")


def _terminate(line: str) -> str:
    """Give a fragment sentence-final punctuation so the voice pauses.

    List items are usually written without a full stop. Spoken back-to-back
    with no terminator they run together into one breathless sentence, which
    is the single most obvious "robot reading a web page" tell.
    """
    stripped = line.strip().rstrip(",;:")
    if not stripped:
        return ""
    return stripped if stripped[-1] in ".!?" else stripped + "."


def flatten_markdown(text: str) -> str:
    """Turn line-structured markdown into flowing prose.

    Handles the constructs a model reaches for when asked something that
    sounds list-shaped ("what should I check before a road trip"), which the
    benchmark showed it does even when the system prompt forbids it.
    """
    if not text:
        return ""

    text = _TABLE_RULE.sub("", text)
    text = _TABLE_ROW.sub(lambda m: m.group(1).replace("|", ", "), text)
    text = _HEADING.sub(lambda m: _terminate(m.group(1)), text)
    text = _BLOCKQUOTE.sub("", text)

    out: list[str] = []
    for line in text.splitlines():
        if _LIST_MARKER.match(line):
            out.append(_terminate(_LIST_MARKER.sub("", line, count=1)))
        else:
            out.append(line)
    return "\n".join(out)


def expand_symbols(text: str) -> str:
    """Replace symbols that a synthesiser mispronounces or reads literally."""
    if not text:
        return ""
    for pattern, replacement in _SYMBOLS:
        text = pattern.sub(replacement, text)
    return text


def normalise_for_speech(text: str, us_units: bool = True) -> str:
    """Full pipeline: convert units, expand symbols, tidy whitespace."""
    if not text:
        return ""
    if us_units:
        text = convert_units(text)
    text = expand_symbols(text)
    return re.sub(r"\s+", " ", text).strip()


# ---------------------------------------------------------------------------
# Sentence chunking
# ---------------------------------------------------------------------------
#
# Piper is invoked once per chunk, and each invocation is a separate `aplay`.
# That buys two things a single long utterance cannot:
#
#   * audio starts after the FIRST sentence is synthesised rather than the
#     whole reply, and
#   * playback has seams. A 68-second answer was previously one uninterruptible
#     `aplay` call; in chunks it can be stopped between sentences.
#
# Chunks are merged up to a target length because the seams cost something
# too: each one is a process spawn and a short silence, and chopping every
# clause makes speech sound stilted.

_SENTENCE_SPLIT = re.compile(r"(?<=[.!?])\s+")

# Abbreviations that end in a period but do not end a sentence. Splitting on
# them produces an audible stumble mid-phrase.
_ABBREVIATIONS = (
    "mr.", "mrs.", "ms.", "dr.", "prof.", "sr.", "jr.", "st.",
    "e.g.", "i.e.", "etc.", "vs.", "approx.", "dept.", "fig.",
    "no.", "inc.", "ltd.", "co.", "u.s.", "u.k.", "a.m.", "p.m.",
)


def _ends_with_abbreviation(text: str) -> bool:
    lowered = text.lower().rstrip()
    return any(lowered.endswith(abbr) for abbr in _ABBREVIATIONS)


def split_sentences(text: str) -> list[str]:
    """Split into sentences, keeping abbreviations intact."""
    if not text or not text.strip():
        return []

    pieces = _SENTENCE_SPLIT.split(text.strip())
    merged: list[str] = []
    for piece in pieces:
        if merged and _ends_with_abbreviation(merged[-1]):
            merged[-1] = f"{merged[-1]} {piece}"
        else:
            merged.append(piece)
    return [p.strip() for p in merged if p.strip()]


def chunk_for_speech(
    text: str, target_chars: int = 220, max_chars: int = 400
) -> list[str]:
    """Group sentences into speakable chunks.

    ``target_chars`` is roughly ten seconds of speech: short enough that an
    interruption is answered promptly, long enough that the seams are not
    distracting.

    A single sentence longer than ``max_chars`` is split on clause boundaries
    rather than left whole, because one runaway sentence would otherwise
    restore exactly the uninterruptible block this exists to prevent.
    """
    chunks: list[str] = []
    current = ""

    for sentence in split_sentences(text):
        for part in _split_long_sentence(sentence, max_chars):
            if not current:
                current = part
            elif len(current) + len(part) + 1 <= target_chars:
                current = f"{current} {part}"
            else:
                chunks.append(current)
                current = part

    if current:
        chunks.append(current)
    return chunks


def _split_long_sentence(sentence: str, max_chars: int) -> list[str]:
    """Break an over-long sentence at clause boundaries, then at whitespace."""
    if len(sentence) <= max_chars:
        return [sentence]

    parts: list[str] = []
    current = ""
    # Semicolons and commas are natural breath points; splitting there is
    # far less noticeable than splitting mid-clause.
    for piece in re.split(r"(?<=[;,])\s+", sentence):
        if not current:
            current = piece
        elif len(current) + len(piece) + 1 <= max_chars:
            current = f"{current} {piece}"
        else:
            parts.append(current)
            current = piece
    if current:
        parts.append(current)

    # Last resort: a single clause still too long gets cut on word boundaries.
    final: list[str] = []
    for part in parts:
        while len(part) > max_chars:
            cut = part.rfind(" ", 0, max_chars)
            if cut <= 0:
                cut = max_chars
            final.append(part[:cut].strip())
            part = part[cut:].strip()
        if part:
            final.append(part)
    return final
