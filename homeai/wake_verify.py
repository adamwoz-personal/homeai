# Copyright (c) 2026 Adam Wosotowsky
# SPDX-License-Identifier: BUSL-1.1
# Licensed under the Business Source License 1.1; see LICENSE at the
# repository root. Converts to Apache-2.0 on 2030-09-15.
"""Second-stage wake check: did Whisper actually hear "Jarvis"?

openWakeWord alone woke Jarvis about a dozen times in one day (2026-10-02/03)
on household speech and TV, often scoring 0.85-0.97: "Love you, Valor",
"David.", "Cody will be back in a moment...", "Thank you. You're welcome.".
No threshold separates those from real wakes, which score 0.92-0.99.

So the daemon transcribes the ~2 s before the trigger together with the
utterance. A real wake has the wake word in that transcript, and anything
else is ignored silently. Whisper sometimes misspells the name ("Jervis",
"Javis"), so tokens are matched by similarity. WAKE_SIMILARITY = 0.75 accepts
those and rejects real names that are close to it: Travis 0.67, Marvin 0.67,
David 0.55. A word starting with the name's first four letters ("Jarvan")
also counts. See tools/wake_word_similarity.py.
"""

from __future__ import annotations

import re
from difflib import SequenceMatcher

WAKE_SIMILARITY = 0.75

# Whisper's stock hallucinations on noise and room audio. Never a request.
_HALLUCINATIONS = (
    "thanks for watching", "thank you for watching", "subscribe", "please subscribe",
    "see you next time", "you",
)
# How a conversation ends, not how one starts. Answered when a conversation is
# in progress, dropped when one would open with it: asked to answer "Thank
# you. You're welcome." out of nowhere, the model narrated its own reasoning
# for 27 s (2026-10-03 20:26).
_CLOSINGS = (
    "thank you", "thanks", "thank you very much", "thank you so much", "thanks a lot",
    "you're welcome", "you are welcome", "bye", "goodbye", "bye bye",
    "okay", "ok", "great", "cool", "nice", "amazing",
)


def _only(phrases) -> re.Pattern[str]:
    alternatives = "|".join(re.escape(p) for p in sorted(phrases, key=len, reverse=True))
    return re.compile(r"^(?:(?:" + alternatives + r")\b[\s,.!]*)+$")


_HALLUCINATION_RE = _only(_HALLUCINATIONS)
_PLEASANTRY_RE = _only(_HALLUCINATIONS + _CLOSINGS)


def _normalise(text: str) -> str:
    return re.sub(r"\s+", " ", (text or "").lower().replace("\u2019", "'")).strip()


def _name_words(wake_model: str) -> list[str]:
    # "hey_jarvis" -> ["jarvis"]: the greeting is too common to verify on.
    words = [w for w in re.split(r"[^a-z]+", wake_model.lower()) if len(w) > 3]
    return words or [wake_model.lower()]


def _tokens(text: str) -> list[tuple[str, int]]:
    """(normalised word, end offset) for each word in ``text``."""
    return [(m.group(0).lower().removesuffix("'s"), m.end())
            for m in re.finditer(r"[A-Za-z']+", text or "")]


def _is_name(token: str, names: list[str]) -> bool:
    # The prefix rule catches endings Whisper garbles ("Hey Jarvan", heard
    # live and scored 0.67); the names it must reject start differently.
    return any(
        SequenceMatcher(None, name, token).ratio() >= WAKE_SIMILARITY
        or (len(name) >= 5 and token.startswith(name[:4]))
        for name in names
    )


def mentions_wake_word(text: str, wake_model: str = "hey_jarvis") -> bool:
    names = _name_words(wake_model)
    return any(_is_name(tok, names) for tok, _ in _tokens(text))


def strip_wake_phrase(text: str, wake_model: str = "hey_jarvis") -> str:
    """Drop everything up to and including the first wake-word token.

    The pre-roll often carries the tail of earlier talk ("...so anyway, hey
    Jarvis, what's the time"), which belongs to nobody's question.
    """
    names = _name_words(wake_model)
    for tok, end in _tokens(text):
        if _is_name(tok, names):
            return text[end:].lstrip(" ,.!?;:-'\"").strip()
    return text.strip()


def is_hallucination_only(text: str) -> bool:
    """Only Whisper's stock noise phrases: drop always."""
    normalised = _normalise(text)
    return bool(normalised) and bool(_HALLUCINATION_RE.match(normalised))


def is_pleasantry_only(text: str) -> bool:
    """Only closings and/or hallucinations: no request in it."""
    normalised = _normalise(text)
    return bool(normalised) and bool(_PLEASANTRY_RE.match(normalised))


def closing_reply(text: str) -> str:
    """What Jarvis says to a closing mid-conversation; "" means stay quiet.

    Not sent to the model: given "thank you" plus the conversation window,
    the 8B voice model took it as a cue to continue the topic and spoke for
    23 s (2026-10-03 20:45).
    """
    normalised = _normalise(text)
    if re.search(r"\bthank|\bthanks\b", normalised):
        return "You're welcome."
    if re.search(r"\bbye\b|\bgoodbye\b", normalised):
        return "Bye for now."
    return ""
