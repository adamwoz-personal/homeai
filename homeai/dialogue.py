# Copyright (c) 2026 Adam Wosotowsky
# SPDX-License-Identifier: BUSL-1.1
# Licensed under the Business Source License 1.1; see LICENSE at the
# repository root. Converts to Apache-2.0 on 2030-09-15.
"""Turn-taking decisions for a spoken conversation.

Pure functions with no audio or threads, so the rules can be unit tested.

Follow-up listening
-------------------
When Jarvis ends a reply with a question, the natural thing is to just answer
it. Requiring "Hey Jarvis" first makes a conversation feel like a series of
separate commands. So when a reply *invites* an answer, the daemon opens the
microphone for a short window without the wake word.

That window is also a way for background audio to reach the agent without
anyone addressing it -- a television answering Jarvis's question, which
produces another question, which opens another window. ``FollowupChain``
caps how many windows can open in a row before the wake word is required
again.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

from .speech import split_sentences


def invites_reply(text: str) -> bool:
    """True if the spoken reply ends by asking the listener something.

    Only the *final* sentence counts. A question in the middle of an answer is
    usually rhetorical ("So is free will real? I think...") and the speaker has
    already moved on; a question at the end is handing the turn over.
    """
    if not text:
        return False
    sentences = split_sentences(text.strip())
    if not sentences:
        return False
    return sentences[-1].rstrip().rstrip("\"')]").endswith("?")


_TRAILING_OFF = re.compile(r"(\.\.\.|\u2026)[\"')\]\s]*$")


def is_trailing_fragment(text: str) -> bool:
    """True if Whisper heard speech that trails off mid-thought.

    Whisper ends a transcript with "..." when the speaker had not finished.
    Observed live: someone telling a visitor about Jarvis said its name, and
    the rest ("I will run by the local AI, so there are...") went to the
    agent, which gave a 28 s second take on the previous topic. A question
    that trails off ("what's the weather in...?") still ends in "?", so it
    is answered.
    """
    stripped = (text or "").strip()
    return bool(_TRAILING_OFF.search(stripped))


@dataclass
class FollowupChain:
    """Counts consecutive wake-word-free turns and decides when to stop.

    ``max_chain`` is the number of follow-up windows that may open in a row.
    Any turn started with the wake word resets the count, because a person
    deliberately addressed Jarvis.
    """

    max_chain: int = 3
    count: int = 0

    def record_turn(self, from_followup: bool) -> None:
        self.count = self.count + 1 if from_followup else 0

    def may_open(self) -> bool:
        return self.max_chain > 0 and self.count < self.max_chain


# -- spoken length budget -----------------------------------------------------
#
# SOUL.md asks for short replies, and an 8B model largely ignores it: measured
# 2026-10-02 across 36 scripted turns, half ran past a minute of speech and
# one reached 468 words (two and a half minutes). A word count in a prompt is
# a suggestion; this is the guarantee.
#
# Nothing is thrown away. The rest of the reply is held, Jarvis asks whether
# to keep going, and the question opens the follow-up window, so "yes" or
# "go on" speaks the next part without another trip to the model.

CONTINUE_PROMPT = "There's more to it. Want me to keep going?"

# Below this, the remainder is spoken rather than offered. Asking "want me to
# keep going?" to save one sentence costs more time than it saves.
_MIN_REST_FRACTION = 0.25


def _word_count(text: str) -> int:
    return len(text.split())


def split_for_budget(text: str, max_words: int) -> tuple[str, str]:
    """Split ``text`` at a sentence boundary into (speak now, hold back).

    Returns ``(text, "")`` when the reply fits, when ``max_words`` is 0
    (disabled), or when the remainder would be too short to be worth asking
    about. The first sentence is always kept whole, however long: cutting a
    sentence in half is worse than overrunning the budget.
    """
    text = (text or "").strip()
    if max_words <= 0 or _word_count(text) <= max_words:
        return text, ""
    sentences = split_sentences(text)
    head: list[str] = []
    used = 0
    for sentence in sentences:
        n = _word_count(sentence)
        if head and used + n > max_words:
            break
        head.append(sentence)
        used += n
    rest = sentences[len(head):]
    if not rest or sum(_word_count(s) for s in rest) <= max_words * _MIN_REST_FRACTION:
        return text, ""
    return " ".join(head), " ".join(rest)


_REPLY_FILLERS = frozenset({
    "please", "okay", "ok", "um", "uh", "er", "oh", "well", "sure", "yeah",
    "yes", "yep", "yup", "jarvis", "hey", "and",
})

# Exact matches after normalisation, like bargein.is_dismissal: "go on" must
# not swallow "go on about something else".
_CONTINUES = frozenset({
    "", "go on", "go ahead", "keep going", "continue", "carry on",
    "tell me more", "more", "go for it", "do it", "absolutely", "definitely",
    "of course", "i do", "i would", "id like that", "please do",
    "lets hear it", "lets hear the rest", "the rest", "finish", "finish it",
})

_YES_WORDS = frozenset({"yes", "yeah", "yep", "yup", "sure", "okay", "ok"})
_NO_WORDS = frozenset({"no", "nope", "nah"})

_DECLINES = frozenset({
    "no", "nope", "nah", "no thanks", "no thank", "thats fine", "thats ok",
    "thats okay", "im good", "im fine", "thats enough", "thats all",
    "i got it", "got it", "no need", "not now", "maybe later", "no im good",
    "thanks", "thank you", "no thank you", "thats it", "stop there",
})


def _normalise_reply(text: str) -> tuple[list[str], str]:
    cleaned = (text or "").lower().replace("'", "").replace("\u2019", "")
    words = [w for w in re.split(r"[^a-z]+", cleaned) if w]
    return words, " ".join(words)


def is_continue_request(text: str) -> bool:
    """True if ``text`` answers "want me to keep going?" with yes.

    A bare "yes" or "sure please" is all filler, which here means yes; the
    caller only asks when a remainder is actually being held.
    """
    words, _ = _normalise_reply(text)
    if not words:
        return False
    if words[0] in _NO_WORDS:
        return False
    if words[0] in _YES_WORDS:
        # "Yes, thank you" is a yes; the politeness is not a second answer.
        words = [w for w in words if w not in {"thanks", "thank", "you"}]
    kept = " ".join(w for w in words if w not in _REPLY_FILLERS)
    return kept in _CONTINUES


def is_decline(text: str) -> bool:
    """True if ``text`` answers "want me to keep going?" with no."""
    words, joined = _normalise_reply(text)
    if not words or words[0] in _YES_WORDS:
        return False
    if joined in _DECLINES:
        return True
    kept = " ".join(w for w in words if w not in _REPLY_FILLERS or w == "no")
    return kept in _DECLINES


@dataclass
class HeldRemainder:
    """The unspoken rest of a long reply, offered for a limited time.

    Expires so that "yes" ten minutes later, to a different question, is not
    answered with the tail of an old one.
    """

    ttl_s: float = 120.0
    text: str = ""
    at: float = 0.0

    def hold(self, text: str, now: float) -> None:
        self.text, self.at = text, now

    def take(self, now: float) -> str:
        """Return and clear the remainder, or "" if none or expired."""
        text, fresh = self.text, (now - self.at) <= self.ttl_s
        self.clear()
        return text if fresh else ""

    def pending(self, now: float) -> bool:
        return bool(self.text) and (now - self.at) <= self.ttl_s

    def clear(self) -> None:
        self.text, self.at = "", 0.0
