# Copyright (c) 2026 Adam Wosotowsky
# SPDX-License-Identifier: BUSL-1.1
# Licensed under the Business Source License 1.1; see LICENSE at the
# repository root. Converts to Apache-2.0 on 2030-09-15.
"""What may be remembered and when to recall. Pure functions, no database.

Rules (plans/MEMORY_PLAN.md §7, §13):

* Only answered turns from a recognised speaker are remembered. Unknown,
  unverified and other-speaker turns never are: a stranger or the TV must not
  plant memories under someone's name.
* Piper test voices (``ignore_speakers``) are not people.
* Home commands, closings and refusals are not remembered: Home Assistant
  keeps its own history, and "thanks" is noise that dilutes recall.
* Secrets are redacted before anything is stored.
* Recall is skipped for pleasantries and when nobody is recognised.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

from ..safety import detect_leaked_markup
from ..wake_verify import is_pleasantry_only, strip_wake_phrase

# Verdicts whose exchange is worth remembering.
RECORD_VERDICTS = frozenset({"allow", "continued"})

USERNAME_RE = re.compile(r"^[a-z0-9_]{1,32}$")
RESERVED = frozenset({"household", "guest"})


@dataclass(frozen=True)
class Decision:
    record: bool
    reason: str


def username_for(name: str | None) -> str | None:
    """The database username for a display name: "Adam" -> "adam"."""
    if not name:
        return None
    u = re.sub(r"[^a-z0-9_]+", "_", name.strip().lower()).strip("_")[:32]
    if not u or not USERNAME_RE.match(u) or u in RESERVED:
        return None
    return u


def exchange_problem(*, heard: str, reply: str, verdict: str, ok: bool = True) -> str | None:
    """Why this exchange is not worth keeping, whoever said it; None if it is."""
    if not ok:
        return "failed turn"
    if verdict not in RECORD_VERDICTS:
        return f"verdict {verdict or 'none'}"
    if not (heard or "").strip() or not (reply or "").strip():
        return "empty"
    if detect_leaked_markup(heard) or detect_leaked_markup(reply):
        return "leaked markup"
    return None


def should_record(*, heard: str, reply: str, verdict: str, speaker: str | None, ok: bool = True,
                  ignore_speakers: tuple[str, ...] = ()) -> Decision:
    problem = exchange_problem(heard=heard, reply=reply, verdict=verdict, ok=ok)
    if problem:
        return Decision(False, problem)
    user = username_for(speaker)
    if user is None:
        return Decision(False, "speaker not recognised")
    if user in ignore_speakers:
        return Decision(False, "test voice")
    return Decision(True, "ok")


def clean_heard(text: str) -> str:
    """The utterance without its "hey Jarvis" lead-in, if one is present."""
    text = (text or "").strip()
    stripped = strip_wake_phrase(text)
    return stripped or text


# -- secrets ---------------------------------------------------------------

_SECRET_PHRASE = re.compile(
    r"(?i)\b((?:pass ?word|passcode|pin(?: number)?|password|security code|access code|"
    r"alarm code|gate code|door code|api key|secret|token|card number|social security number)"
    r"\s*(?:is|was|=|:|'s)\s*)(\S+(?:\s+\d+)*)")
_SECRET_TOKENS = [
    re.compile(r"\b(?:sk|pk|rk)-[A-Za-z0-9_\-]{16,}"),
    re.compile(r"\bgh[pousr]_[A-Za-z0-9]{20,}"),
    re.compile(r"\bAKIA[0-9A-Z]{16}\b"),
    re.compile(r"\beyJ[A-Za-z0-9_\-]{10,}\.[A-Za-z0-9_\-]{10,}\.[A-Za-z0-9_\-]{10,}"),
    re.compile(r"\b\d{3}-\d{2}-\d{4}\b"),                    # US SSN shape
    re.compile(r"\b[A-Za-z0-9+/_\-]{40,}={0,2}"),             # long opaque token
]
_CARD = re.compile(r"\b(?:\d[ \-]?){13,19}\b")
REDACTED = "[redacted]"


def _luhn(digits: str) -> bool:
    total, alt = 0, False
    for ch in reversed(digits):
        d = int(ch)
        if alt:
            d = d * 2 - 9 if d > 4 else d * 2
        total += d
        alt = not alt
    return total % 10 == 0


def redact(text: str) -> str:
    """Replace things that look like secrets. Errs towards redacting."""
    if not text:
        return text
    text = _SECRET_PHRASE.sub(lambda m: m.group(1) + REDACTED, text)
    for pat in _SECRET_TOKENS:
        text = pat.sub(REDACTED, text)

    def card(m: re.Match) -> str:
        digits = re.sub(r"\D", "", m.group(0))
        return REDACTED if 13 <= len(digits) <= 19 and _luhn(digits) else m.group(0)

    return _CARD.sub(card, text)


# -- recall gating -----------------------------------------------------------

def should_recall(utterance: str, speaker: str | None) -> bool:
    """Worth a database round trip? Cheap checks only."""
    if username_for(speaker) is None:
        return False
    text = (utterance or "").strip()
    if len(re.findall(r"[A-Za-z0-9']+", text)) < 2:
        return False
    if is_pleasantry_only(text):
        return False
    return not detect_leaked_markup(text)
