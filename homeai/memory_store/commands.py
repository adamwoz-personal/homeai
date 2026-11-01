# Copyright (c) 2026 Adam Wosotowsky
# SPDX-License-Identifier: BUSL-1.1
# Licensed under the Business Source License 1.1; see LICENSE at the
# repository root. Converts to Apache-2.0 on 2030-09-15.
"""Spoken memory commands (plan §7.5). Pure parsing, no side effects.

Anchored patterns only: "remember" in the middle of a sentence, or "do you
remember when ...", is a question for the model, not a command.
"""

from __future__ import annotations

import re
from dataclasses import dataclass


@dataclass(frozen=True)
class Command:
    kind: str        # see PATTERNS
    text: str = ""


_LEAD = r"^(?:(?:ok(?:ay)?|hey|so|and|jarvis)[,\s]+)*(?:please\s+|can you\s+|could you\s+)?"
_END = r"[\s.!?,]*$"
_ME = r"(?:me|myself)"

PATTERNS: list[tuple[str, re.Pattern]] = [(k, re.compile(_LEAD + p + _END, re.I)) for k, p in [
    ("forget_all", rf"forget everything(?: (?:you know|you remember|i(?:'ve| have)? told you))?"
                   rf" about {_ME}"),
    ("forget_thread", r"forget (?:that|the|this) whole (?:idea|thing|conversation|topic|plan)"
                      r"(?: (?:about|on) (?P<text>.+?))?"),
    ("forget_about", r"forget (?:everything |all )?(?:what i (?:said|told you)|"
                     r"(?:my )?(?:conversations?|memories)|everything|all) about (?P<text>.+?)"),
    ("forget_last", r"forget (?:that|this|what i just said|the last thing(?: i said)?)"),
    ("stop", rf"(?:stop remembering(?: (?:things|stuff))?(?: about {_ME})?|"
             rf"(?:don't|do not) remember (?:anything|things) about {_ME})"),
    ("start", rf"(?:start remembering(?: (?:things|stuff))?(?: about {_ME})?(?: again)?|"
              rf"you can remember (?:things|stuff) about {_ME}(?: again)?)"),
    ("recall_me", rf"what do you (?:remember|know) about {_ME}"),
    ("recall_about", r"what do you remember about (?P<text>.+?)"),
    ("remember_always", r"(?:always remember|never forget|permanently remember)"
                        r"(?: that)? (?P<text>.+?)"),
    ("remember", r"(?:remember|make a note)(?: that)? (?P<text>(?!when |what |how |who |where |"
                 r"if |whether |to |my voice|me\b|this voice).+?)"),
]]

_YES = re.compile(_LEAD + r"(?:yes|yeah|yep|yup|sure|correct|do it|go ahead|please do|"
                  r"yes please|forget (?:it|them|those)|delete (?:it|them))" + _END, re.I)
_NO = re.compile(_LEAD + r"(?:no|nope|nah|don't|do not|cancel|keep (?:it|them)|never mind|"
                 r"no thanks?(?: you)?)" + _END, re.I)
_CORRECTION = re.compile(
    _LEAD + r"(?:no[,\s]+)?(?:that's|that is) (?:wrong|not right|not true|incorrect|out of date)"
    r"[\s.,;:!-]+(?:it's |it is |actually |now )?(?P<text>.+?)" + _END, re.I)


def parse(utterance: str) -> Command | None:
    text = " ".join((utterance or "").split())
    for kind, pat in PATTERNS:
        m = pat.match(text)
        if m:
            got = (m.groupdict().get("text") or "").strip()
            if kind in ("remember", "remember_always", "recall_about", "forget_about") \
                    and len(got) < 3:
                return None
            return Command(kind, got)
    return None


def confirmation(utterance: str) -> bool | None:
    """True for yes, False for no, None for anything else."""
    text = " ".join((utterance or "").split())
    if _YES.match(text):
        return True
    if _NO.match(text):
        return False
    return None


def correction(utterance: str) -> str | None:
    """The corrected statement in "that's wrong, it's X", or None."""
    m = _CORRECTION.match(" ".join((utterance or "").split()))
    return m.group("text").strip() if m and len(m.group("text").strip()) >= 3 else None


def first_person_to_third(text: str, name: str) -> str:
    """'my sister is Kate' -> 'Adam's sister is Kate', for a stored note.

    A bare "I <verb>" can't be conjugated reliably ("I like" -> "Adam
    like"), so such notes are quoted instead: 'Adam said: "I like ..."'.
    """
    text = text.strip()
    if re.search(r"\bI\b(?!'m|'ve| am\b)", text, flags=re.I):
        return f'{name} said: "{text[:1].upper() + text[1:]}"'
    out = re.sub(r"\bI am\b|\bI'm\b", f"{name} is", text, flags=re.I)
    out = re.sub(r"\bI've\b", f"{name} has", out, flags=re.I)
    out = re.sub(r"\bmy\b", f"{name}'s", out, flags=re.I)
    out = re.sub(r"\bmine\b", f"{name}'s", out, flags=re.I)
    out = re.sub(r"\bme\b", name, out, flags=re.I)
    return out[:1].upper() + out[1:]
