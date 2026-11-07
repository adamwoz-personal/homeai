"""What a spoken sentence asks the art session to do (AVATAR_PLAN §7.5, 6a step 7).

Fast path first: fixed phrases ("again", "go back two", "the second one") are
matched here with no model call. Only creative changes ("make it bluer") fall
through as ``edit`` and go to the voice model as a PromptSpec patch. A sentence
that names nothing art-like returns None, so ordinary talk ("good night") never
becomes a picture. Pure: no I/O.
"""
from __future__ import annotations

import re
from dataclasses import dataclass

NUMBERS = {"one": 1, "two": 2, "three": 3, "four": 4, "five": 5, "six": 6, "seven": 7, "eight": 8, "nine": 9,
           "ten": 10, "a couple of": 2, "a couple": 2, "a few": 4, "some": 4}
ORDINALS = {"first": 1, "second": 2, "third": 3, "fourth": 4, "last": -1}
MAX_VARIATIONS = 4
NUM = r"(\d+|one|two|three|four|five|six|seven|eight|nine|ten|a couple of|a couple|a few|some)"


@dataclass(frozen=True)
class Intent:
    kind: str
    n: int | None = None
    text: str = ""
    name: str = ""


def _number(word: str | None, default: int = 1) -> int:
    if not word:
        return default
    return int(word) if word.isdigit() else NUMBERS[word]


def _tidy(text: str) -> str:
    text = text.lower().strip()
    text = re.sub(r"^(hey jarvis|jarvis)[, ]+", "", text)
    text = re.sub(r"^(please|can you|could you|now)[, ]+", "", text)
    text = re.sub(r"[, ]+please$", "", text)
    return re.sub(r"[.!?]+$", "", text).strip()


PATTERNS: list[tuple[str, re.Pattern]] = [(kind, re.compile(rx)) for kind, rx in [
    ("cancel", r"^(stop|never ?mind|cancel( that| it| the picture)?|forget it)$"),
    ("leave", r"^((leave|exit|quit|close) (the )?art mode|i'?m (done|finished) drawing|stop drawing|"
              r"that'?s enough drawing)$"),
    ("again", r"^((try )?again|another one|one more|do it again|draw it again|redo it|re-?roll)$"),
    ("variations", rf"^(give me|show me|draw|make)( me)? {NUM}( more)?( options| versions| variations| ones)?$"),
    ("variations", r"^(a few|some) (options|versions|variations)$"),
    ("back", rf"^(go |step )?back( {NUM})?( steps?)?$"),
    ("back", r"^undo( that)?$"),
    ("select", r"^(i like |pick |use )?(the )?(first|second|third|fourth|last) one$"),
    ("keep", r"^(keep|save|pin) (that|this|it)( one)?( as (my )?(?P<name>[a-z0-9' -]{1,40}))?$"),
    ("what", r"^(what prompt (made|did) (this|that|it)|what('s| is| was) the prompt|what did you (use|tell it))$"),
    ("compare", r"^(show (them|those|me them) side by side|compare (them|those))$"),
    ("upscale", r"^(upscale (it|that|this)|make it (high|higher) resolution)$"),
    ("show_last", rf"^show me the last {NUM}( pictures| ones)?$"),
]]

NEW = re.compile(r"^(draw|paint|sketch|generate|create|make)( me)? (?!(it|them|her|him|this|that|the picture|"
                 r"everything)\b)(?P<what>.{2,})$")
EDIT_START = re.compile(r"^(make|turn|change|add|remove|take out|get rid of|put|give|use|set|move|show it|"
                        r"less|more|without|with|now)\b")

# Words that mean the house or Jarvis, not the picture: "turn off the lights" during art mode is home control.
HOUSE = re.compile(r"\b(lights?|lamps?|timers?|alarms?|volume|music|song|thermostat|temperature|heat|tv|television|"
                   r"fans?|doors?|locks?|echo|alexa|weather|remind(er)?|list|shopping|calendar|notes?)\b")


def parse(text: str) -> Intent | None:
    said = _tidy(text)
    if not said:
        return None
    for kind, pattern in PATTERNS:
        match = pattern.match(said)
        if not match:
            continue
        if kind in ("variations", "back", "show_last"):
            numbers = [g for g in match.groups() if g and (g.isdigit() or g in NUMBERS)]
            default = 4 if kind == "variations" else 1
            n = _number(numbers[0] if numbers else None, default)
            if kind == "variations":
                n = max(1, min(n, MAX_VARIATIONS))
            return Intent(kind, n=n)
        if kind == "select":
            return Intent(kind, n=ORDINALS[match.group(3)])
        if kind == "keep":
            return Intent(kind, name=(match.group("name") or "").strip())
        return Intent(kind)
    if HOUSE.search(said):
        return None
    new = NEW.match(said)
    if new:
        return Intent("new", text=said)
    if EDIT_START.match(said):
        return Intent("edit", text=said)
    return None
