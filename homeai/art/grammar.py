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


# How people end a long description in art mode: "...on a cliff. That's it, go ahead and draw it."
SIGN_OFF = re.compile(r"(?:[.,!?]?\s*(?:and |okay |ok |so |um |uh )?(?:that'?s it|that'?s all|that'?s everything|"
                      r"go ahead(?: and draw it)?|draw it(?: now)?|draw that(?: now)?|you can draw it(?: now)?))+[.!?]*$")


def _tidy(text: str) -> str:
    text = text.lower().strip()
    text = re.sub(r"^(hey jarvis|jarvis)[, ]+", "", text)
    text = re.sub(r"^(please|can you|could you|now)[, ]+", "", text)
    text = re.sub(r"[, ]+please$", "", text)
    trimmed = SIGN_OFF.sub("", text).strip()
    if len(trimmed) > 2:
        text = trimmed
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


# "Let's start with a new picture. Let's go with a medieval frigate..." was taken as an edit of the dragon
# (Adam, 2026-11-07). A fresh start, with or without what to draw next.
FRESH = re.compile(r"\b(?:start|begin|do|make|try|draw|want)?\s*(?:with |on )?(?:a )?(?:brand )?new "
                   r"(?:picture|drawing|image|painting)\b[.,:;!]*\s*(?P<rest>.*)$")
LEAD_IN = re.compile(r"^(?:of |with )?(?:(?:let'?s|we'?ll|i'?d like to|i want to) )?(?:go with|do|have|make it|try|draw|paint)?\s*"
                     r"(?:(?:a )?(?:picture|drawing|image) of )?")
# A complaint about what the model drew (extra heads, riders, merged limbs): redraw with new seeds, not the same one.
COMPLAINT = re.compile(r"\b(?:ended up with|came out with|there'?s|it has|it'?s got|has|got) (?:a |an )?"
                       r"(?:second|third|extra|another|two|three|four|multiple|too many)\b|"
                       r"\b(?:extra|too many|second|third|two|three|duplicate) (?:heads?|riders?|legs?|arms?|tails?|"
                       r"wings?|hands?|feet|eyes|people|faces?)\b|\b(?:merged|fused|melted|growing out of|"
                       r"stuck to|attached to)\b|\bneeds? to go\b|\bcame out (?:wrong|weird|broken)\b")


def is_complaint(text: str) -> bool:
    return bool(COMPLAINT.search(_tidy(text)))


def parse(text: str) -> Intent | None:
    said = _tidy(text)
    if not said:
        return None
    fresh = FRESH.search(said)
    if fresh:
        rest = LEAD_IN.sub("", fresh.group("rest").strip()).strip(" .,")
        return Intent("new", text=f"draw {rest}") if len(rest) > 2 else Intent("fresh")
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
    if new and said.startswith("make the "):
        return Intent("edit", text=said)        # "make the dragon red" changes it; "make a dragon" is new
    if new:
        return Intent("new", text=said)
    if EDIT_START.match(said):
        return Intent("edit", text=said)
    return None
