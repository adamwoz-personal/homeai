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
ORDINALS = {"first": 1, "second": 2, "third": 3, "fourth": 4, "fifth": 5, "sixth": 6, "seventh": 7, "eighth": 8,
            "last": -1}
MAX_VARIATIONS = 4
MAX_SHOWN = 8           # "show me the last N" and picking by number: the picture window's strip holds 8
PICK_WORD = r"([1-8]|one|two|three|four|five|six|seven|eight|first|second|third|fourth|fifth|sixth|seventh|eighth)"
NUM = r"(\d+|one|two|three|four|five|six|seven|eight|nine|ten|a couple of|a couple|a few|some)"


@dataclass(frozen=True)
class Intent:
    kind: str
    n: int | None = None
    text: str = ""
    name: str = ""
    picks: tuple[int, ...] = ()


def _number(word: str | None, default: int = 1) -> int:
    if not word:
        return default
    if word in ORDINALS:
        return ORDINALS[word]
    return int(word) if word.isdigit() else NUMBERS[word]


# How people end a long description in art mode: "...on a cliff. That's it, go ahead and draw it."
SIGN_OFF = re.compile(r"(?:[.,!?]?\s*(?:and |okay |ok |so |um |uh )?(?:that'?s it|that'?s all|that'?s everything|"
                      r"go ahead(?: and draw it)?|draw it(?: now)?|draw that(?: now)?|you can draw it(?: now)?))+[.!?]*$")


WAKE = re.compile(r"\b(?:hey|hi|okay|ok) (?:jarvis|jervis|travis)\b[,.!]*\s*")


def _tidy(text: str) -> str:
    text = text.lower().strip()
    # "Draw me a dragon. Hey Jarvis, let's draw a dragon flying over the ocean." (Adam, 2026-11-07): saying the wake
    # word again restarts the request, so the words after the last one are what was meant.
    parts = WAKE.split(text)
    if len(parts) > 1 and len(parts[-1].split()) >= 2:
        text = parts[-1]
    text = re.sub(r"^(?:hey |hi |okay |ok )?(?:jarvis|jervis)[,.!]*\s*", "", text)
    text = re.sub(r"^(please|can you|could you|now)[, ]+", "", text)
    text = re.sub(r"[, ]+please$", "", text)
    trimmed = SIGN_OFF.sub("", text).strip()
    if len(trimmed) > 2:
        text = trimmed
    return re.sub(r"[.!?]+$", "", text).strip()


PATTERNS: list[tuple[str, re.Pattern]] = [(kind, re.compile(rx)) for kind, rx in [
    ("cancel", r"^(stop|never ?mind|cancel( that| it| the picture)?|forget it)$"),
    # Adam's tries on 2026-11-12, all missed: "switch to voice mode", "cancel art mode" (heard as "cancel art"),
    # "we're done drawing"; and he calls it "drawing mode".
    ("leave", r"^((leave|exit|quit|close|cancel|end|stop|turn off|switch off|get out of) (the )?"
              r"(art( mode)?|(drawing|draw|picture) mode)|((go|switch|change) )?(back )?to (voice|normal|talking) mode|"
              r"voice mode|(i'?m|we'?re|i am|we are) (all )?(done|finished)( drawing| with (the )?(drawing|pictures?|art))?|"
              r"stop drawing|no more drawing|that'?s enough drawing)$"),
    ("again", r"^((try )?again|another one|one more|do it again|draw it again|redo it|re-?roll)$"),
    ("variations", rf"^(give me|show me|draw|make)( me)? {NUM}( more)?( options| versions| variations| ones)?$"),
    ("variations", r"^(a few|some) (options|versions|variations)$"),
    ("back", rf"^(go |step )?back( {NUM})?( steps?)?$"),
    ("back", r"^undo( that)?$"),
    ("select", r"^(i like |pick |use |show me |go with |let'?s (see|go with|have) )?(the )?"
               r"(first|second|third|fourth|fifth|sixth|seventh|eighth|last) "
               r"(one|picture|image|option|try)$"),
    ("keep", r"^(keep|save|pin) (that|this|it)( one)?( as (my )?(?P<name>[a-z0-9' -]{1,40}))?$"),
    ("what", r"^(what prompt (made|did) (this|that|it)|what('s| is| was) the prompt|what did you (use|tell it))$"),
    ("compare", r"^((show )?(them |those |me them |it )?side by side|compare( (them|those|it|that|this))?"
                r"( (with|to) (the last one|the one before|before|what it was))?)$"),
    # "Make it bigger" stays an edit (the subject's size); only these words mean more pixels.
    ("upscale", r"^(upscale( (it|that|this))?|make (it|that|this) (high|higher) resolution|(high|higher) resolution"
                r"( please)?|(make (it|that|this) )?print size|(make )?a print( size)? (version|copy))$"),
    ("show_last", rf"^show( me)? the last {NUM}( pictures| ones| images)?$"),
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


# Picking from the numbered options by number: "image three", "what is image three", "option 3", "number three".
# All three of Adam's tries went to the patcher instead and changed the picture (2026-11-07).
PICK = re.compile(r"^(?:(?:i like|pick|use|show me|go with|let'?s (?:see|go with|have)|what(?:'s| is| about)|"
                  r"how about|can i see|i want) )?(?:the )?(?:image|option|picture|number|try|version|choice|one) "
                  r"(?:number )?(?P<n>[1-8]|one|two|three|four|five|six|seven|eight)$")
# Two numbered pictures side by side: "compare two and four", "show the first and third side by side".
COMPARE_TWO = re.compile(rf"^(?:compare|show(?: me)?|put) (?:the |number |option )?(?P<a>{PICK_WORD[1:-1]})(?: one)? "
                         rf"(?:and|with|to|against) (?:the |number |option )?(?P<b>{PICK_WORD[1:-1]})(?: ones?)?"
                         rf"(?P<side> side by side)?$")
# "Again. Oh gosh, it's fake again..." took the chatter after "again" as a change (2026-11-07).
AGAIN_THEN = re.compile(r"^(?:try again|again|another one|one more|re-?roll)[.!,]+\s+(?P<rest>.+)$")


def parse(text: str) -> Intent | None:
    said = _tidy(text)
    if not said:
        return None
    pick = PICK.match(said)
    if pick:
        return Intent("select", n=_number(pick.group("n")))
    two = COMPARE_TWO.match(said)
    if two and (said.startswith("compare") or two.group("side")):
        return Intent("compare", picks=(_number(two.group("a")), _number(two.group("b"))))
    again = AGAIN_THEN.match(said)
    if again:
        rest = again.group("rest")
        return Intent("edit", text=rest) if is_complaint(rest) else Intent("again")
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
            return Intent(kind, n=ORDINALS[next(g for g in match.groups() if g in ORDINALS)])
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
