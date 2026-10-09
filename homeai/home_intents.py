# Copyright (c) 2026 Adam Wosotowsky
# SPDX-License-Identifier: BUSL-1.1
# Licensed under the Business Source License 1.1; see LICENSE at the
# repository root. Converts to Apache-2.0 on 2030-09-15.
"""Recognise everyday home commands without the language model.

"Turn off the foyer", "set a pasta timer for ten minutes", "how long is left
on the pasta timer". These are the commands said most often, and the ones
where a fast, certain answer matters. The 8B voice model leaks malformed
tool-call markup on 10-20% of tool-calling turns (ROUTING_GUIDE); a regex
does not. This is the same design as Home Assistant's own Assist, which
tries local intents before any LLM.

Anything not recognised returns None and goes to the model as before, which
also has the same actions as MCP tools for unusual phrasings. So a pattern
here should be *precise*: a miss costs a second of model time, a false match
does the wrong thing to the house.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

# -- numbers and durations ----------------------------------------------------

_UNITS = {"zero": 0, "one": 1, "two": 2, "three": 3, "four": 4, "five": 5, "six": 6,
          "seven": 7, "eight": 8, "nine": 9, "ten": 10, "eleven": 11, "twelve": 12,
          "thirteen": 13, "fourteen": 14, "fifteen": 15, "sixteen": 16, "seventeen": 17,
          "eighteen": 18, "nineteen": 19, "a": 1, "an": 1, "a couple of": 2, "a couple": 2,
          "couple of": 2, "a few": 3, "few": 3}
_TENS = {"twenty": 20, "thirty": 30, "forty": 40, "fifty": 50, "sixty": 60,
         "seventy": 70, "eighty": 80, "ninety": 90}
_UNIT_SECONDS = {"second": 1, "sec": 1, "minute": 60, "min": 60, "hour": 3600, "hr": 3600}

_NUMBER = (r"(?:\d+(?:\.\d+)?|(?:" + "|".join(sorted(_TENS, key=len, reverse=True)) + r")"
           r"(?:[ -](?:one|two|three|four|five|six|seven|eight|nine))?|"
           + "|".join(sorted((k for k in _UNITS), key=len, reverse=True)) + r")")
_UNIT = r"(?:hours?|hrs?|minutes?|mins?|seconds?|secs?)"


def parse_number(text: str) -> float | None:
    text = text.strip().lower().replace("-", " ")
    try:
        return float(text)
    except ValueError:
        pass
    if text in _UNITS:
        return float(_UNITS[text])
    parts = text.split()
    if parts and parts[0] in _TENS:
        rest = " ".join(parts[1:])
        if not rest:
            return float(_TENS[parts[0]])
        if rest in _UNITS and 0 < _UNITS[rest] < 10:
            return float(_TENS[parts[0]] + _UNITS[rest])
    return None


_DURATION_PART = re.compile(
    rf"({_NUMBER})(?:\s+and\s+(a\s+half|a\s+quarter))?\s*[- ]?\s*({_UNIT})(?:\s+and\s+a\s+half)?",
    re.I)


def parse_duration(text: str) -> int | None:
    """'10 minutes', 'an hour and a half', 'two and a half minutes',
    '1 hour 20 minutes', 'half an hour', '90 seconds' -> seconds."""
    t = text.lower().strip().replace(",", " ")
    t = re.sub(r"\s+", " ", t)
    if re.fullmatch(r"(?:a\s+)?half\s+(?:an?\s+)?hour", t):
        return 1800
    if re.fullmatch(r"(?:a\s+)?quarter\s+(?:of\s+)?(?:an?\s+)?hour", t):
        return 900
    total = 0.0
    pos = 0
    matched = False
    for m in _DURATION_PART.finditer(t):
        between = t[pos:m.start()].strip()
        if between not in ("", "and"):
            return None
        value = parse_number(m.group(1))
        if value is None:
            return None
        if m.group(2):
            value += 0.5 if "half" in m.group(2) else 0.25
        if m.group(0).rstrip().endswith("and a half"):
            value += 0.5
        unit = re.sub(r"s$", "", m.group(3).lower())
        total += value * _UNIT_SECONDS[unit]
        pos = m.end()
        matched = True
    if not matched or t[pos:].strip():
        return None
    return int(round(total)) if total > 0 else None


# -- intents ------------------------------------------------------------------

@dataclass(frozen=True)
class HomeIntent:
    kind: str                       # lights | timer_set | timer_cancel | timer_status | announce | music
    args: dict = field(default_factory=dict)


_POLITE_HEAD = re.compile(
    r"^(?:(?:please|okay|ok|so|um|uh|hey|and|now)\b[,\s]*)*"
    r"(?:(?:can|could|would|will) you(?: please)?\s+)?(?:please\s+)?", re.I)
_POLITE_TAIL = re.compile(r"(?:[,\s]+(?:please|for me|thanks|thank you|now))+$", re.I)


def _clean(text: str) -> str:
    t = text.strip().lower()
    t = re.sub(r"[\u2019`]", "'", t)
    t = re.sub(r"[.!?]+$", "", t).strip()
    t = re.sub(r"\s+", " ", t)
    t = _POLITE_HEAD.sub("", t)
    t = _POLITE_TAIL.sub("", t)
    return t.strip(" ,")


_LABEL = r"(?:(?!timer\b)[a-z']+(?: (?!timer\b)[a-z']+){0,2})"
_TIMER_FOR = re.compile(
    rf"^(?:set|start|make|create)(?: me)? (?:a |an |the )?(?:(?P<label>{_LABEL}) )?timer "
    rf"(?:for|of) (?P<dur>.+?)(?: (?:called|named|labeled|labelled|for(?: the)?) (?P<label2>{_LABEL}))?$")
_TIMER_LEAD = re.compile(
    rf"^(?:set|start|make|create)(?: me)? (?:a |an )?(?P<dur>{_NUMBER}(?:[ -]and[ -]a[ -]half)?[ -]{_UNIT}) "
    rf"(?:(?P<label>{_LABEL}) )?timer(?: (?:called|named|for(?: the)?) (?P<label2>{_LABEL}))?$")
_TIMER_CANCEL = re.compile(
    rf"^(?:cancel|stop|delete|remove|clear|end|kill|turn off)(?: the| my| all(?: the| my| of the| of my)?)?"
    rf"(?: (?P<label>{_LABEL}))? timers?$")
_TIMER_STATUS = (
    re.compile(rf"^how (?:much (?:time )?|long )(?:is |'s )?(?:left|remaining|to go)"
               rf"(?: on (?:the |my )?(?:(?P<label>{_LABEL}) )?timer)?$"),
    re.compile(rf"^how(?: is|'s) (?:the |my )(?:(?P<label>{_LABEL}) )?timer(?: doing| going)?$"),
    re.compile(r"^(?:what|which|any) timers?(?: are| is)?(?: there| running| set| going)?$"),
    re.compile(r"^(?:check|list) (?:the |my )?timers?$"),
    re.compile(r"^(?:are|is) there (?:any )?timers?(?: running| set| going)?$"),
)

_ON_OFF = r"(?P<action>on|off)"
_LIGHT_PATTERNS = (
    re.compile(rf"^(?:turn|switch|shut) {_ON_OFF} (?:the )?(?P<target>.+?)$"),
    re.compile(rf"^(?:turn|switch|shut) (?P<target>.+?) {_ON_OFF}$"),
    re.compile(rf"^(?:the )?lights? {_ON_OFF}(?: in (?:the )?(?P<target>.+))?$"),
    re.compile(rf"^(?:the )?(?P<target>.+?) lights? {_ON_OFF}$"),
)
_DIM = re.compile(r"^dim (?:the )?(?P<target>.+?)(?: (?:to|down to) (?P<pct>\d{1,3})(?: ?%| percent)?)?$")
_BRIGHTEN = re.compile(r"^(?:brighten|turn up) (?:the )?(?P<target>.+?)(?: lights?)?$")
_SET_LEVEL = re.compile(
    r"^(?:set|put|turn) (?:the )?(?P<target>.+?) (?:to|at) (?P<pct>\d{1,3})(?: ?%| percent)$")

_ANNOUNCE = (
    re.compile(r"^(?:make an )?announce(?:ment)?(?: to (?:everyone|everybody|the house|the whole house))?"
               r"(?: that)?[:,]? (?P<msg>.+)$"),
    re.compile(r"^tell (?:everyone|everybody|the house|the whole house)(?: that)?[:,]? (?P<msg>.+)$"),
)

_MUSIC_WORDS = re.compile(
    r"\b(?:music|songs?|album|playlist|radio|station|tunes|jazz|blues|rock|pop|classical|"
    r"country|hip hop|rap|folk|soul|funk|reggae|metal|indie|oldies|christmas|lofi|lo fi|"
    r"soundtrack|amazon music)\b")
_PLAY = re.compile(r"^play (?P<what>.+?)(?: (?:in|on) the (?P<where>[a-z' ]+?))?$")
_MUSIC_CONTROL = (
    (re.compile(r"^(?:stop|turn off|kill) (?:the |that )?(?:music|song|radio)$"), "stop"),
    (re.compile(r"^pause (?:the |that )?(?:music|song|radio)$"), "pause"),
    (re.compile(r"^(?:resume|unpause|continue) (?:the |that )?(?:music|song|radio)$"), "resume"),
    (re.compile(r"^(?:skip|next)(?: (?:this|the|that))?(?: song| track)?$|^next song$|^skip (?:this|the) song$"),
     "next"),
)


_ARTICLES = {"the", "my", "a", "an", "that", "this"}


def _label(*values: str | None) -> str | None:
    for v in values:
        if v and v.strip():
            # Mishearings leave stray articles: "set aside the timer" -> "aside".
            words = v.strip().split()
            while words and words[0] in _ARTICLES:
                words.pop(0)
            while words and words[-1] in _ARTICLES:
                words.pop()
            return " ".join(words) or None
    return None


def parse(text: str) -> HomeIntent | None:
    t = _clean(text)
    if not t:
        return None

    # Timers before lights: "turn off the pasta timer" is not a light.
    for m in (_TIMER_FOR.match(t), _TIMER_LEAD.match(t)):
        if m:
            seconds = parse_duration(m.group("dur"))
            if seconds:
                return HomeIntent("timer_set", {"seconds": seconds,
                                                "label": _label(m.group("label"), m.group("label2"))})
    m = _TIMER_CANCEL.match(t)
    if m:
        label = _label(m.group("label"))
        if re.match(r"^(?:cancel|stop|delete|remove|clear|end|kill|turn off) all\b", t):
            label = "all"
        return HomeIntent("timer_cancel", {"label": label})
    for pattern in _TIMER_STATUS:
        m = pattern.match(t)
        if m:
            return HomeIntent("timer_status", {"label": _label(m.groupdict().get("label"))})

    for pattern, action in _MUSIC_CONTROL:
        if pattern.match(t):
            return HomeIntent("music", {"action": action})
    m = _PLAY.match(t)
    if m and _MUSIC_WORDS.search(m.group("what")):
        what = re.sub(r"\s+on amazon music$", "", m.group("what"))
        return HomeIntent("music", {"action": "play", "request": what, "where": m.group("where")})

    for pattern in _ANNOUNCE:
        m = pattern.match(t)
        if m:
            return HomeIntent("announce", {"message": m.group("msg").strip()})

    m = _SET_LEVEL.match(t)
    if m and "timer" not in t:
        return HomeIntent("lights", {"action": "set", "target": _light_target(m.group("target")),
                                     "brightness": int(m.group("pct"))})
    m = _DIM.match(t)
    if m:
        pct = m.group("pct")
        return HomeIntent("lights", {"action": "set" if pct else "dim",
                                     "target": _light_target(m.group("target")),
                                     "brightness": int(pct) if pct else None})
    m = _BRIGHTEN.match(t)
    if m:
        return HomeIntent("lights", {"action": "brighten", "target": _light_target(m.group("target"))})
    for pattern in _LIGHT_PATTERNS:
        m = pattern.match(t)
        if m:
            target = _light_target(m.groupdict().get("target") or "")
            # "turn off the music/tv/timer" are not lights; leave them to the model.
            if target and re.search(r"\b(?:music|song|radio|tv|television|timer|alarm|"
                                    r"computer|fan speed|volume|garage|door|oven|stove|it|that|this)\b",
                                    target) and not re.search(r"\blights?\b|\blamp\b", target):
                return None
            return HomeIntent("lights", {"action": m.group("action"), "target": target or None})
    return None


def _light_target(target: str) -> str:
    t = target.strip()
    if re.match(r"^(?:all|every|everything)\b", t):
        return "all"
    t = re.sub(r"^the\s+", "", t)
    t = re.sub(r"^lights?\s+in\s+(?:the\s+)?", "", t)
    t = re.sub(r"\s+lights?$", "", t)
    if t in ("light", "lights"):
        return ""
    return t.strip()
