"""Checking a finished picture before anyone sees it (ART_FEATURES_PLAN F4).

Adam's dragons came out with two heads and up to three riders. A vision
model counts what the features promise ("a single dragon with one head",
"one samurai sitting on the dragon's back"); a picture that gets a count
wrong is redrawn with a new seed, up to ``MAX_REDRAWS`` times, and only then
shown (the last try, if none passed).

Only counts the spike showed a vision model can check (tools/art/probe_vision_check.py,
minicpm-v on 16 labelled pictures): heads 15/16, riders 16/16. Posture
("seated astride?") was near chance, so it is not asked.

Cost (tools/art/probe_check_cost.py, art mode, 2026-11-08): first check ~2.9 s
(the vision model pushes Jarvis's voice model out of Ollama), later checks
~0.7 s, the voice model back in ~1.9 s; peak VRAM 18.4 / 20 GB; drawing no
slower with the vision model loaded.

Pure apart from ``ollama_ask``.
"""
from __future__ import annotations

import base64
import logging
import re
from homeai import httpjson
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

log = logging.getLogger("homeai.art.checker")

MODEL = "minicpm-v"
MAX_REDRAWS = 3
TIMEOUT_S = 60

COUNTS = {"a": 1, "an": 1, "one": 1, "single": 1, "a single": 1, "lone": 1, "two": 2, "three": 3, "four": 4}
CREATURES = ("dragon", "wyvern", "horse", "unicorn", "pegasus", "griffin", "gryphon", "lion", "tiger", "wolf",
             "bear", "cat", "dog", "fox", "eagle", "bird", "owl", "phoenix", "serpent", "snake", "elephant",
             "deer", "stag", "bull", "dinosaur", "shark", "whale")
PEOPLE = ("samurai", "knight", "rider", "person", "man", "woman", "girl", "boy", "warrior", "wizard", "witch",
          "pirate", "ninja", "princess", "prince", "king", "queen", "soldier", "cowboy", "child", "kid", "elf")
_COUNT = r"(a single|single|lone|one|two|three|four|a|an)"
SINGULAR = {"heads": "head", "riders": "rider", "people": "person"}
_HEADS = re.compile(r"\b(?:with )?" + _COUNT + r" heads?\b")
_MANY_HEADS = re.compile(r"\b(hydra|(two|three|multi|many)[- ]headed)\b")
_RIDING = re.compile(r"\b(riding|ridden|rides|sitting on|seated on|sits on|astride|mounted on|on (?:its|his|her|"
                     r"the [a-z]+'s) back)\b")
_NUMBER = re.compile(r"\b(\d+|zero|none|no|one|two|three|four|five|six)\b")
_WORDS = {"zero": 0, "none": 0, "no": 0, "one": 1, "two": 2, "three": 3, "four": 4, "five": 5, "six": 6}


@dataclass(frozen=True)
class Check:
    kind: str           # "heads", "riders" or "people"
    question: str
    expect: int
    noun: str


def _find(words: tuple[str, ...], text: str) -> str | None:
    return next((w for w in words if re.search(rf"\b{w}s?\b", text)), None)


def _count_before(noun: str, text: str) -> int | None:
    match = re.search(_COUNT + r" (?:[a-z'-]+ ){0,4}" + noun + r"s?\b", text)
    return COUNTS[match.group(1)] if match else None


def checklist(features) -> list[Check]:
    """The counts the features promise. Nothing asked about what they don't state or can't be counted."""
    checks: list[Check] = []
    subject = next((f.text.lower() for f in features if f.kind == "subject"), "")
    creature = _find(CREATURES, subject)
    if creature and not _MANY_HEADS.search(subject) and _count_before(creature, subject) == 1:
        heads = _HEADS.search(subject)
        expect = COUNTS[heads.group(1)] if heads else 1
        checks.append(Check("heads", f"How many heads does the {creature} have? Answer with one number only.",
                            expect, creature))
    if creature:
        for f in features:
            text = f.text.lower()
            person = _find(PEOPLE, text)
            if f.kind in ("subject", "element") and person and _RIDING.search(text):
                count = _count_before(person, text)
                if count is not None:
                    checks.append(Check("riders", f"How many people are riding the {creature}? "
                                                  "Answer with one number only.", count, creature))
                    break
    people = None if creature else people_count(subject)
    if people:
        checks.append(Check("people", "How many people are in this picture? Answer with one number only.",
                            people, "person"))
    return checks


def people_count(text: str) -> int | None:
    """How many people a subject names, when every part of it is a counted person ("a knight and a wizard
    shaking hands" -> 2), else None: an uncounted "squire" or "crowd" would make the check fail a right picture.

    Measured 2026-11-08 on the multi-subject bake-off (two_people scene, 48 pictures): minicpm-v's people count
    matched the contact sheet on 48/48; its 7 flags were boogu's background crowds.
    """
    parts = [p.strip() for p in re.split(r",| and ", text) if p.strip()]
    total = 0
    for part in parts:
        match = re.match(_COUNT + r" (?:[a-z'-]+ ){0,3}?(" + "|".join(PEOPLE) + r")s?\b", part)
        if not match:
            return None
        total += COUNTS[match.group(1)]
    return total or None


def parse_count(answer: str) -> int | None:
    match = _NUMBER.search(answer.lower())
    if not match:
        return None
    word = match.group(1)
    return int(word) if word.isdigit() else _WORDS[word]


def problems(image: Path, checks: list[Check], ask: Callable[[Path, str], str]) -> list[str]:
    """What's wrong ("2 heads", "3 riders"); [] if every count matched or couldn't be read.

    An unreadable answer is not a failure: redrawing on a model's shrug would waste a turn.
    """
    found = []
    for check in checks:
        got = parse_count(ask(image, check.question))
        if got is not None and got != check.expect:
            found.append(f"{got} {check.kind if got != 1 else SINGULAR[check.kind]}")
    return found


def ollama_ask(image: Path, question: str, model: str = MODEL) -> str:
    messages = [{"role": "user", "content": question, "images": [base64.b64encode(Path(image).read_bytes()).decode()]}]
    return httpjson.ollama_chat(model, messages, keep_alive="2m", options={"temperature": 0},
                                timeout=TIMEOUT_S).strip()
