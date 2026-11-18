"""Look, notice, reword: what the image model is told, as opposed to what Adam's features say (ART_FEATURES_PLAN F8).

On 2026-11-18 "an anime girl wielding a katana fighting a chimera" came out wrong 10/10 with klein9: a dragon for
the chimera every time, and two blades every time. Words about counts ("one katana with one blade") didn't help.
Describing instead of naming did (tools/art/bench_wording.py, 8 seeds each):

    old wording        one blade 0/18   goat's head 0/18   snake tail 0/18
    descriptive (v3)   one blade 10/16 (v2+v3, same sword words)   goat's head 5-6/8   snake tail 8/8

So the features Adam sees stay as he said them, and the prompt gets, per feature, in this order:
  1. this picture's own rewrite, if the feature's text hasn't changed since (``spec["wording"]``);
  2. a rewrite that fixed the same feature text before (``Learned``, saved when a reworded picture passed);
  3. the glossary below: names klein9 gets wrong, said as what they look like, each with its own checks;
  4. the feature's text as it is.

When a picture fails its checks twice, ``reword`` asks the voice model for new wording for the failing feature,
given what the checks found and what the picture actually shows. Pure apart from ``Learned`` and ``reword``'s chat.
"""
from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

from homeai.fsutil import atomic_write_json

log = logging.getLogger("homeai.art.wording")

MAX_WORDING = 420             # the glossary alone makes Adam's one-feature sentence 266 characters

# Questions measured on 42 labelled pictures with Jarvis's voice model, thinking off (tools/art/probe_vision_parts.py,
# 2026-11-18): blades 26/34 (good pictures flagged 2/13, wrong ones caught 15/21), goat's head 40/42, snake 37/42.
BLADES_Q = "How many sword blades are in this picture? Answer with one number only."
GOAT_Q = "Is there a goat's head on the creature's back? Answer yes or no."
SNAKE_Q = "Is the creature's tail a snake? Answer yes or no."


@dataclass(frozen=True)
class Ask:
    """One question for the vision model: ``expect`` a number or True/False; ``wrong`` says what failed
    ("{n}" is the number it saw)."""
    kind: str
    question: str
    expect: int | bool
    wrong: str


@dataclass(frozen=True)
class Term:
    pattern: re.Pattern
    visual: str             # replaces the match; "{noun}" is the matched noun
    asks: tuple[Ask, ...]


_HOLD = r"(?:wielding|holding|brandishing|swinging|carrying|gripping|with)"
TERMS = (
    Term(re.compile(r"\b(?:a|an|one|the)\s+chimera\b", re.I),
         "a chimera, a lion with a golden mane that has a second head, a goat's head, rising from the middle of "
         "its back, and a living snake for a tail",
         (Ask("goat", GOAT_Q, True, "no goat's head on its back"),
          Ask("snake", SNAKE_Q, True, "no snake for a tail"))),
    # One hilt, one blade. "With both hands on its one hilt" did better (v2+v3 10/16 vs v4 ~3/8, bench 2026-11-18),
    # but Adam: a katana can be wielded with one hand or two, so the words don't choose; the checker and the
    # redraws make up the difference. Hands the person asks for stay in their own words.
    Term(re.compile(rf"\b{_HOLD}\s+(?:a single|a|one|her|his|their)\s+(?P<noun>katana|sword|longsword)"
                    r"(?:\s+with one blade)?\b", re.I),
         "holding a single {noun} by its one hilt, its one long{curve} blade pointing forward",
         (Ask("blades", BLADES_Q, 1, "{n} sword blades"),)),
)


def expand(text: str) -> tuple[str, list[Ask]]:
    """``text`` with glossary names said as what they look like, and the questions those promise."""
    asks: list[Ask] = []
    for term in TERMS:
        match = term.pattern.search(text)
        if match is None:
            continue
        noun = (match.groupdict().get("noun") or "").lower()
        visual = term.visual.format(noun=noun, curve=" curved" if noun == "katana" else "")
        text = text[:match.start()] + visual + text[match.end():]
        asks += term.asks
    return text, asks


def key(text: str) -> str:
    return re.sub(r"[^a-z0-9 ]", "", text.lower()).strip()


class Learned:
    """Rewrites that fixed a feature, by its text: tried first next time the same words come up."""

    def __init__(self, path: Path) -> None:
        self.path = Path(path)

    def _read(self) -> dict:
        try:
            data = json.loads(self.path.read_text())
            return data if isinstance(data, dict) else {}
        except (OSError, ValueError):
            return {}

    def get(self, text: str) -> str | None:
        value = self._read().get(key(text))
        return value if isinstance(value, str) and value else None

    def remember(self, text: str, wording: str) -> None:
        data = self._read()
        data[key(text)] = wording
        try:
            atomic_write_json(self.path, data, indent=1)
        except OSError as exc:
            log.warning("could not save a learned wording: %s", exc)


def prompt_text(feature, spec: dict, learned: Learned | None = None) -> str:
    """What the image model is told for one feature (precedence in the module docstring)."""
    own = (spec.get("wording") or {}).get(feature.id)
    if isinstance(own, dict) and own.get("for") == feature.text and own.get("text"):
        return own["text"]
    if learned is not None:
        hit = learned.get(feature.text)
        if hit:
            return hit
    return expand(feature.text)[0]


def wordings(features, spec: dict, learned: Learned | None = None) -> dict[str, str]:
    return {f.id: prompt_text(f, spec, learned) for f in features}


QUESTIONS_SCHEMA = {"type": "object", "properties": {"questions": {"type": "array", "items": {"type": "string"}}},
                    "required": ["questions"]}
MAX_ASKED = 2
_YES_NO = re.compile(r"^(is|are|does|do|has|have|can)\b.*\?$", re.I)
_COMPOUND = re.compile(r"\b(and|or)\b", re.I)


def asked_messages(feature_text: str) -> list[dict]:
    return [
        {"role": "system", "content":
            "You check pictures an image model drew. Write at most 2 yes/no questions that someone could answer "
            "by looking at the picture, to check it shows what was asked. Each answer must be YES when the "
            "picture is right. Ask about things that must be visible: body parts, how many, what is held, what "
            "kind of creature. Each question asks about ONE thing (never 'X and Y'). Ask only about what the "
            "words say, not about details of a legend they leave out. Never ask about pose, mood, style, colour "
            "shades or the background. If nothing is worth checking, give no questions. "
            "Answer as JSON: {\"questions\": [\"...?\"]}"},
        {"role": "user", "content": f"What was asked for: {feature_text}"},
    ]


def asked(feature_text: str, chat: Callable[[list[dict], dict], str]) -> list[Ask]:
    """The voice model's own yes/no checks for a feature the glossary doesn't know; [] on any failure."""
    try:
        answer = json.loads(chat(asked_messages(feature_text), QUESTIONS_SCHEMA))
        questions = [re.sub(r"\s+", " ", str(q)).strip() for q in answer.get("questions") or []]
    except (ValueError, AttributeError, TypeError, OSError) as exc:
        log.warning("check questions for %r failed: %s", feature_text, exc)
        return []
    out = []
    for q in questions:
        # 2026-11-18 live: "Does the creature have the body of a horse and the tail of an ox?" failed every
        # ki-rin klein drew, so each picture used all four tries. One thing per question.
        if _YES_NO.match(q) and len(q) <= 160 and not _NEGATIVE.search(q) and not _COMPOUND.search(q):
            out.append(Ask("asked", f"{q} Answer yes or no.", True, f"no to '{q}'"))
    return out[:MAX_ASKED]


REWORD_SCHEMA ={"type": "object", "properties": {"wording": {"type": "string"}}, "required": ["wording"]}


def reword_messages(feature_text: str, current: str, issues: list[str], seen: str) -> list[dict]:
    return [
        {"role": "system", "content":
            "You write prompts for an image model that draws literally and often miscounts. A picture came out "
            "wrong. Rewrite the prompt so the next picture gets the WRONG PART right. Rules: change how the wrong "
            "part is described: make it more concrete, put it first, say its count in words and where it sits. "
            "Describe what should be drawn, never the picture that came out: add nothing the person didn't ask "
            "for (no clothes, hair, colours, or how many hands hold something, of your own). Say only what should "
            "be there, "
            "never what should not (no 'no', 'not', 'without', 'avoid'). Copy every other part of the old prompt "
            "word for word: its words for the parts that came out right are why they came out right. "
            "Under 70 words. Answer as JSON: {\"wording\": \"...\"}"},
        {"role": "user", "content":
            f"The person asked for: {feature_text}\nThe prompt said: {current}\n"
            f"What came out wrong: {', '.join(issues)}\nWhat the picture shows: {seen or 'unknown'}"},
    ]


_NEGATIVE = re.compile(r"\b(no|not|without|avoid|never|don't|isn't)\b", re.I)


def reword(feature_text: str, current: str, issues: list[str], seen: str,
           chat: Callable[[list[dict], dict], str]) -> str | None:
    """New wording for a feature that keeps failing its checks, or None (a bad answer keeps the old one).

    Negative words are refused: klein draws what it's told about ("Avoid two heads" drew two heads, F0)."""
    try:
        answer = json.loads(chat(reword_messages(feature_text, current, issues, seen), REWORD_SCHEMA))
        text = re.sub(r"\s+", " ", str(answer.get("wording") or "")).strip().rstrip(".")
    except (ValueError, AttributeError, TypeError, OSError) as exc:
        log.warning("reword failed: %s", exc)
        return None
    if not text or len(text) > MAX_WORDING or _NEGATIVE.search(text) or text == current:
        log.info("reword refused: %r", text[:120])
        return None
    return text
