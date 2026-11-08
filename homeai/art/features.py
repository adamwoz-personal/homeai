"""Lettered features: the picture's description as a list Adam can see and edit (ART_FEATURES_PLAN F1).

A picture is described by features, each with a letter that stays fixed while
the picture is worked on:

    A  subject   a single red dragon with one head and four legs
    B  element   one samurai in full armor, sitting on the dragon's back
    C  setting   a neon city street at night
    D  style     cyberpunk

The voice model answers each instruction with operations (add, change a
letter, remove a letter) instead of rewriting a whole description, so "make the
dragon red" rewrites A rather than adding a second dragon, and "remove C" works
by letter. The prompt is built from the features in a fixed order: the main
subject first, then things placed relative to it, then where, then the look.
That order is what fixed the rider-on-dragon pictures (bench 2026-11-07:
mount first with counts, 4/4 clean; the old prompts 4/4 broken).

Pure: no I/O. The spec stored on each picture is
``{"features": [...], "aspect": ..., "seed": ...}``.
"""
from __future__ import annotations

import re
import string
from dataclasses import dataclass

from homeai.art.prompt_spec import SIZES, SpecError

KINDS = ("subject", "element", "setting", "style", "colour", "lighting", "camera", "composition")
ORDER = {kind: i for i, kind in enumerate(KINDS)}
MAX_FEATURES = 12
MAX_TEXT = 200
LETTERS = string.ascii_uppercase

OPS_SCHEMA = {
    "type": "object",
    "properties": {
        "about_picture": {"type": "boolean"},
        "ops": {"type": "array", "maxItems": 8, "items": {
            "type": "object",
            "properties": {
                "op": {"type": "string", "enum": ["add", "change", "remove"]},
                "id": {"type": "string", "maxLength": 1},
                "kind": {"type": "string", "enum": list(KINDS)},
                "text": {"type": "string", "maxLength": MAX_TEXT},
            },
            # All required: without "text" the model sent bare {"op": "add", "kind": "subject"} for every
            # "draw a samurai with red armor riding a dragon" (0/5, 2026-11-08); requiring only "text" made it
            # drop "id" (32/45). An add ignores its id, a remove its text.
            "required": ["op", "id", "kind", "text"],
        }},
        "aspect": {"type": ["string", "null"], "enum": [*SIZES, None]},
    },
    "required": ["about_picture", "ops"],
}


@dataclass(frozen=True)
class Feature:
    id: str
    kind: str
    text: str

    def to_json(self) -> dict:
        return {"id": self.id, "kind": self.kind, "text": self.text}


PLACEHOLDER = re.compile(r"^(none|nothing|n/?a|unspecified|unknown|not specified|nowhere( specific| in particular)?|"
                         r"anywhere|no (setting|style|background)|default|generic|any)$")


def _clean(text: str) -> str:
    text = re.sub(r"\s+", " ", str(text)).strip().strip(".")
    if len(text) > MAX_TEXT:
        text = text[:MAX_TEXT].rsplit(" ", 1)[0]
    return text


def load(raw) -> list[Feature]:
    """Features from stored JSON; anything malformed is a SpecError."""
    if not isinstance(raw, list) or len(raw) > MAX_FEATURES:
        raise SpecError(f"features must be a list of up to {MAX_FEATURES}")
    out, seen = [], set()
    for item in raw:
        try:
            fid, kind, text = item["id"], item["kind"], _clean(item["text"])
        except (KeyError, TypeError) as exc:
            raise SpecError(f"bad feature {item!r}") from exc
        if fid not in LETTERS or fid in seen or kind not in KINDS or not text:
            raise SpecError(f"bad feature {item!r}")
        seen.add(fid)
        out.append(Feature(fid, kind, text))
    return out


def dump(features: list[Feature]) -> list[dict]:
    return [f.to_json() for f in features]


def _next_letter(features: list[Feature], used: set[str]) -> str:
    """The first letter never used in this picture's history, so a removed C never comes back as a different C."""
    taken = used | {f.id for f in features}
    for letter in LETTERS:
        if letter not in taken:
            return letter
    raise SpecError("out of letters")


def apply_ops(features: list[Feature], ops: list[dict], used: set[str] | None = None) -> tuple[list[Feature], list[str]]:
    """(new features, the letters that changed). Bad operations are a SpecError: nothing half-applied."""
    current = list(features)
    used = set(used or ()) | {f.id for f in features}
    changed: list[str] = []
    for op in ops:
        kind = op.get("op")
        if kind == "change" and not str(op.get("id") or "").strip():
            # A change with no letter is the model guessing: one replaced the dragon with "A to a steampunk
            # battleship" and another failed the whole change (live and eval, 2026-11-07). Skip it, keep the rest.
            continue
        if kind == "add" and PLACEHOLDER.match(_clean(op.get("text", "")).lower()):
            continue                    # "nowhere specific", "none": the model filling a slot, not a feature
        if kind == "add":
            text, fkind = _clean(op.get("text", "")), op.get("kind") or "element"
            if not text or fkind not in KINDS:
                raise SpecError(f"bad add {op!r}")
            if fkind == "subject" and any(f.kind == "subject" for f in current):
                fkind = "element"       # one main subject; a second "subject" is placed relative to it
            if len(current) >= MAX_FEATURES:
                raise SpecError(f"a picture has at most {MAX_FEATURES} features")
            letter = _next_letter(current, used)
            used.add(letter)
            current.append(Feature(letter, fkind, text))
            changed.append(letter)
        elif kind in ("change", "remove"):
            fid = str(op.get("id", "")).upper()
            index = next((i for i, f in enumerate(current) if f.id == fid), None)
            if index is None:
                raise SpecError(f"there's no feature {fid or '?'}")
            if kind == "remove":
                del current[index]
            else:
                old = current[index]
                text = _clean(op.get("text", "")) or old.text
                fkind = op.get("kind") if op.get("kind") in KINDS else old.kind
                current[index] = Feature(fid, fkind, text)
            changed.append(fid)
        else:
            raise SpecError(f"unknown operation {op!r}")
    if not any(f.kind == "subject" for f in current):
        raise SpecError("a picture needs a main subject")
    return current, changed


COLOURS = ("red", "orange", "yellow", "green", "blue", "purple", "violet", "pink", "black", "white", "grey", "gray",
           "brown", "gold", "golden", "silver", "bronze", "teal", "crimson", "scarlet")
_COLOUR = "(?:" + "|".join(COLOURS) + ")"
# "a samurai with red armor": owner word, then the coloured thing. "not/without/instead of" before it: skipped.
_WITH_COLOUR = re.compile(r"\b([a-z]+) (?:with|in|wearing) (?:a |an |some )?(" + _COLOUR + r" [a-z]+)\b")
_NEGATED = re.compile(r"\b(?:not|no|without|instead of|rather than)\s+(?:a |an |the )?$")


def keep_details(instruction: str, features: list[Feature], changed: list[str]) -> tuple[list[Feature], list[str]]:
    """Put back a colour the person said that the model dropped (live and eval, 2026-11-08: "a samurai with
    red armor riding a dragon" came back as "a samurai sitting on the dragon's back").

    Only "<thing> with/in/wearing <colour> <noun>", only onto the feature that names the thing, and only when
    no feature already has that colour. Model output is never trusted to keep every word; this is the cheap
    deterministic net for the most visible kind of loss.
    """
    said = instruction.lower()
    out, touched = list(features), list(changed)
    for match in _WITH_COLOUR.finditer(said):
        owner, phrase = match.group(1), match.group(2)
        colour = phrase.split()[0]
        if _NEGATED.search(said[:match.start(2)]) or any(re.search(rf"\b{colour}\b", f.text) for f in out):
            continue
        index = next((i for i, f in enumerate(out) if re.search(rf"\b{owner}s?\b", f.text.lower())), None)
        if index is None:
            continue
        f = out[index]
        out[index] = Feature(f.id, f.kind, _clean(f"{f.text}, in {phrase}"))
        if f.id not in touched:
            touched.append(f.id)
    return out, touched


def render(features: list[Feature], aspect: str = "square", steer: list[str] | tuple = ()):
    """The prompt: main subject first, then what's placed relative to it, where, and the look.

    ``steer``: policy words (kid-safe) added to the prompt but never shown as features.
    """
    from homeai.art.prompt_spec import Rendered, _cap
    ordered = sorted(features, key=lambda f: (ORDER[f.kind], f.id))
    sentences = [_cap(f.text) + "." for f in ordered if f.kind in ("subject", "element", "setting",
                                                                   "composition", "camera")]
    look = [f.text for f in ordered if f.kind in ("style", "colour", "lighting")]
    look += [w for w in steer if w.lower() not in " ".join(sentences + look).lower()]
    if look:
        sentences.append(_cap(", ".join(look)) + ".")
    width, height = SIZES.get(aspect, SIZES["square"])
    return Rendered(" ".join(sentences), width, height)


def describe(changed: list[str], before: list[Feature], after: list[Feature]) -> str:
    """Spoken: "Changed A, added C and removed D." Short on purpose; the window shows the words."""
    was = {f.id for f in before}
    now = {f.id for f in after}
    added = [c for c in changed if c in now and c not in was]
    removed = [c for c in changed if c in was and c not in now]
    edited = [c for c in dict.fromkeys(changed) if c in was and c in now]
    parts = [f"{verb} {_and(letters)}" for verb, letters in (("changed", edited), ("added", added),
                                                               ("removed", removed)) if letters]
    if not parts:
        return "Nothing changed."
    text = ", ".join(parts[:-1]) + (" and " if len(parts) > 1 else "") + parts[-1]
    return text[0].upper() + text[1:] + "."


def _and(items: list[str]) -> str:
    items = list(dict.fromkeys(items))
    return items[0] if len(items) == 1 else ", ".join(items[:-1]) + " and " + items[-1]


def from_spec(spec: dict) -> list[Feature]:
    """Older pictures (PromptSpec fields) as features, so a resumed project keeps working."""
    pairs = [("subject", spec.get("subject")), ("element", spec.get("details")), ("setting", spec.get("setting")),
             ("style", spec.get("style")), ("colour", spec.get("palette")), ("lighting", spec.get("lighting")),
             ("camera", spec.get("camera")), ("composition", spec.get("composition"))]
    out = []
    for kind, text in pairs:
        if text and str(text).strip():
            out.append(Feature(LETTERS[len(out)], kind, _clean(text)))
    return out


def features_of(spec: dict) -> list[Feature]:
    if "features" in spec:
        return load(spec["features"])
    return from_spec(spec)


def text_of(spec: dict) -> str:
    """All the words in a spec (for content checks), whichever shape it has."""
    return " ".join(f.text for f in features_of(spec)) if spec else ""


# Letter commands that need no model: "remove C", "delete feature C", "get rid of B".
REMOVE = re.compile(r"^(?:remove|delete|drop|get rid of|take out|lose)(?: feature)? ([a-l])(?: and ([a-l]))?$")


def letter_command(said: str) -> list[dict] | None:
    match = REMOVE.match(said.strip().lower())
    if not match:
        return None
    return [{"op": "remove", "id": g.upper()} for g in match.groups() if g]
