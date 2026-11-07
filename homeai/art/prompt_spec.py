"""Structured prompt spec (AVATAR_PLAN §7.2, 6a step 6).

A picture's prompt is a small record, not free text. The voice model never
rewrites a prompt: it returns a patch (the fields to change, ``null`` to clear
one), which is checked here before it is applied. A bad patch changes nothing.
That keeps prompts coherent over many tweaks, makes "what changed?" a diff, and
gives content policy named fields to inspect. Standard library only.
"""
from __future__ import annotations

from dataclasses import dataclass

TEXT_FIELDS = {
    "subject": 200, "details": 300, "setting": 200, "style": 80, "palette": 80,
    "lighting": 80, "camera": 80, "composition": 120, "negative": 200,
}
ASPECTS = ("square", "portrait", "landscape", "wide")
SIZES = {"square": (1024, 1024), "portrait": (832, 1216), "landscape": (1216, 832), "wide": (1344, 768)}
MAX_REFERENCES = 4
FIELDS = (*TEXT_FIELDS, "references", "aspect", "seed", "strength")
DEFAULTS = {"style": "", "aspect": "square", "references": [], "seed": None, "strength": None}

# The JSON schema handed to the model (Ollama structured outputs), so it can only answer with these fields.
PATCH_SCHEMA = {
    "type": "object",
    "properties": {
        **{name: {"type": ["string", "null"], "maxLength": limit} for name, limit in TEXT_FIELDS.items()},
        "aspect": {"type": ["string", "null"], "enum": [*ASPECTS, None]},
        "references": {"type": ["array", "null"], "items": {"type": "string", "maxLength": 60},
                       "maxItems": MAX_REFERENCES},
    },
    "additionalProperties": False,
}


class SpecError(ValueError):
    pass


def _clean_text(name: str, value) -> str:
    if not isinstance(value, str):
        raise SpecError(f"{name} must be text")
    text = " ".join(value.split())
    if len(text) > TEXT_FIELDS[name]:
        raise SpecError(f"{name} is over {TEXT_FIELDS[name]} characters")
    return text


def validate(spec: dict) -> dict:
    """A clean copy of ``spec``, or SpecError. Unknown fields are refused, not dropped."""
    if not isinstance(spec, dict):
        raise SpecError("a spec is a JSON object")
    unknown = sorted(set(spec) - set(FIELDS))
    if unknown:
        raise SpecError(f"unknown field {unknown[0]}")
    out = {**DEFAULTS}
    for name, value in spec.items():
        if name in TEXT_FIELDS:
            out[name] = _clean_text(name, value) if value is not None else ""
        elif name == "aspect":
            if value not in ASPECTS:
                raise SpecError(f"aspect must be one of {', '.join(ASPECTS)}")
            out[name] = value
        elif name == "references":
            if not isinstance(value, list) or len(value) > MAX_REFERENCES \
                    or not all(isinstance(v, str) and 0 < len(v) <= 60 for v in value):
                raise SpecError(f"references are up to {MAX_REFERENCES} short names")
            out[name] = list(value)
        elif name == "seed":
            if value is not None and (not isinstance(value, int) or isinstance(value, bool) or not 0 <= value < 2**32):
                raise SpecError("seed is a whole number from 0")
            out[name] = value
        elif name == "strength":
            if value is not None and (not isinstance(value, (int, float)) or isinstance(value, bool)
                                      or not 0.0 <= value <= 1.0):
                raise SpecError("strength is from 0 to 1")
            out[name] = value
    if not out.get("subject"):
        raise SpecError("a picture needs a subject")
    return out


def apply_patch(spec: dict, patch: dict) -> dict:
    """``spec`` with ``patch`` applied (null clears a field to its default); SpecError leaves nothing changed."""
    if not isinstance(patch, dict):
        raise SpecError("a patch is a JSON object")
    merged = dict(spec)
    for name, value in patch.items():
        if name not in FIELDS:
            raise SpecError(f"unknown field {name}")
        if value is None:
            merged.pop(name, None)
        else:
            merged[name] = value
    return validate(merged)


def diff(before: dict, after: dict) -> dict:
    """Field -> (before, after) for every field that changed."""
    return {name: (before.get(name), after.get(name)) for name in FIELDS if before.get(name) != after.get(name)}


def describe(changes: dict) -> str:
    """A short spoken summary of ``diff`` output, e.g. "palette: blue; lighting: night"."""
    parts = []
    for name, (_, now) in changes.items():
        if name in ("seed", "strength"):
            continue
        shown = ", ".join(now) if isinstance(now, list) else now
        parts.append(f"{name}: {shown}" if shown else f"{name} cleared")
    return "; ".join(parts) or "nothing"


def _cap(text: str) -> str:
    """First letter up, the rest as written (str.capitalize would lower "Studio Ghibli")."""
    return text[:1].upper() + text[1:]


@dataclass
class Rendered:
    text: str
    width: int
    height: int


def render(spec: dict) -> Rendered:
    """The prompt text for the kept models, all of which read natural sentences (klein, Boogu, Qwen)."""
    spec = validate(spec)
    lead = spec["subject"]
    if spec.get("setting"):
        lead += f", {spec['setting']}"
    sentences = [_cap(lead) + "."]
    if spec.get("details"):
        sentences.append(_cap(spec["details"].rstrip(".")) + ".")
    look = [f"{spec['style']} style" if spec.get("style") else "",
            f"{spec['palette']} colours" if spec.get("palette") else "",
            f"{spec['lighting']} lighting" if spec.get("lighting") else ""]
    look = [part for part in look if part]
    if look:
        sentences.append(_cap(", ".join(look)) + ".")
    for name in ("camera", "composition"):
        if spec.get(name):
            sentences.append(_cap(spec[name].rstrip(".")) + ".")
    if spec.get("negative"):
        sentences.append(f"Avoid {spec['negative'].rstrip('.')}.")
    width, height = SIZES[spec["aspect"]]
    return Rendered(" ".join(sentences), width, height)
