"""Turn a spoken instruction into a PromptSpec patch with the voice model (AVATAR_PLAN §7.2, 6a step 6).

Calls Ollama directly, not through the ZeroClaw agent: no tools exist on this
path, so nothing in the house can act, and the answer is constrained by a JSON
schema (Ollama structured outputs) to the spec's own fields. ``keep_alive`` is
-1 because the voice model is pinned; any other value would unload Jarvis.
If the answer can't be used, the previous spec stands and Jarvis says so.
"""
from __future__ import annotations

import json
import logging
import time
import urllib.error
import urllib.request
from dataclasses import dataclass, field

from homeai.art.prompt_spec import FIELDS, PATCH_SCHEMA, SIZES, TEXT_FIELDS, SpecError, apply_patch

log = logging.getLogger("homeai.art.patcher")

OLLAMA = "http://127.0.0.1:11434"
MODEL = "llama31-voice"
TIMEOUT_S = 30.0
MSG_NOT_CAUGHT = "I didn't catch that change, so the picture stays as it was."

SYSTEM = """You edit the description of a picture. You get the current description as JSON and an instruction.
Answer with only the fields that should change, as JSON. Use null to remove a field. Leave out every field that stays.
Fields: subject (what the picture is of), details, setting (place or time), style (art style), palette (colours),
lighting, camera (angle or distance), composition, negative (things to leave out), aspect (square, portrait, landscape
or wide), references (names of saved characters the person mentions by name, like "Mira"; never descriptions).
If the description is empty, the instruction asks for a new picture: fill in subject, and only the other fields the
instruction names. Don't add a style, palette, lighting, camera, composition or setting nobody asked for.
Keep the subject unless the instruction changes what the picture is of. Never invent details the instruction doesn't ask for.
If the instruction complains about something that went wrong (an extra head, a second rider, something odd in a hand),
fix it: write what should be there instead in details ("a single head", "only one rider", "empty hands"). Never add
the unwanted thing, and never repeat the complaint as a feature."""


@dataclass
class PatchResult:
    spec: dict
    patch: dict
    ok: bool
    message: str = ""
    seconds: float = 0.0
    detail: str = ""
    changed: list = field(default_factory=list)     # feature letters touched (feature specs only)


FEATURE_SYSTEM = """You keep the list of features that describe a picture. You get the current features
(letter, kind, text) and what the person said. Answer with operations on the list:
- change: rewrite one existing feature. Give its letter and the full new text. Use change whenever the person adjusts
  something already in the picture: "make the dragon red" changes the dragon's feature to "a single red dragon ...".
  Never add a second feature for something that already has one.
- add: something new, with its kind.
- remove: take a feature out, by letter.
If the person names a letter ("change B to a knight", "remove C"), act on that letter.
Kinds: subject (the main thing: exactly one, usually the biggest), element (anything else, placed relative to the
subject: "one samurai sitting on the dragon's back"), setting (where and when), style, colour, lighting, camera,
composition.
Write each feature as a short plain phrase. State counts and body parts plainly ("a single head", "four legs",
"one rider"). Write what should be there, never what shouldn't. If the person complains that something came out
wrong (two heads, an extra rider, a leg in the wrong place), change that feature to say the right thing plainly.
For a new picture (no features yet), make one subject feature and add only what the person described; don't invent
a style, colours, lighting or camera they didn't ask for.
If the person rambles or corrects themselves ("a dragon, no wait, a wyvern, um, green"), use what they ended up meaning.
Each thing is described in exactly one feature: never mention an element's thing inside another feature (the rider
belongs in its own feature, not also in the dragon's).
Sizes: when the person says something is too small or too big, write its size relative to the subject in its own
feature ("a huge steam-powered battleship, many times bigger than the dragon").
Ignore words that aren't about the picture (exclamations, questions or remarks to other people in the room). If
nothing said is about the picture, answer with no operations.
Set aspect only if they ask for a shape (square, portrait, landscape, wide)."""


def feature_messages(spec: dict, instruction: str) -> list[dict]:
    from homeai.art import features as F
    current = [f"{f.id} ({f.kind}): {f.text}" for f in F.features_of(spec)] if spec else []
    listing = "\n".join(current) if current else "(none yet: this is a new picture)"
    return [{"role": "system", "content": FEATURE_SYSTEM},
            {"role": "user", "content": f"Current features:\n{listing}\nThe person said: {instruction}"}]


def features_patch_for(spec: dict, instruction: str, chat=None, clock=time.monotonic) -> PatchResult:
    """Operations on the lettered features for ``instruction``; on any failure, ``spec`` unchanged, ``ok`` False."""
    from homeai.art import features as F
    chat = chat or ollama_chat
    start = clock()
    raw = ""
    try:
        before = F.features_of(spec) if spec else []
        ops = F.letter_command(instruction)
        aspect = None
        if ops is None:
            raw = chat(feature_messages(spec, instruction), F.OPS_SCHEMA)
            answer = json.loads(raw)
            if not isinstance(answer, dict) or not isinstance(answer.get("ops"), list):
                raise SpecError("no operations")
            ops, aspect = answer["ops"], answer.get("aspect")
        if not ops and not aspect:
            raise SpecError("empty change")
        used = set(spec.get("letters_used") or []) if spec else set()
        after, changed = F.apply_ops(before, ops, used)
        new = {"features": F.dump(after), "aspect": aspect if aspect in SIZES else (spec or {}).get("aspect", "square"),
               "letters_used": sorted(used | {f.id for f in before} | {f.id for f in after}),
               "seed": (spec or {}).get("seed")}
    except (urllib.error.URLError, OSError, TimeoutError, KeyError, ValueError) as exc:
        detail = f"{type(exc).__name__}: {exc}; model said {raw[:200]!r}"[:400]
        log.warning("features for %r not caught: %s", instruction, detail)
        return PatchResult(spec, {}, False, MSG_NOT_CAUGHT, round(clock() - start, 2), detail)
    return PatchResult(new, {"ops": ops, "aspect": aspect}, True, "", round(clock() - start, 2), changed=changed)


def ollama_chat(messages: list[dict], schema: dict, timeout: float = TIMEOUT_S) -> str:
    payload = {"model": MODEL, "messages": messages, "format": schema, "stream": False, "keep_alive": -1,
               "options": {"temperature": 0}}
    req = urllib.request.Request(f"{OLLAMA}/api/chat", data=json.dumps(payload).encode(),
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as reply:
        return json.loads(reply.read())["message"]["content"]


def messages_for(spec: dict, instruction: str) -> list[dict]:
    current = {k: v for k, v in spec.items() if k in FIELDS and v not in (None, "", [])
               and k not in ("seed", "strength")}
    return [{"role": "system", "content": SYSTEM},
            {"role": "user", "content": f"Current description: {json.dumps(current)}\nInstruction: {instruction}"}]


def _names_in(instruction: str, names) -> bool:
    """Whether every reference the model gave is a name the person actually said."""
    if not isinstance(names, list) or not names:
        return True
    said = instruction.lower()
    return all(isinstance(n, str) and n.lower() in said for n in names)


def clip_long_text(patch: dict) -> dict:
    """Shorten over-long text fields at a word boundary instead of refusing the whole change.

    Ollama's structured output doesn't enforce maxLength, and the model writes
    long lighting and palette descriptions; one 90-character field used to throw
    away a good change ("I didn't catch that change", live 2026-11-07, suspected).
    """
    out = dict(patch)
    for name, limit in TEXT_FIELDS.items():
        value = out.get(name)
        if isinstance(value, str) and len(value) > limit:
            cut = value[:limit].rsplit(" ", 1)[0].rstrip(" ,;:")
            out[name] = cut or value[:limit]
    return out


def patch_for(spec: dict, instruction: str, chat=ollama_chat, clock=time.monotonic) -> PatchResult:
    """The new spec for ``instruction``; on any failure, ``spec`` unchanged with ``ok`` False."""
    start = clock()
    raw = ""
    try:
        raw = chat(messages_for(spec, instruction), PATCH_SCHEMA)
        patch = json.loads(raw)
        if isinstance(patch, dict) and spec.get("subject") and "subject" in patch and not patch["subject"]:
            del patch["subject"]        # a picture always keeps a subject; the model sometimes nulls it in passing
        if not isinstance(patch, dict) or not patch:
            raise SpecError("empty patch")
        if "seed" in patch or "strength" in patch:
            raise SpecError("the model may not set seed or strength")
        patch = {k: v for k, v in clip_long_text(patch).items()
                 if not (isinstance(v, str) and v.strip().lower() in ("none", "n/a", "nothing"))}
        if "references" in patch and not _names_in(instruction, patch["references"]):
            del patch["references"]     # the model files descriptions under references; only names belong there
        new = apply_patch(spec, patch)
    except (urllib.error.URLError, OSError, TimeoutError, KeyError, ValueError) as exc:
        detail = f"{type(exc).__name__}: {exc}; model said {raw[:200]!r}"[:400]
        log.warning("patch for %r not caught: %s", instruction, detail)
        return PatchResult(spec, {}, False, MSG_NOT_CAUGHT, round(clock() - start, 2), detail)
    return PatchResult(new, patch, True, "", round(clock() - start, 2))
