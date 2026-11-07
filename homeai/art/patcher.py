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
from dataclasses import dataclass

from homeai.art.prompt_spec import FIELDS, PATCH_SCHEMA, TEXT_FIELDS, SpecError, apply_patch

log = logging.getLogger("homeai.art.patcher")

OLLAMA = "http://127.0.0.1:11434"
MODEL = "llama31-voice"
TIMEOUT_S = 30.0
MSG_NOT_CAUGHT = "I didn't catch that change, so the picture stays as it was."

SYSTEM = """You edit the description of a picture. You get the current description as JSON and an instruction.
Answer with only the fields that should change, as JSON. Use null to remove a field. Leave out every field that stays.
Fields: subject (what the picture is of), details, setting (place or time), style (art style), palette (colours),
lighting, camera (angle or distance), composition, negative (things to leave out), aspect (square, portrait, landscape
or wide), references (names of saved characters or styles).
If the description is empty, the instruction asks for a new picture: fill in subject and anything else it names.
Keep the subject unless the instruction changes what the picture is of. Never invent details the instruction doesn't ask for."""


@dataclass
class PatchResult:
    spec: dict
    patch: dict
    ok: bool
    message: str = ""
    seconds: float = 0.0
    detail: str = ""


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
        patch = clip_long_text(patch)
        new = apply_patch(spec, patch)
    except (urllib.error.URLError, OSError, TimeoutError, KeyError, ValueError) as exc:
        detail = f"{type(exc).__name__}: {exc}; model said {raw[:200]!r}"[:400]
        log.warning("patch for %r not caught: %s", instruction, detail)
        return PatchResult(spec, {}, False, MSG_NOT_CAUGHT, round(clock() - start, 2), detail)
    return PatchResult(new, patch, True, "", round(clock() - start, 2))
