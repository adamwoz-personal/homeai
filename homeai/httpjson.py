"""JSON over HTTP to the local services (Ollama, ComfyUI), in one place (CODE_REVIEW_PLAN item 1).

Five modules had their own copy of "POST JSON, read JSON" and six defined
Ollama's address. Errors are not caught here: each caller keeps its own
handling, because they differ on purpose (a failed picture patch says "I didn't
catch that"; a failed picture check shows the picture unchecked).
"""
from __future__ import annotations

import json
import urllib.request

OLLAMA = "http://127.0.0.1:11434"


def post_json(url: str, payload: dict | None = None, timeout: float = 30.0) -> dict:
    """POST ``payload`` as JSON (GET when None) and return the JSON reply ({} for an empty body)."""
    data = json.dumps(payload).encode() if payload is not None else None
    req = urllib.request.Request(url, data=data, headers={"Content-Type": "application/json"} if data else {})
    with urllib.request.urlopen(req, timeout=timeout) as reply:
        body = reply.read()
    return json.loads(body) if body else {}


def ollama_chat(model: str, messages: list[dict], *, fmt: dict | None = None, keep_alive: int | str = -1,
                options: dict | None = None, timeout: float = 30.0) -> str:
    """One non-streamed /api/chat answer's text. ``keep_alive`` -1 pins the model (the voice model must stay)."""
    payload: dict = {"model": model, "messages": messages, "stream": False, "keep_alive": keep_alive}
    if fmt is not None:
        payload["format"] = fmt
    if options is not None:
        payload["options"] = options
    return post_json(f"{OLLAMA}/api/chat", payload, timeout)["message"]["content"]
