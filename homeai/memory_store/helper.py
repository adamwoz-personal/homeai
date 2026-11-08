# Copyright (c) 2026 Adam Wosotowsky
# SPDX-License-Identifier: BUSL-1.1
# Licensed under the Business Source License 1.1; see LICENSE at the
# repository root. Converts to Apache-2.0 on 2030-09-15.
"""A small model for "fill this JSON schema" jobs (plan §6.1).

``Helper.structured(task, schema, payload)`` returns a dict that matches
``schema`` or raises. Ollama's ``format`` constrains decoding to the schema,
so malformed JSON should be impossible; the result is still validated here,
because a model can obey the grammar and break the limits (and tests fake
the model). Invalid output gets one retry, then ``HelperInvalid``.

The default runs on the CPU (``num_gpu=0``): the GPU belongs to the voice
model and whisper.
"""

from __future__ import annotations

import json
import logging
import urllib.error
import urllib.request
from typing import Any, Callable

from homeai.httpjson import post_json

log = logging.getLogger(__name__)


class HelperError(RuntimeError):
    """Base class."""


class HelperUnavailable(HelperError):
    """The model could not be reached or crashed; try again later."""


class HelperInvalid(HelperError):
    """The model answered, twice, with something that breaks the schema."""


_TYPES = {"object": dict, "array": list, "string": str, "boolean": bool,
          "integer": int, "number": (int, float)}


def validate(schema: dict[str, Any], value: Any, path: str = "$") -> list[str]:
    """Errors for the JSON-schema subset the helper schemas use. [] if valid.

    Supported: type, enum, properties, required, additionalProperties=false,
    items, minItems, maxItems, minLength, maxLength, minimum, maximum, and
    "type": [..., "null"].
    """
    types = schema.get("type")
    if isinstance(types, list):
        if value is None and "null" in types:
            return []
        types = next(t for t in types if t != "null")
    if types:
        want = _TYPES[types]
        if not isinstance(value, want) or (isinstance(value, bool) and types != "boolean"):
            return [f"{path}: expected {types}"]
    errs: list[str] = []
    if "enum" in schema and value not in schema["enum"]:
        errs.append(f"{path}: {value!r} not in {schema['enum']}")
    if isinstance(value, str):
        if len(value) < schema.get("minLength", 0):
            errs.append(f"{path}: shorter than {schema['minLength']}")
        if len(value) > schema.get("maxLength", 1 << 30):
            errs.append(f"{path}: longer than {schema['maxLength']}")
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        if value < schema.get("minimum", float("-inf")):
            errs.append(f"{path}: below {schema['minimum']}")
        if value > schema.get("maximum", float("inf")):
            errs.append(f"{path}: above {schema['maximum']}")
    if isinstance(value, list):
        if len(value) < schema.get("minItems", 0):
            errs.append(f"{path}: fewer than {schema['minItems']} items")
        if len(value) > schema.get("maxItems", 1 << 30):
            errs.append(f"{path}: more than {schema['maxItems']} items")
        for i, item in enumerate(value):
            errs += validate(schema.get("items", {}), item, f"{path}[{i}]")
    if isinstance(value, dict):
        props = schema.get("properties", {})
        errs += [f"{path}: missing {k}" for k in schema.get("required", []) if k not in value]
        if schema.get("additionalProperties") is False:
            errs += [f"{path}: unexpected {k}" for k in value if k not in props]
        for k, sub in props.items():
            if k in value:
                errs += validate(sub, value[k], f"{path}.{k}")
    return errs


class Helper:
    """Retry-and-validate around ``_complete``, which subclasses provide."""

    name = "helper"

    def _complete(self, task: str, schema: dict, payload: dict) -> str:
        raise NotImplementedError

    def release(self, restore: str | None = None) -> bool:
        """Done for now: give the model's memory back. True if nothing failed."""
        return True

    def structured(self, task: str, schema: dict, payload: dict) -> dict:
        problem = ""
        for attempt in (1, 2):
            raw = self._complete(task, schema, payload)
            try:
                value = json.loads(raw)
            except (TypeError, ValueError):
                problem = "not JSON"
            else:
                errs = validate(schema, value)
                if not errs:
                    return value
                problem = "; ".join(errs[:3])
            log.info("helper output invalid (attempt %d): %s", attempt, problem)
        raise HelperInvalid(problem)


class OllamaHelper(Helper):
    def __init__(self, url: str, model: str, *, threads: int = 6, timeout_s: float = 300.0,
                 ctx: int = 8192, keep_alive: str = "5m",
                 post: Callable[[str, dict, float], dict] | None = None) -> None:
        self.url = url.rstrip("/")
        self.model = self.name = model
        self.threads, self.timeout_s, self.ctx, self.keep_alive = threads, timeout_s, ctx, keep_alive
        self._post = post or _post_json

    def request(self, task: str, schema: dict, payload: dict) -> dict:
        return {"model": self.model, "stream": False, "format": schema,
                "keep_alive": self.keep_alive,
                "messages": [{"role": "system", "content": task},
                             {"role": "user", "content": json.dumps(payload, ensure_ascii=False)}],
                "options": {"temperature": 0, "num_gpu": 0, "num_thread": self.threads,
                            "num_ctx": self.ctx}}

    def _complete(self, task: str, schema: dict, payload: dict) -> str:
        try:
            reply = self._post(f"{self.url}/api/chat", self.request(task, schema, payload),
                               self.timeout_s)
        except Exception as exc:  # noqa: BLE001 - network, HTTP, timeouts: all "try later"
            raise HelperUnavailable(f"{self.model}: {exc}") from exc
        if "error" in reply:
            raise HelperUnavailable(f"{self.model}: {reply['error']}")
        return (reply.get("message") or {}).get("content", "")

    def release(self, restore: str | None = None) -> bool:
        """Unload the helper, then warm ``restore`` (the voice model) and pin it.

        Ollama keeps one model loaded, so the helper evicted the voice model and
        the next spoken question paid a multi-second cold load (2026-11-04).
        Never raises: a failure only costs that reload later.
        """
        steps = [{"model": self.model, "keep_alive": 0}]
        if restore:
            steps.append({"model": restore, "keep_alive": -1})
        for body in steps:
            try:
                reply = self._post(f"{self.url}/api/generate", body, self.timeout_s)
            except Exception as exc:  # noqa: BLE001
                reply = {"error": str(exc)}
            if "error" in reply:
                log.warning("helper release: %s: %s", body["model"], reply["error"])
                return False
        return True


def _post_json(url: str, body: dict, timeout: float) -> dict:
    try:
        return post_json(url, body, timeout)
    except urllib.error.HTTPError as exc:
        return {"error": f"HTTP {exc.code}: {exc.read()[:200].decode(errors='replace')}"}


class ScriptedHelper(Helper):
    """Test double: replies in order (str, dict, or an exception to raise)."""

    name = "scripted"

    def __init__(self, replies: list | None = None,
                 respond: Callable[[str, dict, dict], Any] | None = None) -> None:
        self.replies = list(replies or [])
        self.respond = respond
        self.calls: list[tuple[str, dict]] = []

    def _complete(self, task: str, schema: dict, payload: dict) -> str:
        self.calls.append((task, payload))
        r = self.respond(task, schema, payload) if self.respond else self.replies.pop(0)
        if isinstance(r, Exception):
            raise r
        return r if isinstance(r, str) else json.dumps(r)
