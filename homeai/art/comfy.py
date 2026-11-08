"""ComfyUI client for art mode (AVATAR_PLAN §7.1, 6a step 4).

Submits an API-format graph, waits for it by polling ``/history`` (standard
library only; ComfyUI's websocket only adds step percentages, which nothing
shows yet), fetches the finished image's path, cancels, and turns every failure
into one short sentence Jarvis can say. Promoted from ``tools/art/comfy_common.py``,
whose ``Comfy.run`` worked through the whole bake-off.
"""
from __future__ import annotations

import json
import time
import urllib.error
import urllib.request

from homeai.httpjson import post_json
from dataclasses import dataclass
from pathlib import Path

URL = "http://127.0.0.1:8188"
OUTPUT_DIR = Path.home() / ".local/share/homeai/comfyui/app/output"
DEFAULT_TIMEOUT_S = 600.0
POLL_S = 0.2

# Spoken sentences. Kept short: they are read aloud.
MSG_NOT_RUNNING = "The drawing engine isn't running. Switch to art mode first."
MSG_OUT_OF_MEMORY = "I ran out of graphics memory on that one. Try a smaller picture."
MSG_MISSING_MODEL = "A model that picture needs isn't installed."
MSG_BAD_GRAPH = "That drawing setup isn't valid."
MSG_TIMED_OUT = "That picture took too long, so I stopped it."
MSG_NO_IMAGE = "The drawing finished but made no picture."
MSG_FAILED = "The drawing failed."


class ComfyError(Exception):
    """A failure with a sentence fit to speak in ``spoken`` and the detail for logs in ``detail``."""

    def __init__(self, spoken: str, detail: str = "") -> None:
        super().__init__(f"{spoken} ({detail})" if detail else spoken)
        self.spoken, self.detail = spoken, detail[:500]


@dataclass
class Result:
    prompt_id: str
    image: Path
    seconds: float


_http = post_json


def dangling_links(graph: dict) -> list[str]:
    """Inputs that link to a node the graph doesn't have, as "node.input -> target".

    ComfyUI 0.44 accepts such a graph and then its job thread dies with NodeNotFoundError: the server keeps
    answering, but never runs another job (found by tools/art/comfy_client_live.py, 2026-11-07).
    """
    found = []
    for node_id, node in graph.items():
        inputs = node.get("inputs", {}) if isinstance(node, dict) else {}
        for name, value in inputs.items():
            if isinstance(value, list) and len(value) == 2 and isinstance(value[1], int) \
                    and isinstance(value[0], str) and value[0] not in graph:
                found.append(f"{node_id}.{name} -> {value[0]}")
    return found


def classify(text: str) -> str:
    """The spoken sentence for a ComfyUI error message."""
    low = text.lower()
    if "out of memory" in low or "outofmemory" in low:
        return MSG_OUT_OF_MEMORY
    if "value not in list" in low or ("not found" in low and ("ckpt" in low or "model" in low or "unet" in low)):
        return MSG_MISSING_MODEL
    return MSG_FAILED


class ComfyClient:
    def __init__(self, url: str = URL, output_dir: Path = OUTPUT_DIR, http=_http, clock=time.monotonic,
                 sleep=time.sleep) -> None:
        self.url, self.output_dir = url.rstrip("/"), output_dir
        self._http, self._clock, self._sleep = http, clock, sleep

    def _call(self, path: str, payload: dict | None = None, timeout: float = 30.0) -> dict:
        try:
            return self._http(f"{self.url}{path}", payload, timeout)
        except urllib.error.HTTPError as exc:
            body = exc.read().decode("utf-8", "replace")
            raise ComfyError(classify(body) if exc.code != 400 else _rejected(body), body) from exc
        except (urllib.error.URLError, ConnectionError, TimeoutError, OSError) as exc:
            raise ComfyError(MSG_NOT_RUNNING, str(exc)) from exc
        except ValueError as exc:
            raise ComfyError(MSG_FAILED, f"bad JSON from ComfyUI: {exc}") from exc

    def healthy(self) -> bool:
        try:
            self._call("/system_stats", timeout=3)
            return True
        except ComfyError:
            return False

    def submit(self, graph: dict) -> str:
        broken = dangling_links(graph)
        if broken:
            raise ComfyError(MSG_BAD_GRAPH, "links to missing nodes: " + ", ".join(broken))
        reply = self._call("/prompt", {"prompt": graph})
        prompt_id = reply.get("prompt_id") if isinstance(reply, dict) else None
        if not prompt_id:
            raise ComfyError(MSG_FAILED, f"no prompt_id in {str(reply)[:200]}")
        return prompt_id

    def poll(self, prompt_id: str) -> Path | None:
        """The image once done, None while still working; raises ComfyError if it failed."""
        entry = self._call(f"/history/{prompt_id}").get(prompt_id)
        if not entry:
            return None
        status = entry.get("status", {})
        if status.get("status_str") == "error":
            messages = " ".join(m[1].get("exception_message", "") for m in status.get("messages", [])
                                if m and m[0] == "execution_error")
            raise ComfyError(classify(messages), messages)
        if not status.get("completed"):
            return None
        images = [i for node in entry.get("outputs", {}).values() for i in node.get("images", [])
                  if i.get("type", "output") == "output"]
        if not images:
            raise ComfyError(MSG_NO_IMAGE, prompt_id)
        return self.output_dir / images[0].get("subfolder", "") / images[0]["filename"]

    def wait(self, prompt_id: str, timeout: float = DEFAULT_TIMEOUT_S) -> Result:
        start = self._clock()
        while True:
            image = self.poll(prompt_id)
            if image is not None:
                return Result(prompt_id, image, round(self._clock() - start, 2))
            if self._clock() - start >= timeout:
                self.cancel(prompt_id)
                raise ComfyError(MSG_TIMED_OUT, f"{prompt_id} after {timeout:.0f}s")
            self._sleep(POLL_S)

    def run(self, graph: dict, timeout: float = DEFAULT_TIMEOUT_S) -> Result:
        return self.wait(self.submit(graph), timeout)

    def cancel(self, prompt_id: str) -> None:
        """Drop it from the queue if waiting, interrupt it if running. Best effort: never raises."""
        for path, payload in (("/queue", {"delete": [prompt_id]}), ("/interrupt", {"prompt_id": prompt_id})):
            try:
                self._call(path, payload, timeout=5)
            except ComfyError:
                pass

    def free(self) -> None:
        """Unload models and free memory, e.g. before Jarvis needs the card back."""
        self._call("/free", {"unload_models": True, "free_memory": True})


def _rejected(body: str) -> str:
    """ComfyUI answers 400 when it refuses a graph before running it; a missing model shows up here."""
    try:
        errors = json.loads(body)
    except ValueError:
        return MSG_BAD_GRAPH
    text = json.dumps(errors)
    return MSG_MISSING_MODEL if classify(text) == MSG_MISSING_MODEL else MSG_BAD_GRAPH
