"""homeai/httpjson.py: the one JSON-over-HTTP helper for Ollama and ComfyUI (CODE_REVIEW_PLAN item 1)."""
from __future__ import annotations

import json

import pytest

from homeai import httpjson as H


@pytest.fixture
def wire(monkeypatch):
    sent, replies = [], []

    class Reply:
        def __init__(self, body):
            self.body = body

        def __enter__(self):
            return self

        def __exit__(self, *a):
            return False

        def read(self):
            return self.body

    def urlopen(req, timeout):
        sent.append({"url": req.full_url, "method": req.get_method(), "timeout": timeout,
                     "body": json.loads(req.data) if req.data else None,
                     "type": req.headers.get("Content-type")})
        return Reply(replies.pop(0))
    monkeypatch.setattr(H.urllib.request, "urlopen", urlopen)
    return sent, replies


def test_post_sends_json_and_reads_json(wire):
    sent, replies = wire
    replies.append(b'{"ok": 1}')
    assert H.post_json("http://x/y", {"a": 1}, timeout=7) == {"ok": 1}
    assert sent == [{"url": "http://x/y", "method": "POST", "timeout": 7, "body": {"a": 1},
                     "type": "application/json"}]


def test_no_payload_is_a_get_and_an_empty_body_is_empty(wire):
    sent, replies = wire
    replies.append(b"")
    assert H.post_json("http://x/history") == {}
    assert sent[0]["method"] == "GET" and sent[0]["type"] is None and sent[0]["timeout"] == 30.0


def test_ollama_chat_pins_by_default_and_sends_only_what_is_given(wire):
    sent, replies = wire
    replies += [b'{"message": {"content": "hi"}}'] * 2
    assert H.ollama_chat("m", [{"role": "user", "content": "x"}]) == "hi"
    assert sent[0]["url"] == "http://127.0.0.1:11434/api/chat"
    assert sent[0]["body"] == {"model": "m", "messages": [{"role": "user", "content": "x"}], "stream": False,
                               "keep_alive": -1}
    H.ollama_chat("m", [], fmt={"type": "object"}, keep_alive="2m", options={"temperature": 0}, timeout=5)
    assert sent[1]["body"]["format"] == {"type": "object"} and sent[1]["body"]["keep_alive"] == "2m"
    assert sent[1]["body"]["options"] == {"temperature": 0} and sent[1]["timeout"] == 5


def test_the_checker_asks_with_the_picture_and_lets_the_model_go(wire, tmp_path):
    from homeai.art import checker as CK
    sent, replies = wire
    replies.append(b'{"message": {"content": " 2 \\n"}}')
    image = tmp_path / "x.png"
    image.write_bytes(b"png")
    assert CK.ollama_ask(image, "How many heads?") == "2"
    body = sent[0]["body"]
    assert body["model"] == "minicpm-v" and body["keep_alive"] == "2m" and body["messages"][0]["images"] == ["cG5n"]
