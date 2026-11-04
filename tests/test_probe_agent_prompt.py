"""probe_agent_prompt must never let a model tool call reach ZeroClaw.

The probe's temp agent has the real MCP tools, so a tool call in a forwarded
reply runs on the house (2026-11-04: probing "say hello" played jazz on the
kitchen Echo). These tests drive the real recording proxy over HTTP against a
fake model that always answers with a tool call.
"""

import http.server
import importlib.util
import json
import threading
import urllib.request
from pathlib import Path

import pytest

_PATH = Path(__file__).resolve().parents[1] / "tools" / "probe_agent_prompt.py"
_spec = importlib.util.spec_from_file_location("probe_agent_prompt", _PATH)
probe = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(probe)

TOOLS = [{"type": "function", "function": {"name": "homeai__music"}}]


def _tool_call_reply(content=""):
    return {"choices": [{"index": 0, "finish_reason": "tool_calls", "message": {
        "role": "assistant", "content": content,
        "tool_calls": [{"id": "1", "type": "function", "function": {
            "name": "homeai__music", "arguments": '{"action":"play","request":"jazz"}'}}]}}]}


class _FakeModel(http.server.BaseHTTPRequestHandler):
    hits = 0
    reply: dict = {}

    def log_message(self, *args):
        pass

    def do_POST(self):
        self.rfile.read(int(self.headers.get("Content-Length", 0)))
        type(self).hits += 1
        data = json.dumps(self.reply).encode()
        self.send_response(200)
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)


def _serve(handler):
    server = http.server.ThreadingHTTPServer(("127.0.0.1", 0), handler)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    return server


@pytest.fixture
def proxy(monkeypatch):
    _FakeModel.hits = 0
    _FakeModel.reply = _tool_call_reply()
    model = _serve(_FakeModel)
    monkeypatch.setattr(probe._Recorder, "upstream", f"http://127.0.0.1:{model.server_port}")
    monkeypatch.setattr(probe._Recorder, "captured", [])
    monkeypatch.setattr(probe._Recorder, "suppressed", 0)
    rec = _serve(probe._Recorder)
    yield rec
    rec.shutdown()
    model.shutdown()


def _chat(server, stream=False):
    body = json.dumps({"model": "m", "stream": stream, "tools": TOOLS,
                       "messages": [{"role": "user", "content": "say hello"}]}).encode()
    req = urllib.request.Request(f"http://127.0.0.1:{server.server_port}/v1/chat/completions",
                                 data=body, method="POST")
    with urllib.request.urlopen(req, timeout=10) as resp:
        return resp.read().decode()


def test_default_never_calls_the_model(proxy, monkeypatch):
    monkeypatch.setattr(probe._Recorder, "real_reply", False)
    out = json.loads(_chat(proxy))
    assert _FakeModel.hits == 0
    assert out["choices"][0]["message"] == {"role": "assistant", "content": probe.CANNED}
    assert probe._Recorder.captured[0]["body"]["tools"] == TOOLS


@pytest.mark.parametrize("stream", [False, True])
def test_real_reply_strips_tool_calls(proxy, monkeypatch, stream):
    monkeypatch.setattr(probe._Recorder, "real_reply", True)
    out = _chat(proxy, stream=stream)
    assert _FakeModel.hits == 1
    assert "tool_calls" not in out and "homeai__music" not in out
    assert probe.SUPPRESSED in out
    assert probe._Recorder.suppressed == 1


def test_text_tool_call_is_suppressed_too():
    reply = {"choices": [{"message": {"role": "assistant",
                                      "content": '{"name": "homeai__music", "arguments": {}}'}}]}
    out, n = probe.strip_tool_calls(reply, ["homeai__music"])
    assert n == 1 and out["choices"][0]["message"]["content"] == probe.SUPPRESSED


def test_plain_words_pass_through_untouched():
    reply = {"choices": [{"message": {"role": "assistant", "content": "Hello!"}}]}
    out, n = probe.strip_tool_calls(reply, ["homeai__music"])
    assert n == 0 and out["choices"][0]["message"]["content"] == "Hello!"
