# Copyright (c) 2026 Adam Wosotowsky
# SPDX-License-Identifier: BUSL-1.1
# Licensed under the Business Source License 1.1; see LICENSE at the
# repository root. Converts to Apache-2.0 on 2030-09-15.
"""The coding-tools proxy to HA's MCP server. A fake HA over real HTTP checks
the auth header, status-code hints and notification handling; nothing here
may ever raise into the MCP client."""

from __future__ import annotations

import io
import json
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer

import pytest

from homeai import ha_mcp_proxy as proxy
from homeai.ha_client import HASettings

TOKEN = "coding-token"
REQUESTS: list[dict] = []


class FakeHAMcp(BaseHTTPRequestHandler):
    status = 200
    body: object = {"jsonrpc": "2.0", "id": 1, "result": {"tools": [{"name": "HassTurnOn"}]}}

    def log_message(self, *a):
        pass

    def do_POST(self):
        raw = self.rfile.read(int(self.headers["Content-Length"]))
        REQUESTS.append({"path": self.path, "auth": self.headers.get("Authorization"),
                         "accept": self.headers.get("Accept"), "body": json.loads(raw)})
        if self.headers.get("Authorization") != f"Bearer {TOKEN}":
            self.send_response(401); self.end_headers(); return
        if "id" not in json.loads(raw):
            self.send_response(202); self.end_headers(); return
        self.send_response(type(self).status)
        self.send_header("Content-Type", "application/json")
        self.end_headers()
        payload = type(self).body
        self.wfile.write(payload if isinstance(payload, bytes) else json.dumps(payload).encode())


@pytest.fixture
def server():
    REQUESTS.clear()
    FakeHAMcp.status = 200
    FakeHAMcp.body = {"jsonrpc": "2.0", "id": 1, "result": {"tools": [{"name": "HassTurnOn"}]}}
    httpd = HTTPServer(("127.0.0.1", 0), FakeHAMcp)
    threading.Thread(target=httpd.serve_forever, daemon=True).start()
    yield HASettings(url=f"http://127.0.0.1:{httpd.server_port}", token=TOKEN)
    httpd.shutdown()


def call(line, settings):
    return proxy.handle(json.dumps(line), settings)


def test_forwards_with_bearer_token(server):
    reply = call({"jsonrpc": "2.0", "id": 1, "method": "tools/list"}, server)
    assert reply["result"]["tools"][0]["name"] == "HassTurnOn"
    assert REQUESTS[0]["path"] == "/api/mcp"
    assert REQUESTS[0]["auth"] == f"Bearer {TOKEN}"
    assert "application/json" in REQUESTS[0]["accept"]


def test_notification_gets_no_reply(server):
    assert call({"jsonrpc": "2.0", "method": "notifications/initialized"}, server) is None
    assert REQUESTS[0]["body"]["method"] == "notifications/initialized"


@pytest.mark.parametrize("status,words", [(403, "Require admin"), (404, "Model Context Protocol"),
                                          (500, "HTTP 500")])
def test_http_errors_become_actionable_jsonrpc_errors(server, status, words):
    FakeHAMcp.status = status
    reply = call({"jsonrpc": "2.0", "id": 7, "method": "tools/list"}, server)
    assert reply["id"] == 7 and words in reply["error"]["message"]


def test_bad_token_is_explained(server):
    reply = call({"jsonrpc": "2.0", "id": 2, "method": "tools/list"},
                 HASettings(url=server.url, token="wrong"))
    assert "new long-lived token" in reply["error"]["message"]
    assert "wrong" not in json.dumps(reply)


def test_non_json_reply(server):
    FakeHAMcp.body = b"<html>nope</html>"
    reply = call({"jsonrpc": "2.0", "id": 3, "method": "tools/list"}, server)
    assert "not JSON" in reply["error"]["message"]


def test_unreachable_ha_is_an_error_not_a_crash():
    reply = call({"jsonrpc": "2.0", "id": 4, "method": "tools/list"},
                 HASettings(url="http://127.0.0.1:9", token=TOKEN))
    assert "Cannot reach Home Assistant" in reply["error"]["message"]
    assert call({"jsonrpc": "2.0", "method": "notifications/x"},
                HASettings(url="http://127.0.0.1:9", token=TOKEN)) is None


@pytest.mark.parametrize("line,code", [("not json", -32700), ("[1, 2]", -32600)])
def test_garbage_input(server, line, code):
    assert proxy.handle(line, server)["error"]["code"] == code


def test_unconfigured_still_speaks_mcp(tmp_path):
    out = io.StringIO()
    lines = "\n".join(json.dumps(m) for m in (
        {"jsonrpc": "2.0", "id": 1, "method": "initialize", "params": {}},
        {"jsonrpc": "2.0", "method": "notifications/initialized"},
        {"jsonrpc": "2.0", "id": 2, "method": "tools/list"},
        {"jsonrpc": "2.0", "id": 3, "method": "tools/call", "params": {"name": "HassTurnOn"}},
        {"jsonrpc": "2.0", "id": 4, "method": "resources/list"})) + "\n"
    assert proxy.serve(io.StringIO(lines), out, tmp_path / "missing.env") == 0
    replies = [json.loads(line) for line in out.getvalue().splitlines()]
    assert [r["id"] for r in replies] == [1, 2, 3, 4]          # the notification got none
    assert replies[1]["result"]["tools"] == []
    assert replies[2]["result"]["isError"] and "not set up" in replies[2]["result"]["content"][0]["text"]
    assert replies[3]["error"]["code"] == -32601


def test_serve_end_to_end(server, tmp_path):
    env = tmp_path / "coding.env"
    env.write_text(f"HA_URL={server.url}\nHA_TOKEN={TOKEN}\n")
    out = io.StringIO()
    proxy.serve(io.StringIO('{"jsonrpc":"2.0","id":1,"method":"tools/list"}\n\n'), out, env)
    assert json.loads(out.getvalue())["result"]["tools"][0]["name"] == "HassTurnOn"


def test_uses_its_own_token_file_not_the_voice_one(monkeypatch, tmp_path):
    monkeypatch.delenv("HOMEAI_HA_MCP_ENV", raising=False)
    assert proxy.env_path().name == "ha-coding.env"
    monkeypatch.setenv("HOMEAI_HA_MCP_ENV", str(tmp_path / "x.env"))
    assert proxy.env_path() == tmp_path / "x.env"
