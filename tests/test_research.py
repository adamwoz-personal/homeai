# Copyright (c) 2026 Adam Wosotowsky
# SPDX-License-Identifier: BUSL-1.1
# Licensed under the Business Source License 1.1; see LICENSE at the
# repository root. Converts to Apache-2.0 on 2030-09-15.
"""Tests for web research and the MCP server surface.

No live network. The research module's whole purpose is to be the difference
between a sourced answer and an invented one, so the tests focus on the cases
where it must *refuse* to supply material rather than hand back something thin
enough to invite invention.
"""

from __future__ import annotations

import json
import io

import pytest

from homeai import research as research_mod
from homeai.mcp_server import handle_message, serve
from homeai.research import Research, Source, extract_text, gather, search
from homeai.weather import WeatherError

SEARCH_HTML = """
<div class="result">
  <a class="result__a" href="https://en.wikipedia.org/wiki/MOND">Modified <b>Newtonian</b> dynamics</a>
</div>
<div class="result">
  <a class="result__a" href="//duckduckgo.com/l/?uddg=https%3A%2F%2Fbigthink.com%2Fdark-matter">Big Think</a>
</div>
<div class="result">
  <a class="result__a" href="https://pinterest.com/pin/1">Pinterest board</a>
</div>
"""


class TestExtractText:
    def test_strips_tags_and_scripts(self):
        html = "<p>Hello</p><script>evil()</script><style>x{}</style><p>World</p>"
        out = extract_text(html)
        assert "evil" not in out and "x{}" not in out
        assert "Hello" in out and "World" in out

    def test_block_elements_become_line_breaks(self):
        """Without this, sentences weld into one unreadable run."""
        out = extract_text("<p>First sentence.</p><p>Second sentence.</p>")
        assert "First sentence." in out
        assert "sentence.Second" not in out.replace(" ", "")

    def test_entities_are_unescaped(self):
        assert "Tom & Jerry" in extract_text("<p>Tom &amp; Jerry</p>")

    def test_empty_input(self):
        assert extract_text("") == ""


class TestSearch:
    def test_parses_results(self, monkeypatch):
        monkeypatch.setattr(research_mod, "_SEARCH_URL", "http://stub")
        monkeypatch.setattr(
            research_mod.urllib.request,
            "urlopen",
            lambda *a, **k: _FakeResponse(SEARCH_HTML),
        )
        results = search("mond")
        urls = [r.url for r in results]
        assert "https://en.wikipedia.org/wiki/MOND" in urls

    def test_unwraps_redirector(self, monkeypatch):
        monkeypatch.setattr(
            research_mod.urllib.request,
            "urlopen",
            lambda *a, **k: _FakeResponse(SEARCH_HTML),
        )
        urls = [r.url for r in search("mond")]
        assert "https://bigthink.com/dark-matter" in urls
        assert not any("duckduckgo.com/l/" in u for u in urls)

    def test_skips_social_domains(self, monkeypatch):
        monkeypatch.setattr(
            research_mod.urllib.request,
            "urlopen",
            lambda *a, **k: _FakeResponse(SEARCH_HTML),
        )
        assert not any("pinterest" in r.url for r in search("mond"))

    def test_network_failure_returns_empty_not_raises(self, monkeypatch):
        def boom(*_a, **_k):
            raise OSError("network down")

        monkeypatch.setattr(research_mod.urllib.request, "urlopen", boom)
        assert search("anything") == []

    def test_blank_query(self):
        assert search("") == []


class _FakeResponse:
    def __init__(self, body: str, ctype: str = "text/html; charset=utf-8"):
        self._body = body.encode()
        self.headers = _Headers(ctype)

    def read(self, *_args):
        return self._body

    def __enter__(self):
        return self

    def __exit__(self, *_):
        return False


class _Headers:
    def __init__(self, ctype):
        self._ctype = ctype

    def get(self, _key, default=""):
        return self._ctype

    def get_content_charset(self):
        return "utf-8"


class TestGather:
    def test_thin_pages_are_rejected(self, monkeypatch):
        """Consent walls yield boilerplate that invites invention."""
        monkeypatch.setattr(
            research_mod,
            "search",
            lambda q, limit, timeout: [Source("T", "https://x.test")],
        )
        monkeypatch.setattr(
            research_mod, "_fetch", lambda url, timeout: "<p>Accept cookies</p>"
        )
        result = gather("q")
        assert result.usable == []
        assert result.error

    def test_one_bad_link_does_not_abort_the_batch(self, monkeypatch):
        good = "<p>" + ("substantial content. " * 60) + "</p>"

        monkeypatch.setattr(
            research_mod,
            "search",
            lambda q, limit, timeout: [
                Source("Bad", "https://bad.test"),
                Source("Good", "https://good.test"),
            ],
        )

        def fetch(url, timeout):
            if "bad" in url:
                raise OSError("connection refused")
            return good

        monkeypatch.setattr(research_mod, "_fetch", fetch)
        result = gather("q")
        assert len(result.usable) == 1
        assert result.usable[0].title == "Good"

    def test_no_results_reports_error(self, monkeypatch):
        monkeypatch.setattr(research_mod, "search", lambda q, limit, timeout: [])
        assert gather("q").error == "no search results"


class TestResearchPrompt:
    def test_includes_sources_and_domains(self):
        research = Research(
            query="q",
            sources=[Source("Title", "https://en.wikipedia.org/wiki/X", "body text")],
        )
        prompt = research.as_prompt()
        assert "en.wikipedia.org" in prompt
        assert "body text" in prompt

    def test_empty_results_instruct_against_guessing(self):
        prompt = Research(query="q", sources=[]).as_prompt()
        assert "rather than guessing" in prompt

    def test_truncates_long_sources(self):
        research = Research(query="q", sources=[Source("T", "https://a.test", "x" * 9000)])
        assert len(research.as_prompt(per_source_chars=100)) < 1000


class TestMcpProtocol:
    def test_initialize_reports_protocol(self):
        response = handle_message({"jsonrpc": "2.0", "id": 1, "method": "initialize"})
        assert response["result"]["serverInfo"]["name"] == "homeai"

    def test_tools_list_exposes_both_tools(self):
        response = handle_message({"jsonrpc": "2.0", "id": 2, "method": "tools/list"})
        names = {t["name"] for t in response["result"]["tools"]}
        assert names == {"weather", "research"}

    def test_notification_gets_no_reply(self):
        """Replying to a notification is a protocol violation."""
        assert handle_message({"jsonrpc": "2.0", "method": "notifications/initialized"}) is None

    def test_unknown_method_returns_error_not_crash(self):
        response = handle_message({"jsonrpc": "2.0", "id": 3, "method": "nope"})
        assert response["error"]["code"] == -32601

    def test_unknown_tool_is_reported_as_tool_error(self):
        response = handle_message(
            {"jsonrpc": "2.0", "id": 4, "method": "tools/call",
             "params": {"name": "ghost", "arguments": {}}}
        )
        assert response["result"]["isError"] is True

    def test_weather_tool_error_does_not_kill_server(self, monkeypatch):
        import homeai.mcp_server as mcp

        def boom(_location):
            raise WeatherError("cannot reach weather service")

        monkeypatch.setattr(mcp, "get_weather", boom)
        response = handle_message(
            {"jsonrpc": "2.0", "id": 5, "method": "tools/call",
             "params": {"name": "weather", "arguments": {"location": "X"}}}
        )
        assert response["result"]["isError"] is True
        assert "cannot reach" in response["result"]["content"][0]["text"]

    def test_unexpected_exception_is_contained(self, monkeypatch):
        import homeai.mcp_server as mcp

        monkeypatch.setattr(
            mcp, "get_weather", lambda _l: (_ for _ in ()).throw(RuntimeError("boom"))
        )
        response = handle_message(
            {"jsonrpc": "2.0", "id": 6, "method": "tools/call",
             "params": {"name": "weather", "arguments": {}}}
        )
        assert response["result"]["isError"] is True

    def test_research_requires_a_query(self):
        response = handle_message(
            {"jsonrpc": "2.0", "id": 7, "method": "tools/call",
             "params": {"name": "research", "arguments": {}}}
        )
        assert "No research query" in response["result"]["content"][0]["text"]

    def test_research_clamps_source_count(self, monkeypatch):
        import homeai.mcp_server as mcp

        seen = {}

        def fake(query, limit):
            seen["limit"] = limit
            return "ok"

        monkeypatch.setattr(mcp, "research_prompt", fake)
        handle_message(
            {"jsonrpc": "2.0", "id": 8, "method": "tools/call",
             "params": {"name": "research", "arguments": {"query": "q", "sources": 99}}}
        )
        assert seen["limit"] == 6

    def test_garbage_source_count_falls_back_to_default(self, monkeypatch):
        import homeai.mcp_server as mcp

        seen = {}
        monkeypatch.setattr(
            mcp, "research_prompt",
            lambda query, limit: seen.setdefault("limit", limit) or "ok",
        )
        handle_message(
            {"jsonrpc": "2.0", "id": 9, "method": "tools/call",
             "params": {"name": "research",
                        "arguments": {"query": "q", "sources": "many"}}}
        )
        assert seen["limit"] == 4


class TestMcpStdioLoop:
    def test_invalid_json_returns_parse_error_and_continues(self):
        stdin = io.StringIO('not json\n{"jsonrpc":"2.0","id":1,"method":"tools/list"}\n')
        stdout = io.StringIO()
        serve(stdin, stdout)
        lines = [json.loads(l) for l in stdout.getvalue().splitlines()]
        assert lines[0]["error"]["code"] == -32700
        assert "tools" in lines[1]["result"]

    def test_blank_lines_ignored(self):
        stdout = io.StringIO()
        serve(io.StringIO("\n\n"), stdout)
        assert stdout.getvalue() == ""
