# Copyright (c) 2026 Adam Wosotowsky
# SPDX-License-Identifier: BUSL-1.1
# Licensed under the Business Source License 1.1; see LICENSE at the
# repository root. Converts to Apache-2.0 on 2030-09-15.
"""Unit tests for the stdio JSON-RPC MCP server.

The server is a deliberately narrow boundary for a restricted voice agent, so
these tests exercise the protocol surface directly and patch every networked
tool seam.
"""

from __future__ import annotations

import io
import json

import pytest

from homeai import mcp_server as mcp
from homeai.weather import Conditions, Place, WeatherError


def _request(method: str, *, request_id: int = 1, params: dict | None = None) -> dict:
    message = {"jsonrpc": "2.0", "id": request_id, "method": method}
    if params is not None:
        message["params"] = params
    return message


def _call_tool(name: str, arguments: dict | None = None, *, request_id: int = 1) -> dict:
    return mcp.handle_message(
        _request(
            "tools/call",
            request_id=request_id,
            params={"name": name, "arguments": arguments or {}},
        )
    )


def _text(response: dict) -> str:
    return response["result"]["content"][0]["text"]


def _serve_lines(*messages: str) -> list[dict]:
    stdin = io.StringIO("\n".join(messages) + "\n")
    stdout = io.StringIO()

    assert mcp.serve(stdin, stdout) == 0

    return [json.loads(line) for line in stdout.getvalue().splitlines()]


@pytest.fixture
def fake_conditions() -> Conditions:
    return Conditions(
        place=Place("Lilburn", 33.8901, -84.1429, admin="Georgia"),
        temperature_f=72,
        feels_like_f=75,
        description="clear",
        humidity=45,
        wind_mph=6,
        high_f=81,
        low_f=63,
    )


def test_initialize_handshake_reports_required_fields() -> None:
    response = mcp.handle_message(_request("initialize"))

    assert response == {
        "jsonrpc": "2.0",
        "id": 1,
        "result": {
            "protocolVersion": mcp.PROTOCOL_VERSION,
            "capabilities": {"tools": {}},
            "serverInfo": mcp.SERVER_INFO,
        },
    }


def test_notifications_get_no_response() -> None:
    assert (
        mcp.handle_message(
            {"jsonrpc": "2.0", "method": "notifications/initialized"}
        )
        is None
    )


def test_tools_list_exposes_weather_and_research_metadata() -> None:
    response = mcp.handle_message(_request("tools/list"))
    tools = {tool["name"]: tool for tool in response["result"]["tools"]}

    assert set(tools) == {"weather", "research"}
    for tool in tools.values():
        assert tool["description"]
        assert tool["inputSchema"]["type"] == "object"
        assert "properties" in tool["inputSchema"]
    assert "location" in tools["weather"]["inputSchema"]["properties"]
    assert "query" in tools["research"]["inputSchema"]["properties"]
    assert "sources" in tools["research"]["inputSchema"]["properties"]


def test_research_description_forbids_opinions() -> None:
    research_tool = next(tool for tool in mcp.TOOLS if tool["name"] == "research")

    assert "Never use it for opinions" in research_tool["description"]


def test_weather_call_returns_spoken_conditions(monkeypatch, fake_conditions) -> None:
    calls: list[str] = []

    def fake_get_weather(location: str) -> Conditions:
        calls.append(location)
        return fake_conditions

    monkeypatch.setattr(mcp, "get_weather", fake_get_weather)

    response = _call_tool("weather", {"location": "Lilburn, Georgia"})

    assert calls == ["Lilburn, Georgia"]
    assert response["result"]["isError"] is False
    assert "It's 72 degrees and clear in Lilburn, Georgia." in _text(response)
    assert "temperature 72F" in _text(response)
    assert "wind 6 mph" in _text(response)


def test_weather_uses_home_location_when_omitted(monkeypatch, fake_conditions) -> None:
    calls: list[str] = []
    monkeypatch.setattr(
        mcp,
        "get_weather",
        lambda location: calls.append(location) or fake_conditions,
    )

    _call_tool("weather")

    assert calls == [mcp.DEFAULT_LOCATION]


def test_research_call_returns_prompt(monkeypatch) -> None:
    calls: list[tuple[str, int]] = []

    def fake_research_prompt(query: str, *, limit: int) -> str:
        calls.append((query, limit))
        return "sourced research"

    monkeypatch.setattr(mcp, "research_prompt", fake_research_prompt)

    response = _call_tool("research", {"query": "latest moon mission", "sources": 3})

    assert calls == [("latest moon mission", 3)]
    assert response["result"]["isError"] is False
    assert _text(response) == "sourced research"


def test_research_empty_query_does_not_call_network(monkeypatch) -> None:
    def fail_if_called(*_args, **_kwargs):
        raise AssertionError("research_prompt should not be called")

    monkeypatch.setattr(mcp, "research_prompt", fail_if_called)

    response = _call_tool("research", {"query": "  "})

    assert _text(response) == "No research query was given."


@pytest.mark.parametrize(
    ("raw_sources", "expected"),
    [
        (0, 1),
        (-2, 1),
        (7, 6),
        ("6", 6),
        ("many", 4),
        (None, 4),
    ],
)
def test_research_sources_are_clamped_or_defaulted(
    monkeypatch, raw_sources, expected
) -> None:
    seen: list[int] = []

    def fake_research_prompt(_query: str, *, limit: int) -> str:
        seen.append(limit)
        return "ok"

    monkeypatch.setattr(mcp, "research_prompt", fake_research_prompt)

    _call_tool("research", {"query": "q", "sources": raw_sources})

    assert seen == [expected]


def test_unknown_tool_returns_error_result_not_exception() -> None:
    response = _call_tool("not-a-tool")

    assert response["result"]["isError"] is True
    assert _text(response) == "unknown tool: not-a-tool"


def test_unknown_method_returns_json_rpc_method_not_found() -> None:
    response = mcp.handle_message(_request("does/not/exist", request_id=44))

    assert response["jsonrpc"] == "2.0"
    assert response["id"] == 44
    assert response["error"]["code"] == mcp.METHOD_NOT_FOUND
    assert response["error"]["message"] == "unknown method: does/not/exist"


def test_malformed_json_response_is_parse_error_and_loop_continues() -> None:
    good_request = json.dumps(_request("tools/list", request_id=12))

    responses = _serve_lines("{not json", good_request)

    assert responses[0]["error"]["code"] == mcp.PARSE_ERROR
    assert responses[0]["id"] is None
    assert {tool["name"] for tool in responses[1]["result"]["tools"]} == {
        "weather",
        "research",
    }


def test_notification_inside_stdio_loop_writes_no_response() -> None:
    notification = json.dumps(
        {"jsonrpc": "2.0", "method": "notifications/initialized"}
    )
    request = json.dumps(_request("initialize", request_id=99))

    responses = _serve_lines(notification, request)

    assert len(responses) == 1
    assert responses[0]["id"] == 99


def test_tool_weather_error_is_tool_result_and_loop_continues(monkeypatch) -> None:
    def boom(_location: str) -> Conditions:
        raise WeatherError("cannot reach weather service")

    monkeypatch.setattr(mcp, "get_weather", boom)
    bad_weather = json.dumps(
        _request(
            "tools/call",
            request_id=20,
            params={"name": "weather", "arguments": {"location": "Nowhere"}},
        )
    )
    good_list = json.dumps(_request("tools/list", request_id=21))

    responses = _serve_lines(bad_weather, good_list)

    assert responses[0]["result"]["isError"] is True
    assert _text(responses[0]) == "cannot reach weather service"
    assert {tool["name"] for tool in responses[1]["result"]["tools"]} == {
        "weather",
        "research",
    }


def test_unexpected_tool_exception_is_error_result_and_loop_continues(
    monkeypatch,
) -> None:
    def boom(_arguments: dict) -> str:
        raise RuntimeError("boom")

    monkeypatch.setitem(mcp.HANDLERS, "weather", boom)
    bad_tool = json.dumps(
        _request(
            "tools/call",
            request_id=30,
            params={"name": "weather", "arguments": {"location": "X"}},
        )
    )
    good_initialize = json.dumps(_request("initialize", request_id=31))

    responses = _serve_lines(bad_tool, good_initialize)

    assert responses[0]["result"]["isError"] is True
    assert _text(responses[0]) == "tool weather failed: boom"
    assert responses[1]["result"]["serverInfo"]["name"] == "homeai"


# -- home tools ---------------------------------------------------------------

from homeai.home import EchoQuiet, Home, TimerLedger  # noqa: E402
from tests.fake_ha import FakeHA  # noqa: E402

HOME_TOOL_NAMES = {"lights", "timer", "announce", "music"}


@pytest.fixture
def house(monkeypatch, tmp_path):
    ha = FakeHA()
    monkeypatch.setattr(mcp, "_HOME", Home(ha, ledger=TimerLedger(tmp_path / "t.json"),
                                           quiet=EchoQuiet(tmp_path / "q"),
                                           sleep=ha.slept.append))
    return ha


def _tool_names(response):
    return {t["name"] for t in response["result"]["tools"]}


def test_home_tools_hidden_without_a_house():
    assert not HOME_TOOL_NAMES & _tool_names(mcp.handle_message(_request("tools/list")))


def test_home_tools_listed_with_a_house(monkeypatch):
    monkeypatch.setenv("HA_TOKEN", "tok")
    assert HOME_TOOL_NAMES <= _tool_names(mcp.handle_message(_request("tools/list")))


def test_home_tool_without_a_house_is_a_clean_error(monkeypatch):
    monkeypatch.setattr(mcp, "_HOME", None)
    response = _call_tool("lights", {"target": "foyer", "action": "off"})
    assert response["result"]["isError"] is True
    assert "isn't set up" in _text(response)


def test_lights_tool(house):
    response = _call_tool("lights", {"target": "foyer", "action": "off"})
    assert _text(response) == "Foyer off."
    assert house.calls[-1] == ("light", "turn_off", {"entity_id": ["light.foyer_foyer"]})


def test_timer_tool_minutes(house):
    assert _text(_call_tool("timer", {"action": "set", "minutes": 2.5, "label": "eggs"})) == \
        "Eggs timer set for 2 minutes and 30 seconds."
    assert _text(_call_tool("timer", {"action": "status"})).startswith("The eggs timer has")
    assert _text(_call_tool("timer", {"action": "cancel"})) == "Eggs timer cancelled."


@pytest.mark.parametrize("args", [{"action": "set"}, {"action": "set", "minutes": "soon"},
                                  {"action": "snooze"}])
def test_timer_tool_bad_arguments(house, args):
    response = _call_tool("timer", args)
    assert response["result"]["isError"] is True
    assert house.calls == []


def test_announce_and_music_tools(house):
    assert _text(_call_tool("announce", {"message": "dinner"})) == "Announced."
    assert _text(_call_tool("music", {"action": "play", "request": "jazz"})) == \
        "Playing jazz on the Kitchen alexa."


def test_music_tool_refuses_purchases(house):
    response = _call_tool("music", {"action": "play", "request": "buy it"})
    assert response["result"]["isError"] is True
    assert house.alexa_commands() == []
