# Copyright (c) 2026 Adam Wosotowsky
# SPDX-License-Identifier: BUSL-1.1
# Licensed under the Business Source License 1.1; see LICENSE at the
# repository root. Converts to Apache-2.0 on 2030-09-15.
"""MCP server exposing this project's local capabilities to the agent.

Why an MCP server rather than a ZeroClaw skill
-----------------------------------------------
The voice agent is deliberately restricted: it has no shell access, because a
voice waveform carries no authentication and anyone within earshot -- including
the television -- can issue commands. A skill that shells out would therefore
either not run for the voice agent, or would require punching a hole in
exactly the boundary that is protecting the machine.

MCP tools are a *narrow, typed* surface. ``weather`` can only ever look up
weather. Adding it grants no new general capability, so the restricted profile
stays restricted. The same reasoning will apply to the Home Assistant and
timer tools that come later, which is why this is built as a general server
rather than a one-off weather patch.

Protocol
--------
JSON-RPC 2.0 over stdio, one message per line. Only the three methods a tool
provider actually needs are implemented: ``initialize``, ``tools/list`` and
``tools/call``. Unknown methods return a proper JSON-RPC error rather than
crashing, because an unrecognised notification from a newer client must not
take the assistant's weather tool offline.

A tool that raises returns ``isError`` with a readable message instead of
propagating, so a transient upstream outage degrades one answer rather than
killing the server.
"""

from __future__ import annotations

import json
import logging
import sys
from typing import Any, Callable

from .research import research_prompt
from .weather import WeatherError, get_weather

log = logging.getLogger("homeai.mcp")

PROTOCOL_VERSION = "2024-11-05"
SERVER_INFO = {"name": "homeai", "version": "1.0.0"}

# JSON-RPC error codes from the specification.
PARSE_ERROR = -32700
INVALID_REQUEST = -32600
METHOD_NOT_FOUND = -32601
INTERNAL_ERROR = -32603


def _tool_weather(arguments: dict[str, Any]) -> str:
    """Current conditions for a place.

    The default location matters: the overwhelmingly common question is
    "what's the weather" with no place at all, and answering for the wrong
    city is worse than answering slowly.
    """
    location = str(arguments.get("location") or "").strip()
    if not location:
        location = DEFAULT_LOCATION

    conditions = get_weather(location)
    # Include the numbers as well as the sentence: the model may be asked a
    # follow-up ("do I need a jacket?") that needs the value, not the prose.
    return (
        f"{conditions.spoken()} "
        f"(temperature {conditions.temperature_f:.0f}F, "
        f"feels like {conditions.feels_like_f:.0f}F, "
        f"humidity {conditions.humidity}%, "
        f"wind {conditions.wind_mph:.0f} mph)"
    )


DEFAULT_LOCATION = "Lilburn, Georgia"


def _tool_research(arguments: dict[str, Any]) -> str:
    """Search the web, read the pages, and return sourced excerpts.

    This returns *source material*, not an answer. The model forms the opinion
    -- that is the point. Handing back snippets instead of page text is what
    produced the infamous "MONDS" answer about football recruits.
    """
    query = str(arguments.get("query") or "").strip()
    if not query:
        return "No research query was given."

    limit = arguments.get("sources", 4)
    try:
        limit = max(1, min(6, int(limit)))
    except (TypeError, ValueError):
        limit = 4

    return research_prompt(query, limit=limit)


TOOLS: list[dict[str, Any]] = [
    {
        "name": "weather",
        "description": (
            "Get current weather conditions and today's high and low for a "
            "location. Returns temperature in Fahrenheit and wind in miles "
            "per hour. If the user does not name a place, omit the location "
            "argument to use their home location."
        ),
        "inputSchema": {
            "type": "object",
            "properties": {
                "location": {
                    "type": "string",
                    "description": (
                        "City and state or country, for example "
                        "'Lilburn, Georgia' or 'Paris, France'. Omit for the "
                        "user's home location."
                    ),
                }
            },
            "required": [],
        },
    },
    {
        "name": "research",
        "description": (
            "Search the web and read the resulting pages, returning sourced "
            "excerpts. Use this whenever you are asked about something you "
            "are unsure of, that may have changed recently, or that involves "
            "specific facts, names, numbers, or current events. Before "
            "calling it, tell the user you are looking it up. The tool "
            "returns source material, not an answer: read the excerpts and "
            "form your own conclusion, and say so if the sources are weak or "
            "disagree. Never state anything the sources do not support."
        ),
        "inputSchema": {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": (
                        "A specific, self-contained search query. Expand any "
                        "pronouns or references from earlier in the "
                        "conversation, because this tool has no conversational "
                        "context: search 'MOND modified Newtonian dynamics' "
                        "rather than 'MONDS'."
                    ),
                },
                "sources": {
                    "type": "integer",
                    "description": "How many pages to read, 1 to 6. Default 4.",
                },
            },
            "required": ["query"],
        },
    },
]

HANDLERS: dict[str, Callable[[dict[str, Any]], str]] = {
    "weather": _tool_weather,
    "research": _tool_research,
}


def _result(request_id: Any, result: Any) -> dict[str, Any]:
    return {"jsonrpc": "2.0", "id": request_id, "result": result}


def _error(request_id: Any, code: int, message: str) -> dict[str, Any]:
    return {
        "jsonrpc": "2.0",
        "id": request_id,
        "error": {"code": code, "message": message},
    }


def _text_content(text: str, is_error: bool = False) -> dict[str, Any]:
    return {"content": [{"type": "text", "text": text}], "isError": is_error}


def handle_message(message: dict[str, Any]) -> dict[str, Any] | None:
    """Process one JSON-RPC message. Returns None for notifications.

    Notifications (messages with no ``id``) must not be answered; replying to
    one is a protocol violation that some clients treat as fatal.
    """
    method = message.get("method")
    request_id = message.get("id")
    is_notification = "id" not in message

    if not method:
        return None if is_notification else _error(
            request_id, INVALID_REQUEST, "missing method"
        )

    if method == "initialize":
        return _result(
            request_id,
            {
                "protocolVersion": PROTOCOL_VERSION,
                "capabilities": {"tools": {}},
                "serverInfo": SERVER_INFO,
            },
        )

    if method == "tools/list":
        return _result(request_id, {"tools": TOOLS})

    if method == "tools/call":
        params = message.get("params") or {}
        name = params.get("name")
        arguments = params.get("arguments") or {}

        handler = HANDLERS.get(name)
        if handler is None:
            return _result(
                request_id, _text_content(f"unknown tool: {name}", is_error=True)
            )

        try:
            return _result(request_id, _text_content(handler(arguments)))
        except WeatherError as exc:
            # Expected, actionable failure: report it as tool output so the
            # model can tell the user plainly rather than inventing a forecast.
            return _result(request_id, _text_content(str(exc), is_error=True))
        except Exception as exc:  # noqa: BLE001 - a tool must not kill the server
            log.exception("tool %s failed", name)
            return _result(
                request_id,
                _text_content(f"tool {name} failed: {exc}", is_error=True),
            )

    if is_notification:
        # e.g. notifications/initialized -- acknowledge by staying silent.
        return None

    return _error(request_id, METHOD_NOT_FOUND, f"unknown method: {method}")


def serve(stdin=None, stdout=None) -> int:
    """Run the stdio message loop until EOF."""
    stdin = stdin or sys.stdin
    stdout = stdout or sys.stdout

    for line in stdin:
        line = line.strip()
        if not line:
            continue

        try:
            message = json.loads(line)
        except json.JSONDecodeError:
            response = _error(None, PARSE_ERROR, "invalid JSON")
        else:
            try:
                response = handle_message(message)
            except Exception as exc:  # noqa: BLE001
                log.exception("dispatch failed")
                response = _error(
                    message.get("id"), INTERNAL_ERROR, f"internal error: {exc}"
                )

        if response is not None:
            stdout.write(json.dumps(response) + "\n")
            stdout.flush()

    return 0


if __name__ == "__main__":
    logging.basicConfig(level=logging.WARNING, stream=sys.stderr)
    raise SystemExit(serve())
