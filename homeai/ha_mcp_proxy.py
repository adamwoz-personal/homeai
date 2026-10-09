# Copyright (c) 2026 Adam Wosotowsky
# SPDX-License-Identifier: BUSL-1.1
# Licensed under the Business Source License 1.1; see LICENSE at the
# repository root. Converts to Apache-2.0 on 2030-09-15.
"""stdio MCP proxy to Home Assistant's own MCP server, for CODING tools only.

Copilot CLI and ZeroClaw's `builder` agent launch this (bin/homeai-ha-mcp); it
forwards each JSON-RPC message to HA's stateless Streamable HTTP endpoint
(POST <HA_URL>/api/mcp). Why a proxy rather than pointing the clients at the
URL: the bearer token stays in a mode-600 file instead of in each client's
config, and one entry works for every stdio-only client.

The voice assistant must never see this: its home tools are the narrow typed
ones in homeai/mcp_server.py. This server exposes HA's Assist LLM API (only
entities exposed to Assist). Keep the garage door out of Assist exposure.

Token: a SEPARATE file from the voice token, ~/.config/homeai/ha-coding.env
(or $HOMEAI_HA_MCP_ENV) with HA_URL= and HA_TOKEN=. HA requires an admin
user for /api/mcp unless "Require admin" is turned off in the integration's
options; see docs/RUNBOOK.md "Home Assistant MCP for coding".

Never crashes the client: any failure becomes a JSON-RPC error or an
isError tool result with a sentence saying what to fix.
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path
from typing import IO, Any

import requests

from .ha_client import HASettings, is_configured, load_settings

DEFAULT_ENV = Path(os.path.expanduser("~/.config/homeai/ha-coding.env"))
TIMEOUT_S = 60
PROTOCOL = "2025-06-18"
SETUP = ("Home Assistant MCP is not set up. Add the 'Model Context Protocol Server' "
         "integration in Home Assistant, create a token, and put HA_URL= and HA_TOKEN= "
         f"in {DEFAULT_ENV} (chmod 600). See docs/RUNBOOK.md.")

HINTS = {
    401: "Home Assistant rejected the coding token; create a new long-lived token.",
    403: "Home Assistant says this user may not use MCP: use an admin user, or turn "
         "off 'Require admin' in the Model Context Protocol Server options.",
    404: "Home Assistant has no MCP server: add the 'Model Context Protocol Server' "
         "integration (Settings > Devices & services).",
}


def env_path() -> Path:
    return Path(os.environ.get("HOMEAI_HA_MCP_ENV") or DEFAULT_ENV)


def _error(msg_id: Any, code: int, message: str) -> dict:
    return {"jsonrpc": "2.0", "id": msg_id, "error": {"code": code, "message": message}}


def _tool_error(msg_id: Any, text: str) -> dict:
    return {"jsonrpc": "2.0", "id": msg_id,
            "result": {"content": [{"type": "text", "text": text}], "isError": True}}


def _local(msg: dict) -> dict:
    """Answer without HA: used while HA MCP is not configured."""
    method, msg_id = msg.get("method"), msg.get("id")
    if method == "initialize":
        return {"jsonrpc": "2.0", "id": msg_id, "result": {
            "protocolVersion": PROTOCOL, "capabilities": {"tools": {}},
            "serverInfo": {"name": "homeassistant (not set up)", "version": "0"}}}
    if method == "tools/list":
        return {"jsonrpc": "2.0", "id": msg_id, "result": {"tools": []}}
    if method == "tools/call":
        return _tool_error(msg_id, SETUP)
    if method == "ping":
        return {"jsonrpc": "2.0", "id": msg_id, "result": {}}
    return _error(msg_id, -32601, "Method not found")


def forward(msg: dict, settings: HASettings, post=requests.post) -> dict | None:
    """Send one message to HA. Returns the reply, or None for notifications."""
    msg_id = msg.get("id")
    try:
        resp = post(f"{settings.url}/api/mcp", json=msg, timeout=TIMEOUT_S, headers={
            "Authorization": f"Bearer {settings.token}",
            "Accept": "application/json",
            "Content-Type": "application/json"})
    except requests.RequestException as exc:
        if msg_id is None:
            return None
        return _error(msg_id, -32000, f"Cannot reach Home Assistant at {settings.url} "
                                      f"({type(exc).__name__})")
    if msg_id is None:  # notification: HA answers 202 and there is nothing to relay
        return None
    if resp.status_code in HINTS:
        return _error(msg_id, -32000, HINTS[resp.status_code])
    if resp.status_code >= 400:
        return _error(msg_id, -32000, f"Home Assistant returned HTTP {resp.status_code}")
    try:
        reply = resp.json()
    except ValueError:
        return _error(msg_id, -32000, "Home Assistant sent something that is not JSON")
    return reply if isinstance(reply, dict) else _error(msg_id, -32000, "Unexpected reply shape")


def handle(line: str, settings: HASettings | None, post=requests.post) -> dict | None:
    try:
        msg = json.loads(line)
    except ValueError:
        return _error(None, -32700, "Parse error")
    if not isinstance(msg, dict):
        return _error(None, -32600, "Invalid request")
    if "method" not in msg:  # a response from the client; nothing to do
        return None
    if settings is None:
        return _local(msg) if msg.get("id") is not None else None
    return forward(msg, settings, post)


def serve(stdin: IO[str], stdout: IO[str], path: Path | None = None, post=requests.post) -> int:
    path = path or env_path()
    settings = load_settings(path) if is_configured(path) else None
    if settings is None:
        print(f"homeai-ha-mcp: {SETUP}", file=sys.stderr)
    for line in stdin:
        if not line.strip():
            continue
        reply = handle(line, settings, post)
        if reply is not None:
            stdout.write(json.dumps(reply) + "\n")
            stdout.flush()
    return 0


def main() -> int:
    return serve(sys.stdin, sys.stdout)


if __name__ == "__main__":
    sys.exit(main())
