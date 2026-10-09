#!/usr/bin/env python3
"""Live check of the coding MCP proxy: initialize + tools/list through
bin/homeai-ha-mcp. Exit 0 = tools listed, 1 = not set up / failing."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
MESSAGES = [
    {"jsonrpc": "2.0", "id": 1, "method": "initialize", "params": {
        "protocolVersion": "2025-06-18", "capabilities": {},
        "clientInfo": {"name": "ha_mcp_check", "version": "0"}}},
    {"jsonrpc": "2.0", "method": "notifications/initialized"},
    {"jsonrpc": "2.0", "id": 2, "method": "tools/list"},
]


def main() -> int:
    stdin = "".join(json.dumps(m) + "\n" for m in MESSAGES)
    proc = subprocess.run([str(ROOT / "bin" / "homeai-ha-mcp")], input=stdin,
                          capture_output=True, text=True, timeout=120)
    replies = {r["id"]: r for r in map(json.loads, proc.stdout.splitlines())}
    if proc.stderr.strip():
        print(proc.stderr.strip())
    err = replies.get(2, {}).get("error")
    tools = replies.get(2, {}).get("result", {}).get("tools", [])
    if err or not tools:
        print("FAIL:", err["message"] if err else "no tools (HA MCP not set up?)")
        return 1
    print(f"OK: {len(tools)} tools: " + ", ".join(t["name"] for t in tools))
    return 0


if __name__ == "__main__":
    sys.exit(main())
