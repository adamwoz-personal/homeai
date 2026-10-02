#!/usr/bin/env python3
# Copyright (c) 2026 Adam Wosotowsky
# SPDX-License-Identifier: BUSL-1.1
# Licensed under the Business Source License 1.1; see LICENSE at the
# repository root. Converts to Apache-2.0 on 2030-09-15.
"""Show exactly what ZeroClaw sends to the model for one question.

Why this exists
---------------
SOUL.md is only part of the prompt. ZeroClaw wraps it in its own system text,
adds AGENTS.md, a tool list and conversation scaffolding. When the agent
misbehaves through ZeroClaw but not when the model is called directly (hedging
was measured at 12/36 through ZeroClaw against 1/33 direct), the only way to
know why is to read the actual request.

How it works, without touching the live voice agent
---------------------------------------------------
1. Copies ``config.toml`` and the agent's workspace into a temp directory.
2. Rewrites the model endpoint in that copy to point at a local recording
   proxy, which forwards to the real endpoint.
3. Runs ``zeroclaw agent --config-dir <temp>`` once and prints the captured
   system prompt, tool names and message roles.

The live ``~/.zeroclaw`` is never modified.

Usage
-----
    probe_agent_prompt.py "What do you think about free will?"
    probe_agent_prompt.py --agent local --full --save /tmp/prompt.json "hi"
"""

from __future__ import annotations

import argparse
import http.server
import json
import shutil
import socket
import subprocess
import sys
import tempfile
import threading
import urllib.request
from pathlib import Path

ZC_HOME = Path.home() / ".zeroclaw"


class _Recorder(http.server.BaseHTTPRequestHandler):
    upstream = ""
    captured: list[dict] = []

    def log_message(self, *args):  # silence default stderr logging
        pass

    def _forward(self, body: bytes | None):
        req = urllib.request.Request(
            self.upstream + self.path,
            data=body,
            method=self.command,
            headers={k: v for k, v in self.headers.items() if k.lower() != "host"},
        )
        try:
            with urllib.request.urlopen(req, timeout=600) as resp:
                data = resp.read()
                status, headers = resp.status, resp.headers
        except urllib.error.HTTPError as exc:
            data, status, headers = exc.read(), exc.code, exc.headers
        self.send_response(status)
        for key, value in headers.items():
            if key.lower() not in ("transfer-encoding", "connection", "content-length"):
                self.send_header(key, value)
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def do_POST(self):
        body = self.rfile.read(int(self.headers.get("Content-Length", 0)))
        try:
            self.captured.append({"path": self.path, "body": json.loads(body)})
        except ValueError:
            self.captured.append({"path": self.path, "raw": body.decode(errors="replace")})
        self._forward(body)

    def do_GET(self):
        self._forward(None)


def _free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def set_toml_key(config: str, section: str, assignment: str) -> str:
    """Set ``key = value`` inside ``[section]``, replacing an existing key.

    Text-level on purpose: no TOML writer is installed, and round-tripping
    through a parser would drop the config's explanatory comments.
    """
    key = assignment.split("=", 1)[0].strip()
    lines = config.splitlines()
    header = f"[{section}]"
    try:
        start = next(i for i, ln in enumerate(lines) if ln.strip() == header)
    except StopIteration:
        raise ValueError(f"section {header} not found") from None
    end = next((i for i in range(start + 1, len(lines))
                if lines[i].lstrip().startswith("[")), len(lines))
    for i in range(start + 1, end):
        if lines[i].split("=", 1)[0].strip() == key:
            lines[i] = assignment
            break
    else:
        lines.insert(start + 1, assignment)
    return "\n".join(lines) + "\n"


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("message")
    ap.add_argument("--agent", default="local")
    ap.add_argument("--upstream", default="http://127.0.0.1:11434",
                    help="real model server; its URL is rewritten in the temp config")
    ap.add_argument("--full", action="store_true", help="print the full system prompt")
    ap.add_argument("--save", help="write captured requests as JSON")
    ap.add_argument("--with-memory", action="store_true",
                    help="link the live data/ directory so ZeroClaw's own memory "
                         "recall is included (the run may write a memory row)")
    ap.add_argument("--set", nargs=2, action="append", default=[],
                    metavar=("SECTION", "KEY=VALUE"),
                    help="override a key in the TEMP config only, e.g. "
                         "--set risk_profiles.voice 'allowed_tools = [\"calculator\"]'")
    args = ap.parse_args(argv)

    port = _free_port()
    proxy = f"http://127.0.0.1:{port}"
    _Recorder.upstream = args.upstream.rstrip("/")
    server = http.server.ThreadingHTTPServer(("127.0.0.1", port), _Recorder)
    threading.Thread(target=server.serve_forever, daemon=True).start()

    with tempfile.TemporaryDirectory(prefix="zc-probe-") as tmp:
        tmpdir = Path(tmp)
        config = (ZC_HOME / "config.toml").read_text(encoding="utf-8")
        if args.upstream.rstrip("/") not in config:
            print(f"error: {args.upstream} not found in config.toml", file=sys.stderr)
            return 2
        config = config.replace(args.upstream.rstrip("/"), proxy)
        for section, assignment in args.set:
            config = set_toml_key(config, section, assignment)
        # Encrypted values (gateway pairing tokens) need .secret_key, which
        # ZeroClaw refuses to read through a symlink. The voice agent needs no
        # secrets, so drop them rather than copying the key anywhere.
        stripped = [ln for ln in config.splitlines() if "enc2:" not in ln]
        (tmpdir / "config.toml").write_text("\n".join(stripped) + "\n", encoding="utf-8")
        src = ZC_HOME / "agents" / args.agent
        if src.exists():
            shutil.copytree(src, tmpdir / "agents" / args.agent, symlinks=True,
                            ignore=shutil.ignore_patterns("*.sqlite*", "*.db", "state"))
        if args.with_memory and (ZC_HOME / "data").exists():
            (tmpdir / "data").symlink_to(ZC_HOME / "data")
        for name in ("shared", "voice-scratch"):
            if (ZC_HOME / name).exists():
                (tmpdir / name).symlink_to(ZC_HOME / name)

        proc = subprocess.run(
            ["zeroclaw", "agent", "--config-dir", str(tmpdir), "-a", args.agent,
             "--message", args.message],
            capture_output=True, text=True, timeout=600,
        )
    server.shutdown()

    reqs = [c for c in _Recorder.captured if "body" in c and "messages" in c["body"]]
    print(f"zeroclaw exit {proc.returncode}; {len(reqs)} model request(s) captured")
    if proc.returncode != 0:
        print(proc.stderr[-2000:])
    if not reqs:
        return 1

    first = reqs[0]["body"]
    msgs = first.get("messages", [])
    system = "\n\n".join(m.get("content") or "" for m in msgs if m.get("role") == "system")
    tools = [t.get("function", {}).get("name") for t in first.get("tools", []) or []]
    print(f"model {first.get('model')}  temperature {first.get('temperature')}")
    print(f"roles {[m.get('role') for m in msgs]}")
    print(f"system prompt {len(system)} chars, ~{len(system.split())} words")
    print(f"tools ({len(tools)}): {', '.join(t for t in tools if t)}")
    print("\n--- system prompt" + ("" if args.full else " (first 3000 chars; --full for all)") + " ---")
    print(system if args.full else system[:3000])
    print("\n--- reply ---")
    print(proc.stdout.strip()[-1500:])

    if args.save:
        Path(args.save).write_text(json.dumps(_Recorder.captured, indent=2), encoding="utf-8")
        print(f"\nsaved to {args.save}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
