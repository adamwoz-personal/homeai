#!/usr/bin/env python3
# Copyright (c) 2026 Adam Wosotowsky
# SPDX-License-Identifier: BUSL-1.1
# Licensed under the Business Source License 1.1; see LICENSE at the
# repository root. Converts to Apache-2.0 on 2030-09-15.
"""Walk the model requests saved by ``probe_agent_prompt.py --save``.

``probe_agent_prompt.py`` prints the first request's system prompt and tool
list. The interesting part is often *later*: which tool the model called
and what came back. On 2026-10-02 the second request showed the model
calling ``memory_recall`` and receiving a month-old hedged answer, which it
then repeated.

Usage
-----
    inspect_captured_prompt.py /tmp/capture.json            # every request, no system text
    inspect_captured_prompt.py /tmp/capture.json --system   # include system prompts
    inspect_captured_prompt.py /tmp/capture.json --request 2 --width 4000
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("capture")
    ap.add_argument("--request", type=int, help="only this request (1-based)")
    ap.add_argument("--system", action="store_true", help="also print system messages")
    ap.add_argument("--width", type=int, default=1500, help="characters per message")
    args = ap.parse_args(argv)

    captured = json.loads(Path(args.capture).expanduser().read_text(encoding="utf-8"))
    requests = [c["body"] for c in captured
                if isinstance(c.get("body"), dict) and "messages" in c["body"]]
    if not requests:
        print("no model requests in capture", file=sys.stderr)
        return 1

    for index, body in enumerate(requests, start=1):
        if args.request and index != args.request:
            continue
        tools = [t.get("function", {}).get("name") for t in body.get("tools") or []]
        print(f"===== request {index}/{len(requests)}  model={body.get('model')}  "
              f"tools={len(tools)}")
        for msg in body["messages"]:
            role = msg.get("role")
            if role == "system" and not args.system:
                print(f"[system: {len(msg.get('content') or '')} chars hidden]")
                continue
            for call in msg.get("tool_calls") or []:
                fn = call.get("function", {})
                print(f"{role} CALLS {fn.get('name')}({fn.get('arguments')})")
            content = msg.get("content") or ""
            if content:
                print(f"{role}: {content[: args.width]}")
            print()
    return 0


if __name__ == "__main__":
    sys.exit(main())
