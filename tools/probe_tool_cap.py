#!/usr/bin/env python3
# Copyright (c) 2026 Adam Wosotowsky
# SPDX-License-Identifier: BUSL-1.1
"""Does an agent really get more than 10 tool calls per request?

ZeroClaw's default max_tool_iterations is 10; `runtime_profile = "heavy_duty"`
(max_tool_iterations = 100) is supposed to lift it for the builder agent.
This proves it with a chain the agent cannot shortcut by reading ahead: a
local HTTP server where each page only reveals the URL of the next, and the
final page holds a code word. Every request is logged, with its User-Agent,
ZeroClaw's web_fetch refuses localhost (SSRF guard), so the agent fetches
with `curl` through its shell tool. A shell loop (one tool call doing all the
fetching) is detected by timing: separate tool calls are at least one model
turn apart (well over 0.3s), a loop's requests are milliseconds apart. That
run is reported INCONCLUSIVE rather than PASS.

    tools/probe_tool_cap.py                    # builder, 15 steps
    tools/probe_tool_cap.py --agent builder --steps 25
"""

from __future__ import annotations

import argparse
import http.server
import secrets
import subprocess
import sys
import threading
import time


def make_handler(steps: int, tokens: list[str], codeword: str, log: list[tuple[float, int, str]]):
    class Handler(http.server.BaseHTTPRequestHandler):
        def log_message(self, *args):
            pass

        def do_GET(self):
            parts = self.path.strip("/").split("/")
            step = -1
            if len(parts) == 2 and parts[0] == "step" and parts[1] in tokens:
                step = tokens.index(parts[1])
            log.append((time.monotonic(), step, self.headers.get("User-Agent", "")))
            if step < 0:
                body = "Not found. Only follow links given to you."
                self.send_response(404)
            elif step + 1 < steps:
                body = (f"Step {step + 1} of {steps}. Not done yet. "
                        f"Next, fetch: http://127.0.0.1:{self.server.server_port}/step/{tokens[step + 1]}")
                self.send_response(200)
            else:
                body = f"Final step {steps} of {steps}. The code word is: {codeword}"
                self.send_response(200)
            self.send_header("Content-Type", "text/plain")
            self.end_headers()
            self.wfile.write(body.encode())

    return Handler


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--agent", default="builder")
    ap.add_argument("--steps", type=int, default=15)
    ap.add_argument("--timeout", type=float, default=900)
    args = ap.parse_args(argv)

    tokens = [secrets.token_hex(4) for _ in range(args.steps)]
    codeword = "zebra-" + secrets.token_hex(3)
    log: list[tuple[float, int, str]] = []
    server = http.server.ThreadingHTTPServer(("127.0.0.1", 0),
                                             make_handler(args.steps, tokens, codeword, log))
    threading.Thread(target=server.serve_forever, daemon=True).start()
    start_url = f"http://127.0.0.1:{server.server_port}/step/{tokens[0]}"
    prompt = (
        f"Run `curl -s {start_url}` with your shell tool. Each page gives the next URL. "
        "Keep fetching with curl, ONE shell tool call per page, until a page gives you the "
        "code word. Do not write scripts or loops. Reply with only the code word."
    )
    t0 = time.monotonic()
    try:
        proc = subprocess.run(["zeroclaw", "agent", "-a", args.agent, "-m", prompt],
                              capture_output=True, text=True, timeout=args.timeout)
        reply = proc.stdout + proc.stderr
    except subprocess.TimeoutExpired:
        reply = "(timed out)"
    elapsed = time.monotonic() - t0
    server.shutdown()

    reached = max((s for _, s, _ in log), default=-1) + 1
    agents = {ua.split("/")[0] for _, s, ua in log if s >= 0}
    times = sorted(t for t, s, _ in log if s >= 0)
    gaps = sorted(b - a for a, b in zip(times, times[1:]))
    median_gap = gaps[len(gaps) // 2] if gaps else 0.0
    looped = len(gaps) >= 3 and median_gap < 0.3
    got_word = codeword in reply
    print(f"agent {args.agent}: reached step {reached}/{args.steps}, {len(log)} requests, "
          f"{elapsed:.0f}s, median gap {median_gap:.2f}s, user-agents {sorted(agents) or '-'}")
    print(f"reply tail: {reply.strip()[-300:]}")
    if got_word and reached == args.steps and not looped:
        print(f"PASS: {args.steps} sequential tool calls in one request (cap is above 10)")
        return 0
    if looped:
        print("INCONCLUSIVE: the agent fetched with a script/loop, not one tool call per step")
        return 2
    if reached == 10:
        print("FAIL: stopped at exactly 10 -- the 10-tool-call cap still applies")
    else:
        print("FAIL: chain not completed")
    return 1


if __name__ == "__main__":
    sys.exit(main())
