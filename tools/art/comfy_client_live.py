#!/usr/bin/env python3
"""Check homeai/art/comfy.py against the real ComfyUI in art mode (AVATAR_PLAN 6a step 4).

Why: the unit tests use a fake server, and the spoken error sentences depend on
ComfyUI's real wording. This switches to art mode, draws one klein 4B picture
through ComfyClient, sends a graph naming a model that isn't installed and a
graph with a broken link, checks each is told apart, and goes back to voice
mode. It never talks to the voice agent.

Writes ~/homeai-bench/comfy-client-<time>/result.json.

Usage:
    .venv/bin/python tools/art/comfy_client_live.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "tools" / "art"))

from comfy_common import workflow  # noqa: E402

from homeai import gpu_mode as G  # noqa: E402
from homeai.art import comfy as C  # noqa: E402
from homeai.benchio import bench_path  # noqa: E402


def attempt(client: C.ComfyClient, graph: dict) -> dict:
    try:
        result = client.run(graph, timeout=300)
        return {"image": str(result.image), "exists": result.image.exists(), "seconds": result.seconds}
    except C.ComfyError as exc:
        return {"spoken": exc.spoken, "detail": exc.detail[:300]}


def main() -> int:
    switcher = G.ModeSwitcher(G.System(), log=lambda line: print("  ", line, flush=True))
    if switcher.mode() != "voice":
        print(f"start in voice mode (now {switcher.mode()})", file=sys.stderr)
        return 1
    out = bench_path("comfy-client")
    out.mkdir(parents=True)
    good = workflow("klein", "a red fox sitting in snow", 7, 512, 512)
    missing = json.loads(json.dumps(good))
    for node in missing.values():
        for key in ("unet_name", "ckpt_name"):
            if key in node["inputs"]:
                node["inputs"][key] = "not-installed.safetensors"
    broken = json.loads(json.dumps(good))
    next(iter(broken.values()))["inputs"]["bogus_link"] = ["no-such-node", 0]
    results = {}
    try:
        switcher.art()
        client = C.ComfyClient()
        results = {"good": attempt(client, good), "missing_model": attempt(client, missing),
                   "broken": attempt(client, broken)}
    finally:
        switcher.voice()
    checks = {"good drew a picture": results.get("good", {}).get("exists") is True,
              "missing model named": results.get("missing_model", {}).get("spoken") == C.MSG_MISSING_MODEL,
              "broken graph refused": results.get("broken", {}).get("spoken") in (C.MSG_BAD_GRAPH, C.MSG_FAILED)}
    (out / "result.json").write_text(json.dumps({"results": results, "checks": checks}, indent=1))
    print(json.dumps({"results": results, "checks": checks}, indent=1))
    print(out)
    return 0 if all(checks.values()) else 1


if __name__ == "__main__":
    raise SystemExit(main())
