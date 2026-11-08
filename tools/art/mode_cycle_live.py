#!/usr/bin/env python3
"""Cycle voice -> art -> voice on the real machine, timing each switch (AVATAR_PLAN 6a step 3).

Why: the unit tests use a fake system. This proves on the box that art mode
stops the coder, starts ComfyUI under its memory cap, keeps Jarvis listening
throughout, and that voice mode brings the coder back healthy. It never talks
to the voice agent, so nothing in the house can act.

Writes ~/homeai-bench/mode-cycle-<time>/result.json.

Usage:
    .venv/bin/python tools/art/mode_cycle_live.py [--cycles 1]
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

from homeai import gpu_mode as G  # noqa: E402
from homeai.benchio import bench_path  # noqa: E402


def unit_property(unit: str, name: str) -> str:
    proc = subprocess.run(["systemctl", "--user", "show", "-p", name, "--value", unit],
                          capture_output=True, text=True, check=False)
    return proc.stdout.strip()


def snapshot(switcher: G.ModeSwitcher) -> dict:
    st = switcher.status()
    return {"mode": st.mode, "jarvis": st.jarvis, "coder_up": st.coder_up, "comfy_up": st.comfy_up,
            "voice_loaded": switcher.voice_loaded(), "vram_used_gb": round(st.vram_used_gb, 2)}


def timed(step, switcher: G.ModeSwitcher) -> dict:
    start = time.monotonic()
    error = ""
    try:
        step()
    except G.ModeError as exc:
        error = str(exc)
    return {"seconds": round(time.monotonic() - start, 1), "error": error, **snapshot(switcher)}


def problems(result: dict) -> list[str]:
    """Everything that must hold after each switch, as plain sentences; empty means it passed."""
    found = []
    art, voice = result["art"], result["voice"]
    if art["error"] or voice["error"]:
        found.append(f"a switch failed: {art['error'] or voice['error']}")
    if art["mode"] != "art" or not art["comfy_up"] or art["coder_up"]:
        found.append("art mode did not swap the coder for ComfyUI")
    if result["memory_max"] != str(24 * 1024 ** 3) or result["swap_max"] != "0":
        found.append(f"ComfyUI was not capped (MemoryMax {result['memory_max']}, MemorySwapMax {result['swap_max']})")
    if not (art["jarvis"] and voice["jarvis"]):
        found.append("Jarvis was not running throughout")
    if voice["mode"] != "voice" or voice["comfy_up"] or not voice["coder_up"] or not voice["voice_loaded"]:
        found.append("voice mode did not come back whole")
    return found


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--cycles", type=int, default=1)
    args = ap.parse_args()
    switcher = G.ModeSwitcher(G.System(), log=lambda line: print("  ", line, flush=True))
    if switcher.mode() != "voice":
        print(f"start in voice mode (now {switcher.mode()})", file=sys.stderr)
        return 1
    out = bench_path("mode-cycle")
    out.mkdir(parents=True)
    runs = []
    for n in range(args.cycles):
        print(f"cycle {n + 1}: art", flush=True)
        result = {"before": snapshot(switcher), "art": timed(switcher.art, switcher)}
        result["memory_max"] = unit_property(G.COMFY_UNIT, "MemoryMax")
        result["swap_max"] = unit_property(G.COMFY_UNIT, "MemorySwapMax")
        print(f"cycle {n + 1}: voice", flush=True)
        result["voice"] = timed(switcher.voice, switcher)
        result["problems"] = problems(result)
        runs.append(result)
        print(json.dumps(result, indent=1))
    (out / "result.json").write_text(json.dumps(runs, indent=1))
    print(out)
    return 1 if any(r["problems"] for r in runs) else 0


if __name__ == "__main__":
    raise SystemExit(main())
