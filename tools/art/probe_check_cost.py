#!/usr/bin/env python3
"""What does checking each picture with a vision model cost in art mode? (ART_FEATURES_PLAN F4, step 2)

Spike 1 showed minicpm-v can count heads and riders (15/16, 16/16). Before
building "draw, check, redraw", this measures the price on this box:

- Ollama holds one model (OLLAMA_MAX_LOADED_MODELS=1, deliberate), so a check
  pushes Jarvis's voice model out: how long does the vision model take to load
  and answer, and how long does the voice model take to come back?
- Does klein in ComfyUI plus the vision model fit in 20 GB, and does drawing
  slow down while the vision model is loaded? Peak VRAM is sampled every 0.2 s.

Each round: draw (voice model loaded), check (vision model loads), draw again
(vision model loaded), check (warm), reload the voice model. Run when nobody is
talking to Jarvis: it switches to art mode if needed (the coder stops) and back.

Writes ~/homeai-bench/check-cost-<time>/result.json.

    .venv/bin/python tools/art/probe_check_cost.py [--rounds 3] [--model minicpm-v]
"""
from __future__ import annotations

import argparse
import json
import statistics
import sys
import threading
import time
import urllib.request
from datetime import datetime
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

from homeai import gpu_mode as G  # noqa: E402
from homeai.art import comfy as C  # noqa: E402
from homeai.art import workflows  # noqa: E402
from homeai.art.patcher import MODEL as VOICE_MODEL  # noqa: E402
from tools.art.probe_vision_check import ask  # noqa: E402

from homeai.httpjson import OLLAMA  # noqa: E402
PROMPT = ("A single red dragon with one head and four legs, flying. One samurai in full armor sitting on the "
          "dragon's back. Over the ocean.")
QUESTION = "How many heads does the dragon have? Answer with a number only."


class PeakVram:
    def __init__(self, system) -> None:
        self.system, self.peak, self._stop = system, 0.0, threading.Event()
        self._thread = threading.Thread(target=self._run, daemon=True)

    def _run(self) -> None:
        while not self._stop.is_set():
            self.peak = max(self.peak, self.system.vram_gb()[1])
            time.sleep(0.2)

    def __enter__(self):
        self._thread.start()
        return self

    def __exit__(self, *exc) -> None:
        self._stop.set()
        self._thread.join(timeout=2)


def loaded() -> list[str]:
    with urllib.request.urlopen(f"{OLLAMA}/api/ps", timeout=10) as reply:
        return [m["name"] for m in json.loads(reply.read())["models"]]


def reload_voice() -> None:
    """The same request the art patcher makes, pinned the same way (keep_alive -1)."""
    payload = {"model": VOICE_MODEL, "messages": [{"role": "user", "content": "Say ok."}], "stream": False,
               "keep_alive": -1, "options": {"temperature": 0, "num_predict": 2}}
    req = urllib.request.Request(f"{OLLAMA}/api/chat", data=json.dumps(payload).encode(),
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=300) as reply:
        reply.read()


def timed(fn):
    start = time.monotonic()
    value = fn()
    return round(time.monotonic() - start, 2), value


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--rounds", type=int, default=3)
    ap.add_argument("--model", default="minicpm-v")
    args = ap.parse_args()
    system = G.System()
    switcher = G.ModeSwitcher(system, log=lambda line: print("  ", line, flush=True))
    if switcher.mode() == "coding":
        print("not from coding mode", file=sys.stderr)
        return 1
    started = switcher.mode() != "art"
    out = Path.home() / "homeai-bench" / f"check-cost-{datetime.now():%Y%m%d-%H%M%S}"
    out.mkdir(parents=True)
    rows = []
    try:
        if started:
            switcher.art()
        client = C.ComfyClient()
        reload_voice()
        client.run(workflows.klein9(PROMPT, 99, 1024, 1024, prefix="bench/check-cost"))     # warm klein
        for n in range(args.rounds):
            row = {"round": n, "vram_start": round(system.vram_gb()[1], 2), "loaded_start": loaded()}
            with PeakVram(system) as peak:
                row["draw_voice_loaded_s"], result = timed(lambda: client.run(
                    workflows.klein9(PROMPT, 100 + n, 1024, 1024, prefix="bench/check-cost")))
            row["vram_peak_draw_voice"] = round(peak.peak, 2)
            with PeakVram(system) as peak:
                row["check_cold_s"], row["answer_cold"] = timed(lambda: ask(args.model, Path(result.image), QUESTION))
            row["vram_peak_check"] = round(peak.peak, 2)
            row["loaded_after_check"] = loaded()
            with PeakVram(system) as peak:
                row["draw_vlm_loaded_s"], result = timed(lambda: client.run(
                    workflows.klein9(PROMPT, 200 + n, 1024, 1024, prefix="bench/check-cost")))
            row["vram_peak_draw_vlm"] = round(peak.peak, 2)
            row["check_warm_s"], row["answer_warm"] = timed(lambda: ask(args.model, Path(result.image), QUESTION))
            row["voice_reload_s"], _ = timed(reload_voice)
            row["loaded_end"] = loaded()
            rows.append(row)
            print(json.dumps(row), flush=True)
    finally:
        try:
            reload_voice()
        finally:
            if started:
                switcher.voice()
    keys = [k for k in rows[0] if k.endswith("_s") or k.startswith("vram")] if rows else []
    summary = {k: {"median": statistics.median(r[k] for r in rows), "max": max(r[k] for r in rows)} for k in keys}
    summary["vram_total"] = round(system.vram_gb()[0], 2)
    (out / "result.json").write_text(json.dumps({"model": args.model, "summary": summary, "rows": rows}, indent=1))
    print(json.dumps(summary, indent=1))
    print(out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
