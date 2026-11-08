#!/usr/bin/env python3
"""Art mode by voice, end to end on the box (AVATAR_PLAN 6a step 8 functionality test).

Why: the unit tests use a fake switcher, queue and window. This types sentences
into the real controller (homeai/art/voice.py): it switches the card to art mode,
opens the real picture window, draws with ComfyUI, saves to a scratch Pictures
folder, and leaves art mode, then checks Jarvis and the coder are back. No
microphone, no speech, no agent: the voice model is only called by the patcher,
without tools, so nothing in the house can act.

The window listens on a private event bus, not the daemon's, so the running
Jarvis is undisturbed. The window saves a screenshot of itself after each new picture.

Writes ~/homeai-bench/art-voice-<time>/{result.json,window-*.png,Pictures/}.

Usage:
    .venv/bin/python tools/art/art_voice_live.py
"""
from __future__ import annotations

import json
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

from homeai import gpu_mode as G  # noqa: E402
from homeai.art import checker as CK  # noqa: E402
from homeai.art import voice as AV  # noqa: E402
from homeai.avatar.bus import EventBus  # noqa: E402

SCRIPT = ("switch to art mode", "draw a samurai with red armor riding a dragon", "make the dragon green",
          "again", "turn off the lights", "I'm done drawing")


def unit_active(unit: str) -> bool:
    return subprocess.run(["systemctl", "--user", "is-active", "--quiet", unit]).returncode == 0


def main() -> int:
    switcher = G.ModeSwitcher(G.System(), log=lambda line: print("  ", line, flush=True))
    if switcher.mode() == "coding":
        print("start in voice or art mode, not coding", file=sys.stderr)
        return 1
    out = Path.home() / "homeai-bench" / f"art-voice-{datetime.now():%Y%m%d-%H%M%S}"
    out.mkdir(parents=True)
    bus = EventBus(out / "events.sock", source="art-voice-live")
    bus.start()
    left: list[float] = []

    def background(fn):                                      # leave synchronously, and time it
        start = time.monotonic()
        fn()
        left.append(round(time.monotonic() - start, 1))

    asked: list[str] = []

    def checker(image, question):                           # the real vision model, counted
        asked.append(question)
        return CK.ollama_ask(image, question)
    art = AV.ArtVoice(switcher, AV.build_studio, bus.publish, viewer=AV.Viewer(socket=str(out / "events.sock"), shots=str(out / "window")),
                      pictures=out / "Pictures", background=background, checker=checker)
    steps, checks, shots, saved = [], {}, 0, []
    project_ids: set[str] = set()                            # only this run's pictures count (a leftover job may finish)
    try:
        for said in SCRIPT:
            start = time.monotonic()
            spoken = art.handle(said, "Adam")
            reply_s = round(time.monotonic() - start, 2)
            jobs = list(art.session.jobs) if art.session else []

            def busy():                                      # redraws after a failed check add jobs later
                live = list(art.session.jobs) if art.session else []
                return art.queue and (any((j := art.queue.get(job)) and j.state not in ("done", "failed", "cancelled")
                                          for job in live) or art.status.startswith(("Redrawing", "Checking", "Drawing")))
            while busy():
                time.sleep(0.2)
            done_s = round(time.monotonic() - start, 2)
            row = {"said": said, "spoken": spoken, "reply_s": reply_s, "done_s": done_s,
                   "viewer_up": unit_active(AV.VIEWER_UNIT)}
            if spoken and jobs and art.active:
                time.sleep(1.5)                              # the window polls at 10 fps
                shots += 1
            if art.session and art.session.project:
                project_ids.add(art.session.project.id)
            steps.append(row)
            print(f"{said!r:58} -> {spoken!r} ({reply_s}s, done {done_s}s)", flush=True)
        saved = sorted(p.name for p in (out / "Pictures").rglob("*.png"))
        checks["entered by voice"] = steps[0]["spoken"] == AV.MSG_READY
        checks["window open in art mode"] = all(s["viewer_up"] for s in steps[:5])
        ours = [n for n in saved if any(n.startswith(pid) for pid in project_ids)]
        checks["three pictures saved"] = len(ours) == 3
        checks["window showed each picture"] = len(list((out / "window").glob("window-*.png"))) >= shots
        checks["features window drawn"] = any((out / "window").glob("features-*.png"))
        adam = [p for p in (out / "Pictures" / "adam").glob("*") if any(p.name.startswith(pid) for pid in project_ids)]
        checks["three pictures saved for Adam"] = len([p for p in adam if p.suffix == ".png"]) == 3
        checks["each with its description"] = len([p for p in adam if p.suffix == ".json"]) == 3
        checks["house talk passed on"] = steps[4]["spoken"] is None
        checks["left by voice"] = steps[5]["spoken"] == AV.MSG_LEFT
        checks["window closed"] = not unit_active(AV.VIEWER_UNIT)
        checks["back in voice mode"] = switcher.mode() == "voice"
        checks["coder healthy"] = switcher.coder_healthy()
        checks["pictures checked"] = len(asked) >= 6               # heads and riders, three pictures at least
        checks["voice model back"] = switcher.voice_loaded()
        checks["Jarvis up"] = switcher.status().jarvis
    finally:
        if art.active:
            art.leave("probe cleanup")
        if switcher.mode() != "voice":
            switcher.voice()
        bus.stop()
    (out / "result.json").write_text(json.dumps({"checks": checks, "steps": steps, "leave_s": left,
                                                 "saved": saved}, indent=1))
    print(json.dumps(checks, indent=1))
    print(out)
    return 0 if checks and all(checks.values()) else 1


if __name__ == "__main__":
    raise SystemExit(main())
