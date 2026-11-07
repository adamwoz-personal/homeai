#!/usr/bin/env python3
"""A scripted drawing session against the real voice model and ComfyUI (AVATAR_PLAN 6a functionality test).

Why: each piece has unit tests with fakes; this proves they work together on
the box: grammar -> voice-model patch -> policy -> queue -> ComfyUI -> store.
It types the sentences (no microphone, no speech) and calls the voice model
only through the patcher (no tools, so nothing in the house can act). It
switches to art mode, runs the script, goes back to voice mode, and writes a
contact sheet of every picture with its sentence for Adam to look at.

Writes ~/homeai-bench/art-session-<time>/{result.json,sheet.png}; projects go
to a scratch store there, not the real gallery.

Usage:
    .venv/bin/python tools/art/art_session_live.py
"""
from __future__ import annotations

import json
import sys
import time
from datetime import datetime
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

from homeai import gpu_mode as G  # noqa: E402
from homeai.art import comfy as C  # noqa: E402
from homeai.art import jobs as J  # noqa: E402
from homeai.art.session import ArtSession  # noqa: E402
from homeai.art.store import Store  # noqa: E402

SCRIPT = ("draw a red fox sitting in the snow", "make it blue", "add some fog", "again", "go back two",
          "make it a watercolour", "keep that", "what prompt made this", "turn off the lights")


def sheet(rows: list[tuple[str, Path | None]], path: Path) -> None:
    from PIL import Image, ImageDraw

    cell, label = 320, 40
    pictures = [(text, image) for text, image in rows if image]
    canvas = Image.new("RGB", (cell * max(1, len(pictures)), cell + label), "white")
    draw = ImageDraw.Draw(canvas)
    for i, (text, image) in enumerate(pictures):
        thumb = Image.open(image).convert("RGB")
        thumb.thumbnail((cell, cell))
        canvas.paste(thumb, (i * cell, label))
        draw.text((i * cell + 4, 4), text[:44], fill="black")
    canvas.save(path)


def main() -> int:
    switcher = G.ModeSwitcher(G.System(), log=lambda line: print("  ", line, flush=True))
    if switcher.mode() != "voice":
        print(f"start in voice mode (now {switcher.mode()})", file=sys.stderr)
        return 1
    out = Path.home() / "homeai-bench" / f"art-session-{datetime.now():%Y%m%d-%H%M%S}"
    store = Store(out / "projects")
    steps, checks = [], {}
    try:
        switcher.art()
        queue = J.JobQueue(C.ComfyClient(), store, path=out / "queue.json", restart=switcher.restart_comfy)
        session = ArtSession(store, queue, "probe")
        queue._notify = lambda event: session.on_event(event)
        queue.start()
        for said in SCRIPT:
            start = time.monotonic()
            reply = session.handle(said)
            reply_s = round(time.monotonic() - start, 2)
            for job in reply.jobs if reply else []:
                while (state := queue.get(job).state) not in J.FINAL:
                    time.sleep(0.2)
            done_s = round(time.monotonic() - start, 2)
            node = session._current()
            steps.append({"said": said, "kind": reply.kind if reply else None, "spoken": reply.spoken if reply else None,
                          "reply_s": reply_s, "done_s": done_s, "node": node.id if node else None,
                          "image": str(session.project.image(node.id)) if node and reply and reply.jobs else None})
            print(f"{said!r:40} -> {steps[-1]['spoken']!r} ({reply_s}s, done {done_s}s)", flush=True)
        queue.stop()
        project = store.open(session.project.id)
        checks["every picture drawn"] = all(Path(s["image"]).exists() for s in steps if s["image"])
        checks["edit kept the seed"] = project.nodes["002"].spec["seed"] == project.nodes["001"].spec["seed"]
        checks["edit changed only the palette"] = project.nodes["002"].spec["palette"] != "" and \
            project.nodes["002"].spec["subject"] == project.nodes["001"].spec["subject"]
        checks["went back two"] = steps[4]["node"] == "001"     # again (004) is a sibling of fog: 004 -> 002 -> 001
        checks["branch from there"] = project.nodes[steps[5]["node"]].parent == "001"
        checks["kept"] = project.info["pins"] == [steps[5]["node"]]
        checks["house talk passed on"] = steps[-1]["kind"] is None
        checks["voice model still loaded"] = switcher.voice_loaded()
        checks["Jarvis up"] = switcher.status().jarvis
    finally:
        switcher.voice()
    out.mkdir(parents=True, exist_ok=True)
    sheet([(s["said"], Path(s["image"])) for s in steps if s["image"] and Path(s["image"]).exists()], out / "sheet.png")
    (out / "result.json").write_text(json.dumps({"checks": checks, "steps": steps}, indent=1))
    print(json.dumps(checks, indent=1))
    print(out)
    return 0 if checks and all(checks.values()) else 1


if __name__ == "__main__":
    raise SystemExit(main())
