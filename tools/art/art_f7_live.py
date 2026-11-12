#!/usr/bin/env python3
"""ART_FEATURES_PLAN F7 against the real voice model and ComfyUI: last N, compare, upscale, save by name.

Why: the unit tests use fakes; this runs the workflow Adam agreed on (2026-11-12) end to end on the box:
draw -> four options -> pick -> edits -> "show me the last four" -> "compare two and four" -> pick ->
"upscale it" -> "save that as Beacon" (twice: Beacon 2). It also retries Adam's deck sentence that failed
that afternoon. Sentences are typed into ArtVoice (no microphone); the voice model is reached only through
the patcher (no tools, so nothing in the house can act). Projects and pictures go to a scratch folder, not
the real gallery; no picture window is opened.

Leaves the box in voice mode. Writes ~/homeai-bench/art-f7-<time>/{result.json,sheet.png}.

Usage:
    .venv/bin/python tools/art/art_f7_live.py
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

from homeai import gpu_mode as G  # noqa: E402
from homeai.art import comfy as C  # noqa: E402
from homeai.art import jobs as J  # noqa: E402
from homeai.art import voice as AV  # noqa: E402
from homeai.art.session import ArtSession  # noqa: E402
from homeai.art.store import Store  # noqa: E402
from homeai.benchio import bench_path  # noqa: E402

DECK = ("let's make assembly instructions for an 8 foot by 8 foot deck that is on the ground that would be "
        "used to put a tent on top of")
SCRIPT = (DECK, "start a new picture: a lighthouse on a rocky coast at sunset", "give me four", "the second one",
          "make it night", "add some fog", "show me the last four", "compare two and four", "number two",
          "upscale it", "save that as Beacon", "save that as Beacon")


class NoWindow:
    def start(self):
        pass

    def stop(self):
        pass


def main() -> int:
    from PIL import Image

    switcher = G.ModeSwitcher(G.System(), log=lambda line: print("  ", line, flush=True))
    out = bench_path("art-f7")
    pictures = out / "Pictures"
    published: list[dict] = []
    built = {}

    def build(owner, notify):
        store = Store(out / "projects")
        queue = J.JobQueue(C.ComfyClient(), store, path=out / "queue.json", notify=notify,
                           restart=switcher.restart_comfy)
        built["queue"] = queue
        return ArtSession(store, queue, owner), queue

    art = AV.ArtVoice(switcher, build, lambda etype, data: published.append(data) if etype == "art" else None,
                      viewer=NoWindow(), pictures=pictures, background=lambda fn: fn(), idle_s=3600)
    art._start_idle_watch = lambda: None
    steps, checks = [], {}
    try:
        art.handle("let's draw", "Adam")
        for said in SCRIPT:
            start = time.monotonic()
            spoken = art.handle(said, "Adam")
            reply_s = round(time.monotonic() - start, 2)
            queue = built["queue"]
            for job in list(art.session.jobs):
                while (got := queue.get(job)) is not None and got.state not in J.FINAL:
                    time.sleep(0.2)
            state = published[-1] if published else {}
            node = art.session._current()
            steps.append({"said": said, "spoken": spoken, "reply_s": reply_s, "done_s": round(time.monotonic() - start, 2),
                          "project": art.session.project.id if art.session.project else None,
                          "node": node.id if node else None, "compare": state.get("compare"),
                          "options": [(o["n"], o.get("project"), o["node"]) for o in state.get("options") or []]})
            print(f"{said[:60]!r:62} -> {spoken!r} ({reply_s}s, done {steps[-1]['done_s']}s)", flush=True)
        by = {s["said"]: s for s in steps}
        last4, pair = by["show me the last four"], by["compare two and four"]
        checks["deck sentence not answered as a change"] = "stays as it was" not in (steps[0]["spoken"] or "")
        checks["last four on a sheet"] = len(last4["options"]) == 4 and not last4["compare"]
        checks["compare shows 2 and 4"] = [o[0] for o in pair["options"]] == [2, 4] and pair["compare"] is True
        picked = [o for o in last4["options"] if o[0] == 2][0]
        checks["number two picks sheet 2"] = (by["number two"]["project"], by["number two"]["node"]) == picked[1:]
        project, node = by["upscale it"]["project"], by["upscale it"]["node"]
        original = pictures / "adam" / AV.export_name(project, node)
        bigger = pictures / "adam" / f"{project}-{node}-2x.png"
        if original.exists() and bigger.exists():
            with Image.open(original) as a, Image.open(bigger) as b:
                checks["2x copy is twice the size"] = (b.width, b.height) == (2 * a.width, 2 * a.height)
        else:
            checks["2x copy is twice the size"] = False
        checks["upscale kept the picture current"] = by["upscale it"]["node"] == by["number two"]["node"]
        checks["saved as beacon and beacon-2"] = (pictures / "adam" / "beacon.png").exists() and \
            (pictures / "adam" / "beacon-2.png").exists() and (by["save that as Beacon"]["spoken"] or "").lower() == "saved as beacon 2."
        missing = [p.name for p in pictures.rglob("*.png")
                   if not p.with_suffix(".json").exists()
                   or not all(json.loads(p.with_suffix(".json").read_text()).get(k) for k in ("prompt", "seed", "model"))]
        checks["every saved picture has prompt, seed and model"] = not missing and any(pictures.rglob("*.png"))
        checks["voice model still loaded"] = switcher.voice_loaded()
    finally:
        try:
            art.leave("probe done")
        finally:
            if switcher.mode() != "voice":
                switcher.voice()
    rows = sorted(pictures.rglob("*.png"), key=lambda p: p.stat().st_mtime)
    if rows:
        cell = 256
        sheet = Image.new("RGB", (cell * min(8, len(rows)), cell * ((len(rows) + 7) // 8)), "white")
        for i, path in enumerate(rows):
            with Image.open(path) as im:
                im = im.convert("RGB")
                im.thumbnail((cell, cell))
                sheet.paste(im, ((i % 8) * cell, (i // 8) * cell))
        sheet.save(out / "sheet.png")
    (out / "result.json").write_text(json.dumps({"checks": checks, "steps": steps}, indent=1))
    print(json.dumps(checks, indent=1))
    print(out)
    return 0 if checks and all(checks.values()) else 1


if __name__ == "__main__":
    raise SystemExit(main())
