#!/usr/bin/env python3
"""Look, notice, reword, end to end on the box (ART_FEATURES_PLAN F8 functionality test).

Why: the unit tests fake the image model, the checker and the rewriter. This types Adam's 2026-11-18 sentence into
the real art controller (homeai/art/voice.py) with the real ComfyUI (klein9), the real checker and the real rewriter
(both Jarvis's own voice model), and records every try: the prompt sent, what the check found, any rewrite, and
which picture would have been shown. Then a contact sheet to judge by eye.

Starts art mode if it is off and goes back to voice after (leaves it on if it was on). Opens no window, speaks nothing and writes to a
scratch store, so a running Jarvis is undisturbed (it shares the GPU and the voice model for a few minutes). The
voice model is called without tools, so nothing in the house can act.

Writes ~/homeai-bench/f8-loop-<time>/{result.json,sheet.png,projects/,Pictures/}.

Usage:
    .venv/bin/python tools/art/live_f8_loop.py [--rounds 3] [--said "draw ..."]
"""
from __future__ import annotations

import argparse
import json
import sys
import time
import urllib.request
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

from homeai.art import checker as CK  # noqa: E402
from homeai.art import patcher as P  # noqa: E402
from homeai.art import voice as AV  # noqa: E402
from homeai.benchio import bench_path  # noqa: E402
from tools.art.bench_rider import sheet  # noqa: E402

SAID = "draw an anime girl wielding a katana fighting a chimera"


class StaySwitcher:
    """Art mode is already on (ComfyUI up); never switch anything."""

    def mode(self):
        return "art"

    def art(self):
        pass

    def voice(self):
        pass


class NoViewer:
    def start(self):
        pass

    def stop(self):
        pass


def comfy_up() -> bool:
    try:
        with urllib.request.urlopen("http://127.0.0.1:8188/queue", timeout=2):
            return True
    except OSError:
        return False


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n", 1)[0])
    ap.add_argument("--rounds", type=int, default=3, help="the sentence, then 'again' this many times minus one")
    ap.add_argument("--said", default=SAID)
    args = ap.parse_args(argv)
    from homeai import gpu_mode as G
    modes = G.ModeSwitcher(G.System(), log=lambda line: print("  ", line, flush=True))
    if modes.mode() == "coding":
        print("not from coding mode", file=sys.stderr)
        return 1
    started = modes.mode() != "art"
    if started:
        modes.art()                      # and back to voice at the end, as tools/art/bench_rider.py does
    try:
        return run(args)
    finally:
        if started:
            modes.voice()


def run(args) -> int:
    if not comfy_up():
        print("ComfyUI isn't up", file=sys.stderr)
        return 1
    out = bench_path("f8-loop")
    out.mkdir(parents=True)
    from homeai.art import comfy, jobs
    from homeai.art.session import ArtSession
    from homeai.art.store import Store

    def build(owner, notify):
        store = Store(out / "projects")
        queue = jobs.JobQueue(comfy.ComfyClient(), store, notify=notify, restart=lambda: None)
        return ArtSession(store, queue, owner), queue

    asked, rewrites = [], []

    def checker(image, question):
        answer = CK.ollama_ask(image, question)
        asked.append({"image": Path(image).name, "question": question, "answer": answer})
        return answer

    def reword_chat(messages, schema):
        answer = P.ollama_chat(messages, schema)
        kind = "questions" if "questions" in schema.get("properties", {}) else "reword"
        rewrites.append({"kind": kind, "asked": messages[-1]["content"], "answer": answer})
        return answer

    art = AV.ArtVoice(StaySwitcher(), build, lambda kind, data: None, viewer=NoViewer(), pictures=out / "Pictures",
                      background=lambda fn: fn(), checker=checker, reword_chat=reword_chat)
    art._start_idle_watch = lambda: None
    start = time.monotonic()
    spoken = []
    try:
        for i in range(args.rounds):
            if i:
                art.handle("start a new picture", "Adam")
            spoken.append(art.handle(args.said, "Adam"))
            time.sleep(0.5)
            while art.queue and (any((j := art.queue.get(job)) and j.state not in ("done", "failed", "cancelled")
                                     for job in list(art.session.jobs)) or
                                 art.status.startswith(("Redrawing", "Checking", "Drawing"))):
                time.sleep(0.3)
            print(f"round {i + 1}: {spoken[-1]!r} -> {art.status!r}", flush=True)
    finally:
        projects = sorted((out / "projects").glob("*/project.json"))
        if art.active:
            art.leave("probe cleanup")
    rows, tries = [], []
    for pj in projects:
        for node_json in sorted((pj.parent / "nodes").glob("*.json")):
            node = json.loads(node_json.read_text())
            node_id, image = node_json.stem, node_json.with_suffix(".png")
            meta = node.get("meta") or {}
            label = f"{pj.parent.name[9:15]} {node_id} {'X ' + ', '.join(meta['checked']) if meta.get('checked') else 'ok'}"
            tries.append({"project": pj.parent.name, "node": node_id, "state": node.get("state"),
                          "prompt": meta.get("prompt"), "checked": meta.get("checked"), "tries": meta.get("tries", 0),
                          "reworded": bool((node.get("spec") or {}).get("wording"))})
            if image.exists():
                rows.append((label, image))
    if rows:
        sheet(rows, out / "sheet.png", 4)
    shown = sorted(p.name for p in (out / "Pictures").rglob("*.png"))
    result = {"said": args.said, "spoken": spoken, "seconds": round(time.monotonic() - start, 1), "shown": shown,
              "tries": tries, "asked": asked, "rewrites": rewrites}
    (out / "result.json").write_text(json.dumps(result, indent=1))
    for t in tries:
        print(f"{t['node']} tries={t['tries']} reworded={t['reworded']} checked={t['checked']} {t['state']}")
    print(f"{len(shown)} shown, {len(rewrites)} rewrite calls, {len(asked)} questions, {result['seconds']} s")
    print(out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
