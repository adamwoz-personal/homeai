#!/usr/bin/env python3
"""Run the art job queue against the real ComfyUI in art mode (AVATAR_PLAN 6a step 5).

Why: the unit tests use a fake client. This proves on the box that queued jobs
draw one at a time in order (previews first), that cancelling a waiting job
removes it, that cancelling a running job interrupts ComfyUI, that results land
in the project store, and that restart_comfy brings back a working ComfyUI. It
switches to art mode and back to voice, and never talks to the voice agent.

Projects go to a scratch store under ~/homeai-bench/queue-live-<time>/, not
the real gallery.

Usage:
    .venv/bin/python tools/art/queue_live.py
"""
from __future__ import annotations

import json
import sys
import time
from datetime import datetime
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "tools" / "art"))

from comfy_common import workflow  # noqa: E402

from homeai import gpu_mode as G  # noqa: E402
from homeai.art import comfy as C  # noqa: E402
from homeai.art import jobs as J  # noqa: E402
from homeai.art import store as S  # noqa: E402

PROMPTS = ("a red fox in snow", "a lighthouse at dusk", "a teapot shaped like a frog", "a castle on a hill")


def main() -> int:
    switcher = G.ModeSwitcher(G.System(), log=lambda line: print("  ", line, flush=True))
    if switcher.mode() != "voice":
        print(f"start in voice mode (now {switcher.mode()})", file=sys.stderr)
        return 1
    out = Path.home() / "homeai-bench" / f"queue-live-{datetime.now():%Y%m%d-%H%M%S}"
    store = S.Store(out / "projects")
    project = store.create("general", "probe", "queue live")
    events: list[dict] = []
    checks: dict[str, bool] = {}
    try:
        switcher.art()
        queue = J.JobQueue(C.ComfyClient(), store, path=out / "queue.json",
                           notify=lambda e: events.append({**e, "t": round(time.monotonic(), 2)}),
                           restart=switcher.restart_comfy)
        nodes = [project.add(text, {"subject": text}, parent=None).id for text in PROMPTS]
        jobs = [queue.submit(project.id, node, workflow("klein", text, 7, 512, 512, prefix="queue-live"))
                for node, text in zip(nodes[:3], PROMPTS)]
        preview = queue.submit(project.id, nodes[3], workflow("klein", PROMPTS[3], 7, 512, 512, prefix="queue-live"),
                               priority=J.PREVIEW)
        checks["preview first in line"] = queue.waiting()[0] == preview.id
        queue.cancel(jobs[2].id)
        checks["waiting job cancelled"] = queue.get(jobs[2].id).state == "cancelled"
        queue.start()
        deadline = time.monotonic() + 240
        while time.monotonic() < deadline and not all(
                queue.get(j.id).state in J.FINAL for j in (preview, jobs[0], jobs[1])):
            time.sleep(0.5)
        order = [e["job"] for e in events if e["state"] == "done"]
        checks["done in order"] = order == [preview.id, jobs[0].id, jobs[1].id]
        reopened = store.open(project.id)
        checks["pictures in the store"] = all(reopened.image(n).exists() for n in (nodes[0], nodes[1], nodes[3]))
        checks["cancelled node marked failed"] = reopened.nodes[nodes[2]].state == "failed"
        long = workflow("klein", "a detailed city", 8, 1024, 1024, prefix="queue-live")
        node = project.add("long one", {}, parent=None).id
        running = queue.submit(project.id, node, long)
        while queue.get(running.id).state != "running" or not queue.get(running.id).prompt_id:
            time.sleep(0.05)
        queue.cancel(running.id)
        while queue.get(running.id).state not in J.FINAL:
            time.sleep(0.2)
        checks["running job interrupted"] = queue.get(running.id).state == "cancelled"
        queue.stop()
        switcher.restart_comfy()
        after = C.ComfyClient().run(workflow("klein", "a blue bird", 9, 512, 512, prefix="queue-live"), timeout=120)
        checks["works after restart"] = after.image.exists()
    finally:
        switcher.voice()
    out.mkdir(parents=True, exist_ok=True)
    (out / "result.json").write_text(json.dumps({"checks": checks, "events": events}, indent=1))
    print(json.dumps(checks, indent=1))
    print(out)
    return 0 if checks and all(checks.values()) else 1


if __name__ == "__main__":
    raise SystemExit(main())
