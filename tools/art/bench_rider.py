#!/usr/bin/env python3
"""Do our prompt changes cut two-headed dragons and extra riders? (ART_FEATURES_PLAN F0 live check)

Why: Adam's dragon session (2026-11-07) produced two heads, up to three riders
and a samurai leg used as a dragon leg. Some of that was in our prompts
("with two riders and two heads", "Avoid cyberpunk samurai's right hand").
This draws the same scene from the old prompts and the new ones, several seeds
each, with klein 9B (the default), and writes a contact sheet to count defects
by eye (and later by the F4 checker).

Needs ComfyUI up (art mode). Starts art mode if it's off and goes back to voice
after; leaves it on if it was already on. Nothing is spoken.

Writes ~/homeai-bench/rider-<time>/{result.json,sheet.png,<case>-<seed>.png}.

Usage:
    .venv/bin/python tools/art/bench_rider.py [--seeds 4]
"""
from __future__ import annotations

import argparse
import json
import shutil
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

from homeai import gpu_mode as G  # noqa: E402
from homeai.art import comfy as C  # noqa: E402
from homeai.art import prompt_spec as PS  # noqa: E402
from homeai.art import workflows  # noqa: E402
from homeai.benchio import bench_path  # noqa: E402

NODE_006 = {"subject": "oriental undead samurai", "details": "riding an oriental undead dragon, in full samurai armor",
            "style": "cyberpunk style", "palette": "neon colours", "lighting": "high contrast lighting",
            "camera": "low angle", "composition": "dynamic pose", "negative": "cyberpunk samurai's right hand"}
CASES = {
    # Prompts exactly as sent in Adam's session (stored nodes 005 and 006).
    "old-005": "Undead dragon. Being ridden by a cyberpunk samurai, with two riders and two heads, with a more skeletal "
               "appearance. Cyberpunk style, neon colours colours, high contrast lighting. Low angle. Dynamic pose. "
               "Avoid cyberpunk samurai's right hand.",
    "old-006": "Oriental undead samurai. Riding an oriental undead dragon, in full samurai armor. Cyberpunk style, "
               "neon colours colours, high contrast lighting. Low angle. Dynamic pose. Avoid cyberpunk samurai's right hand.",
    # The same spec through today's renderer (no "Avoid", no doubled words).
    "new-006": PS.render(NODE_006).text,
    # What F1's features should produce: the mount first, counts said plainly, the rider placed.
    "counts": "An oriental undead dragon with a single head and four legs. One undead samurai in full samurai armor "
              "sits on its back, holding the reins with both hands. Cyberpunk style, neon colours, high contrast "
              "lighting. Low angle.",
}


def sheet(rows: list[tuple[str, Path]], path: Path, per_row: int) -> None:
    from PIL import Image, ImageDraw
    cell, label = 300, 22
    rows_n = (len(rows) + per_row - 1) // per_row
    canvas = Image.new("RGB", (cell * per_row, (cell + label) * rows_n), "white")
    draw = ImageDraw.Draw(canvas)
    for i, (name, image) in enumerate(rows):
        x, y = (i % per_row) * cell, (i // per_row) * (cell + label)
        thumb = Image.open(image).convert("RGB")
        thumb.thumbnail((cell, cell))
        canvas.paste(thumb, (x, y + label))
        draw.text((x + 4, y + 4), name, fill="black")
    canvas.save(path)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--seeds", type=int, default=4)
    args = ap.parse_args()
    switcher = G.ModeSwitcher(G.System(), log=lambda line: print("  ", line, flush=True))
    started = switcher.mode() != "art"
    if switcher.mode() == "coding":
        print("not from coding mode", file=sys.stderr)
        return 1
    out = bench_path("rider")
    out.mkdir(parents=True)
    rows, results = [], []
    try:
        if started:
            switcher.art()
        client = C.ComfyClient()
        for name, prompt in CASES.items():
            for seed in range(1, args.seeds + 1):
                start = time.monotonic()
                result = client.run(workflows.klein9(prompt, seed, 1216, 832, prefix=f"bench/rider-{name}"))
                target = out / f"{name}-{seed}.png"
                shutil.copyfile(result.image, target)
                rows.append((f"{name} seed {seed}", target))
                results.append({"case": name, "seed": seed, "seconds": round(time.monotonic() - start, 1)})
                print(f"{name} seed {seed}: {results[-1]['seconds']}s", flush=True)
    finally:
        if started:
            switcher.voice()
    sheet(rows, out / "sheet.png", args.seeds)
    (out / "result.json").write_text(json.dumps({"prompts": CASES, "runs": results,
                                                 "defects": "counted by eye on sheet.png; see ROUTING_GUIDE"}, indent=1))
    print(out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
