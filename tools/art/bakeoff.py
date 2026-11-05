#!/usr/bin/env python3
"""Run the art scenarios on every image model, for Adam to rate (AVATAR_PLAN §8, Phase 5).

Why: numbers plus Adam's ratings pick each preset's default model. Every prompt runs
on every model with the same seeds, and each scenario gets one contact sheet per
seed: a column per model. results.json has seconds and any errors; ratings merge
into it later. Anime models only run the scenarios they're meant for. A model that
fails is recorded and skipped. Output goes to ~/homeai-bench/bakeoff-<time>/. The
coder is stopped meanwhile. Timings here are not a benchmark: use comfy_smoke.py.

Usage:
    .venv/bin/python tools/art/bakeoff.py [--model klein] [--scenario landscape] [--seeds 3]
"""
from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from comfy_common import ANIME, MODELS, RETIRED, Comfy, CoderStopped, workflow  # noqa: E402

MIRA = ("Mira, a young elven ranger with short copper hair, a scar through her left eyebrow, "
        "a green hooded cloak and a silver leaf brooch")
# scenario: (width, height, prompts)
SCENARIOS = {
    "landscape": (1344, 768, ["a vast mountain valley at dawn, mist in the pines, a river catching the first light, "
                              "painterly fantasy landscape"]),
    "battle_map": (1024, 1024, ["top-down tabletop battle map of a ruined stone keep, cracked flagstones, a collapsed "
                                "wall, a well in the courtyard, even lighting, no perspective, no grid, no text"]),
    "battle_map_grid": (1024, 1024, ["top-down tabletop D&D battle map of a dungeon with three rooms joined by corridors, "
                                     "stone floors, a square one-inch grid drawn over the whole map, even lighting, "
                                     "no perspective, no text"]),
    "floor_plan": (1344, 768, ["architectural floor plan of a two-bedroom cottage, black ink on white paper, top-down, "
                               "walls, doors and windows drawn to scale, furniture outlines, no text"]),
    "world_map": (1344, 768, ["hand-drawn fantasy world map on aged parchment, coastlines, mountains, forests and "
                              "rivers in ink, compass rose, no lettering"]),
    "character": (1024, 1024, [f"portrait of {MIRA}, fantasy illustration",
                               f"{MIRA}, sitting in a crowded tavern, fantasy illustration",
                               f"{MIRA}, climbing a snowy mountain pass, fantasy illustration"]),
    "anime": (1024, 1024, ["anime key visual of a girl with a giant paper crane on a rooftop at sunset, clean line art"]),
    "kid": (768, 768, ["a friendly green dragon having a picnic with a snake, bright picture book style",
                       "a purple snake wearing a tiny crown, bright picture book style",
                       "a baby dragon learning to fly over a castle, bright picture book style",
                       "a dragon and a snake building a sandcastle, bright picture book style",
                       "a sleepy dragon curled around a pile of books, bright picture book style"]),
    "storyboard": (1344, 768, ["a small robot finds a seed in a ruined city, wide shot, soft film still",
                               "the small robot waters a sprout in a broken flowerpot, close shot, soft film still",
                               "the small robot under a huge tree growing through the city, wide shot, soft film still"]),
    "architecture": (1344, 768, ["concept art of a cosy timber-frame cottage interior, reading nook, stone hearth, "
                                 "warm afternoon light"]),
    "labels": (1024, 1024, ["a shop sign that reads \"THE GILDED FOX\" in gold letters on dark wood"]),
    "avatar": (1024, 1024, ["front portrait of a serene android woman, green stippled pointillist style, "
                            "circuit seams across the face, black background"]),
}


ANIME_SCENARIOS = ("anime", "character", "kid")


def sheet(rows: dict[str, list[str]], path: Path, cell: int = 480) -> None:
    """One column per model, one row per image, labelled."""
    from PIL import Image, ImageDraw

    height = max(len(v) for v in rows.values())
    canvas = Image.new("RGB", (cell * len(rows), 24 + cell * height), "black")
    draw = ImageDraw.Draw(canvas)
    for col, (model, images) in enumerate(rows.items()):
        draw.text((col * cell + 6, 6), model, fill="white")
        for row, image in enumerate(images):
            with Image.open(image) as picture:
                picture.thumbnail((cell, cell))
                canvas.paste(picture.convert("RGB"), (col * cell, 24 + row * cell))
    canvas.save(path)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--model", action="append", choices=MODELS)
    ap.add_argument("--scenario", action="append", choices=tuple(SCENARIOS))
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--no-voice", action="store_true", help="stop Jarvis and unload the voice model during the run")
    args = ap.parse_args()
    models = args.model or tuple(m for m in MODELS if m not in RETIRED)
    chosen = args.scenario or tuple(SCENARIOS)
    out = Path.home() / "homeai-bench" / f"bakeoff-{datetime.now():%Y%m%d-%H%M%S}"
    out.mkdir(parents=True)
    results = {}
    images: dict[tuple[str, int], dict[str, list[str]]] = {}
    with CoderStopped(voice_off=args.no_voice):
        for model in models:                            # one server per model: load it once, RAM stays bounded
            try:
                with Comfy(out / f"comfy-{model}.log") as comfy:
                    for scenario in chosen:
                        if model in ANIME and scenario not in ANIME_SCENARIOS:
                            continue
                        width, height, prompts = SCENARIOS[scenario]
                        for p, prompt in enumerate(prompts):
                            for seed in range(args.seeds):
                                run = comfy.run(workflow(model, prompt, 1000 + seed, width, height, f"{scenario}-{model}"))
                                results.setdefault(scenario, {}).setdefault(model, []).append(run)
                                print(scenario, model, p, seed, run.get("seconds"), run.get("error", "")[:120], flush=True)
                                if "image" in run:
                                    images.setdefault((scenario, seed), {}).setdefault(model, []).append(run["image"])
                                elif "died" in run.get("error", ""):
                                    raise RuntimeError(run["error"])
                        (out / "results.json").write_text(json.dumps(results, indent=1))
            except RuntimeError as exc:
                print(model, "FAILED:", exc, flush=True)
    for (scenario, seed), rows in images.items():
        sheet(rows, out / f"{scenario}-seed{seed}.png", cell=384)
    print(out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
