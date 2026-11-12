#!/usr/bin/env python3
"""Which image model draws scenes with several subjects right? (ART_FEATURES_PLAN F5)

Why: Adam's dragons came out with two heads and extra riders, and a person
larger than a battleship. The first bake-off (AVATAR_PLAN "Bake-off result")
had single-subject prompts only. This draws multi-subject scenes on each model,
8 seeds each, with the prompts art mode really sends (features -> ``features.render``),
then counts defects with the F4 checker (heads, riders) plus bench-only count
questions (people, fruit) that the F4 spike did not validate: spot-check those
by eye on the sheets before trusting them.

Two steps, so drawing (needs the GPU, stops the coder) and checking (needs
Ollama's vision model, evicts the voice model) never overlap:

    .venv/bin/python tools/art/bakeoff_multi.py draw [--model klein9 ...] [--seeds 8] [--no-voice] [--into DIR]
    .venv/bin/python tools/art/bakeoff_multi.py check DIR

``draw`` starts its own ComfyUI per model (like bakeoff.py): don't use art mode
meanwhile. ``--into`` adds models to an earlier run's folder (a second pass
with Jarvis off). ``check`` writes check.json, a contact sheet per scene
(rows models, columns seeds, red frame = checker found a problem) and prints a
markdown table. Output: ~/homeai-bench/multi-<time>/.
"""
from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from comfy_common import MODELS, Comfy, CoderStopped, workflow  # noqa: E402
from homeai.art import checker as CK  # noqa: E402
from homeai.art import features as F  # noqa: E402
from homeai.benchio import bench_path  # noqa: E402


def _feats(*pairs) -> list[F.Feature]:
    return [F.Feature(chr(65 + i), kind, text) for i, (kind, text) in enumerate(pairs)]


def _ask(kind: str, question: str, expect: int) -> CK.Check:
    return CK.Check(kind, question + " Answer with one number only.", expect, "")


# scene: (features, bench-only checks on top of CK.checklist)
SCENES = {
    "rider": (_feats(("subject", "a single red dragon with one head and four legs, flying"),
                     ("element", "one samurai in full armor sitting on the dragon's back"),
                     ("setting", "over the ocean")), []),
    "two_riders": (_feats(("subject", "a single elephant with one head"),
                          ("element", "two kids riding on the elephant's back"),
                          ("setting", "a jungle river")), []),
    "battleship": (_feats(("subject", "a single green dragon with one head"),
                          ("element", "a huge steam-powered battleship, many times bigger than the dragon"),
                          ("setting", "a stormy sea")), []),
    "two_people": (_feats(("subject", "a knight and a wizard shaking hands"),
                          ("setting", "in front of a castle")),
                   []),            # people: CK.checklist asks since 2026-11-11
    "fruit": (_feats(("subject", "three red apples and two green pears"), ("setting", "on a wooden table")),
              [_ask("apples", "How many apples are in this picture?", 3),
               _ask("pears", "How many pears are in this picture?", 2)]),
}


def prompt_of(scene: str):
    return F.render(SCENES[scene][0], "square")


def checks_of(scene: str) -> list[CK.Check]:
    features, extra = SCENES[scene]
    return CK.checklist(features) + extra


def draw(args) -> int:
    out = Path(args.into) if args.into else bench_path("multi")
    out.mkdir(parents=True, exist_ok=True)
    path = out / "result.json"
    data = json.loads(path.read_text()) if path.exists() else {"prompts": {}, "runs": []}
    data["prompts"] = {s: prompt_of(s).text for s in SCENES}
    with CoderStopped(voice_off=args.no_voice):
        for model in args.model:
            try:
                with Comfy(out / f"comfy-{model}.log") as comfy:
                    for scene in SCENES:
                        rendered = prompt_of(scene)
                        for seed in range(1, args.seeds + 1):
                            run = comfy.run(workflow(model, rendered.text, seed, rendered.width, rendered.height,
                                                     f"multi-{model}"))
                            row = {"model": model, "scene": scene, "seed": seed, "seconds": run["seconds"],
                                   "voice": not args.no_voice}
                            if "image" in run:
                                target = out / f"{scene}-{model}-{seed}.png"
                                shutil.copyfile(run["image"], target)
                                row["image"] = target.name
                            else:
                                row["error"] = run["error"]
                            data["runs"].append(row)
                            print(json.dumps(row), flush=True)
                            path.write_text(json.dumps(data, indent=1))
                            if "died" in run.get("error", ""):
                                raise RuntimeError(run["error"])
            except RuntimeError as exc:
                print(model, "FAILED:", exc, flush=True)
    print(out)
    return 0


def sheet(out: Path, scene: str, runs: list[dict]) -> None:
    from PIL import Image, ImageDraw
    cell, label, frame = 256, 110, 6
    models = list(dict.fromkeys(r["model"] for r in runs))
    seeds = sorted({r["seed"] for r in runs})
    canvas = Image.new("RGB", (label + cell * len(seeds), 24 + cell * len(models)), "white")
    draw_ = ImageDraw.Draw(canvas)
    for i, seed in enumerate(seeds):
        draw_.text((label + i * cell + 6, 6), f"seed {seed}", fill="black")
    for row, model in enumerate(models):
        y = 24 + row * cell
        draw_.text((6, y + 6), model, fill="black")
        for r in (r for r in runs if r["model"] == model and "image" in r):
            x = label + seeds.index(r["seed"]) * cell
            with Image.open(out / r["image"]) as picture:
                picture = picture.convert("RGB")
                picture.thumbnail((cell - 2 * frame, cell - 2 * frame))
                if r.get("problems"):
                    draw_.rectangle((x, y, x + cell - 1, y + cell - 1), fill="red")
                canvas.paste(picture, (x + frame, y + frame))
            if r.get("problems"):
                draw_.text((x + frame + 2, y + cell - frame - 14), ", ".join(r["problems"]), fill="red")
    canvas.save(out / f"sheet-{scene}.png")


def check(args) -> int:
    out = Path(args.folder)
    data = json.loads((out / "result.json").read_text())
    for r in data["runs"]:
        if "image" in r:
            r["problems"] = CK.problems(out / r["image"], checks_of(r["scene"]), CK.ollama_ask)
            print(r["scene"], r["model"], r["seed"], r["problems"], flush=True)
    (out / "check.json").write_text(json.dumps(data, indent=1))
    for scene in SCENES:
        sheet(out, scene, [r for r in data["runs"] if r["scene"] == scene])
    models = list(dict.fromkeys(r["model"] for r in data["runs"]))
    print("| model | " + " | ".join(SCENES) + " | clean | median s |")
    print("|---" * (len(SCENES) + 3) + "|")
    for model in models:
        mine = [r for r in data["runs"] if r["model"] == model]
        cells = []
        for scene in SCENES:
            rs = [r for r in mine if r["scene"] == scene and "image" in r]
            cells.append(f"{sum(1 for r in rs if not r['problems'])}/{len(rs)}")
        drawn = [r for r in mine if "image" in r]
        warm = sorted(r["seconds"] for r in drawn[1:]) or [0]
        print(f"| {model} | " + " | ".join(cells) + f" | {sum(1 for r in drawn if not r['problems'])}/{len(drawn)}"
              f" | {warm[len(warm) // 2]} |")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = ap.add_subparsers(dest="step", required=True)
    d = sub.add_parser("draw")
    d.add_argument("--model", action="append", choices=MODELS, required=True)
    d.add_argument("--seeds", type=int, default=8)
    d.add_argument("--no-voice", action="store_true", help="stop Jarvis during the run (models that need the card)")
    d.add_argument("--into")
    c = sub.add_parser("check")
    c.add_argument("folder")
    args = ap.parse_args()
    return draw(args) if args.step == "draw" else check(args)


if __name__ == "__main__":
    raise SystemExit(main())
