#!/usr/bin/env python3
"""Does a scene of mannequins as klein's reference fix multi-figure pictures? (ART_FEATURES_PLAN F10 step 3)

Why: Adam's paladin-and-troll pictures go wrong with two figures (extra blades, a shield for the troll, the two
standing side by side instead of facing off). The pose library (homeai/poses) can draw a scene: forms from motion
capture, turned to face each other, placed left and right. klein 9B takes a picture as a ReferenceLatent (pose
probe 2026-11-06). This draws the same prompt with and without the scene reference, same seeds, and checks both
with the F4 checks (blades, shields: tools/art/bakeoff_multi.py "paladin") plus a sheet for the eye.

Two steps like the bake-off (drawing needs ComfyUI, checking needs the voice model):
    .venv/bin/python tools/poses/probe_scene_reference.py draw [--seeds 8] [--scene facing_off]
    .venv/bin/python tools/poses/probe_scene_reference.py check DIR
Output: ~/homeai-bench/scene-ref-<time>/ (reference.png, pictures, result.json, check.json, sheet.png)
"""
from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "tools" / "art"))

from homeai.benchio import bench_path  # noqa: E402
from homeai.poses import forms as FM  # noqa: E402
from homeai.poses import render as R  # noqa: E402

LIBRARY = REPO / "homeai/poses/library"
SIZE = 1024
# A scene: which forms, turned how, where across the picture (0..1), how big. The troll is a size up.
SCENES = {
    "facing_off": {"scene": "paladin", "figures": [("sword-guard", 90, 0.32, 1.0), ("boxing-guard", 270, 0.7, 1.2)],
                   "who": ["the paladin", "the troll"]},
    # 2026-11-20 first run: overlapping mannequins gave a third person in 2 of 8 and the plain reference
    # background gave studio-grey photos. Wider apart, and the words ask for the scene's own background.
    "facing_off_wide": {"scene": "paladin", "figures": [("sword-guard", 90, 0.24, 0.9), ("boxing-guard", 270, 0.77, 1.1)],
                        "who": ["the paladin", "the troll"]},
}
REFERENCE_WORDS = ("Use the reference picture only for the poses and places of the figures: {placed}. Turn the "
                   "wooden mannequins into the real characters described, with real faces, skin and clothes; no wood, "
                   "no ball joints, no mannequin parts. Exactly two figures. Paint a full, detailed background that suits the scene, in a painted fantasy illustration style, not the reference's plain backdrop.")


def scene_picture(name: str, path: Path) -> Path:
    figures = []
    for form, turn, x, scale in SCENES[name]["figures"]:
        spec = json.loads((LIBRARY / f"{form}.json").read_text())
        figures.append(R.Placed(FM.turned(spec["joints"], turn), x=x, scale=scale))
    R.draw(figures, SIZE, SIZE).save(path)
    return path


def prompt_with_reference(base: str, name: str) -> str:
    who = SCENES[name]["who"]
    placed = f"{who[0]} on the left facing right, {who[1]} on the right facing left"
    return base + " " + REFERENCE_WORDS.format(placed=placed)


def draw(args) -> int:
    import bakeoff_multi as B
    from comfy_common import ROOT, Comfy, CoderStopped, workflow
    from probe_pose_control import MODELS, klein_reference

    out = bench_path("scene-ref")
    out.mkdir(parents=True)
    spec = SCENES[args.scene]
    reference = scene_picture(args.scene, out / "reference.png")
    shutil.copy(reference, ROOT / "app/input" / f"scene-{args.scene}.png")
    base = B.prompt_of(spec["scene"]).text
    data = {"scene": args.scene, "prompt": base, "with_reference": prompt_with_reference(base, args.scene), "runs": []}
    with CoderStopped(), Comfy(out / "comfy.log") as comfy:
        for seed in range(1, args.seeds + 1):
            for mode in ("plain", "reference"):
                graph = (workflow("klein9", base, seed, SIZE, SIZE, f"scene-{mode}") if mode == "plain" else
                         klein_reference(*MODELS["klein9"], f"scene-{args.scene}.png", data["with_reference"], seed,
                                         SIZE, SIZE, f"scene-{mode}"))
                run = comfy.run(graph)
                row = {"mode": mode, "seed": seed, "seconds": run["seconds"]}
                if "image" in run:
                    row["image"] = f"{mode}-{seed}.png"
                    shutil.copyfile(run["image"], out / row["image"])
                else:
                    row["error"] = run["error"]
                data["runs"].append(row)
                print(json.dumps(row), flush=True)
                (out / "result.json").write_text(json.dumps(data, indent=1))
    print(out)
    return 0


def check(args) -> int:
    import bakeoff_multi as B
    from bench_rider import sheet
    from homeai.art import checker as CK

    out = Path(args.folder)
    data = json.loads((out / "result.json").read_text())
    checks = B.checks_of(SCENES[data["scene"]]["scene"])
    for row in data["runs"]:
        if "image" in row:
            row["problems"] = CK.problems(out / row["image"], checks, CK.ollama_ask)
            print(row["mode"], row["seed"], row["problems"], flush=True)
    (out / "check.json").write_text(json.dumps(data, indent=1))
    rows = [("reference", out / "reference.png")]
    for mode in ("plain", "reference"):
        for row in data["runs"]:
            if row["mode"] == mode and "image" in row:
                rows.append((f"{mode} {row['seed']} {'; '.join(row['problems']) or 'ok'}", out / row["image"]))
    sheet(rows, out / "sheet.png", 5)
    for mode in ("plain", "reference"):
        mine = [r for r in data["runs"] if r["mode"] == mode and "image" in r]
        print(f"{mode}: {sum(1 for r in mine if not r['problems'])}/{len(mine)} clean")
    return 0


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = ap.add_subparsers(dest="step", required=True)
    d = sub.add_parser("draw")
    d.add_argument("--seeds", type=int, default=8)
    d.add_argument("--scene", default="facing_off", choices=list(SCENES))
    c = sub.add_parser("check")
    c.add_argument("folder")
    args = ap.parse_args(argv)
    return draw(args) if args.step == "draw" else check(args)


if __name__ == "__main__":
    raise SystemExit(main())
