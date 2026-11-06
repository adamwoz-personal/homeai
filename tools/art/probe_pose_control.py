#!/usr/bin/env python3
"""Can an art model draw a character in Ani's mannequin pose? (REMOTE_APP_PLAN §9, contract R9)

Why: the pose feature needs a model that follows a pose picture, and that was never
tested. This gives each of her exports to the model as a reference image and asks
for a character in the same pose, with 3 seeds. Writes sheet.png (mannequin, then
each seed) and overlay.png (the spec's true joints drawn on each result, to judge
limb by limb) to ~/homeai-bench/pose-probe-<time>/. The coder is stopped meanwhile.

Usage:
    .venv/bin/python tools/art/probe_pose_control.py [--exports DIR] [--model klein9] [--seeds 3] [--prompt TEXT]
"""
from __future__ import annotations

import argparse
import json
import shutil
import sys
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from comfy_common import ROOT, Comfy, CoderStopped  # noqa: E402

EXPORTS = Path.home() / "ani/pose-exports/2026-11-06"
PROMPT = ("Turn the wooden mannequin into {who}, full body, keeping exactly the same pose: the same head tilt, "
          "arm positions, elbow bends and leg stance. Fantasy storybook illustration, plain soft background.")
BONES = (("head", "neckTop"), ("neckTop", "chestTop"), ("shoulderL", "shoulderR"), ("shoulderL", "elbowL"),
         ("elbowL", "handL"), ("shoulderR", "elbowR"), ("elbowR", "handR"), ("chestTop", "hips"), ("hipL", "hipR"),
         ("hipL", "kneeL"), ("kneeL", "ankleL"), ("hipR", "kneeR"), ("kneeR", "ankleR"))


def klein_reference(unet: str, clip: str, image: str, prompt: str, seed: int, width: int, height: int, prefix: str) -> dict:
    """ComfyUI's klein image edit template: the picture joins both conditionings as a reference latent."""
    return {
        "unet": {"class_type": "UNETLoader", "inputs": {"unet_name": unet, "weight_dtype": "default"}},
        "clip": {"class_type": "CLIPLoader", "inputs": {"clip_name": clip, "type": "flux2"}},
        "vae": {"class_type": "VAELoader", "inputs": {"vae_name": "flux2-vae.safetensors"}},
        "load": {"class_type": "LoadImage", "inputs": {"image": image}},
        "encode": {"class_type": "VAEEncode", "inputs": {"pixels": ["load", 0], "vae": ["vae", 0]}},
        "pos": {"class_type": "CLIPTextEncode", "inputs": {"text": prompt, "clip": ["clip", 0]}},
        "zero": {"class_type": "ConditioningZeroOut", "inputs": {"conditioning": ["pos", 0]}},
        "ref_pos": {"class_type": "ReferenceLatent", "inputs": {"conditioning": ["pos", 0], "latent": ["encode", 0]}},
        "ref_neg": {"class_type": "ReferenceLatent", "inputs": {"conditioning": ["zero", 0], "latent": ["encode", 0]}},
        "latent": {"class_type": "EmptyFlux2LatentImage", "inputs": {"width": width, "height": height, "batch_size": 1}},
        "sigmas": {"class_type": "Flux2Scheduler", "inputs": {"steps": 4, "width": width, "height": height}},
        "noise": {"class_type": "RandomNoise", "inputs": {"noise_seed": seed}},
        "pick": {"class_type": "KSamplerSelect", "inputs": {"sampler_name": "euler"}},
        "guider": {"class_type": "CFGGuider", "inputs": {"model": ["unet", 0], "positive": ["ref_pos", 0],
                                                         "negative": ["ref_neg", 0], "cfg": 1.0}},
        "sample": {"class_type": "SamplerCustomAdvanced", "inputs": {
            "noise": ["noise", 0], "guider": ["guider", 0], "sampler": ["pick", 0], "sigmas": ["sigmas", 0],
            "latent_image": ["latent", 0]}},
        "decode": {"class_type": "VAEDecode", "inputs": {"samples": ["sample", 0], "vae": ["vae", 0]}},
        "save": {"class_type": "SaveImage", "inputs": {"images": ["decode", 0], "filename_prefix": prefix}},
    }


MODELS = {"klein9": ("flux-2-klein-9b-fp8.safetensors", "qwen_3_8b_fp8mixed.safetensors"),
          "klein": ("flux-2-klein-4b.safetensors", "qwen_3_4b.safetensors")}


def joints_in_pixels(spec: dict, fit: dict, width: int) -> dict[str, tuple[float, float]]:
    """The spec's joints on a picture ``width`` wide, using the export's fit (pixel = canvas x scale + offset)."""
    shrink = width / fit["imageWidth"]
    return {name: ((p["x"] * fit["scale"] + fit["offsetX"]) * shrink, (p["y"] * fit["scale"] + fit["offsetY"]) * shrink)
            for name, p in spec["joints"].items()}


def overlay(image, spec: dict, fit: dict):
    """``image`` with the true skeleton drawn over it in red."""
    from PIL import ImageDraw

    picture = image.convert("RGB")
    draw = ImageDraw.Draw(picture)
    points = joints_in_pixels(spec, fit, picture.width)
    for a, b in BONES:
        draw.line([points[a], points[b]], fill=(255, 0, 0), width=3)
    return picture


def grid(rows: list[tuple[str, list]], path: Path, cell: int = 300) -> None:
    """One row per pose: its label, then each picture."""
    from PIL import Image, ImageDraw

    high = cell * 1040 // 800
    wide = max(len(pictures) for _, pictures in rows)
    page = Image.new("RGB", (cell * wide, (high + 22) * len(rows)), "black")
    draw = ImageDraw.Draw(page)
    for r, (label, pictures) in enumerate(rows):
        y = r * (high + 22)
        draw.text((6, y + 5), label, fill="white")
        for c, picture in enumerate(pictures):
            page.paste(picture.convert("RGB").resize((cell, high)), (c * cell, y + 22))
    page.save(path)


def main() -> int:
    from PIL import Image

    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--exports", type=Path, default=EXPORTS)
    ap.add_argument("--model", choices=MODELS, default="klein9")
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--prompt", default=PROMPT)
    args = ap.parse_args()
    out = Path.home() / "homeai-bench" / f"pose-probe-{datetime.now():%Y%m%d-%H%M%S}"
    out.mkdir(parents=True)
    poses = sorted(p.stem for p in args.exports.glob("*.png"))
    results, clean, marked = {}, [], []
    with CoderStopped(), Comfy(out / "comfy.log") as comfy:
        for name in poses:
            shutil.copy(args.exports / f"{name}.png", ROOT / "app/input" / f"pose-{name}.png")
            spec = json.loads((args.exports / f"{name}.json").read_text())
            fit = json.loads((args.exports / f"{name}.fit.json").read_text())
            prompt = args.prompt.format(who=spec["description"])
            mannequin = Image.open(args.exports / f"{name}.png")
            row, row_marked = [mannequin], [overlay(mannequin, spec, fit)]
            for seed in range(args.seeds):
                run = comfy.run(klein_reference(*MODELS[args.model], f"pose-{name}.png", prompt, 2000 + seed,
                                                fit["imageWidth"], fit["imageHeight"], f"pose-{args.model}-{name}"))
                results.setdefault(name, []).append(run)
                print(name, seed, run.get("seconds"), run.get("error", "")[:200], flush=True)
                if "image" in run:
                    picture = Image.open(run["image"])
                    row.append(picture)
                    row_marked.append(overlay(picture, spec, fit))
            clean.append((f"{name} ({args.model})", row))
            marked.append((name, row_marked))
    (out / "results.json").write_text(json.dumps(results, indent=1))
    grid(clean, out / "sheet.png")
    grid(marked, out / "overlay.png")
    print(out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
