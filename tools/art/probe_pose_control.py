#!/usr/bin/env python3
"""Can an art model draw a character in Ani's mannequin pose? (REMOTE_APP_PLAN §9, contract R9)

Why: the pose feature needs a model that follows a pose picture, and that was never
tested. This gives each of her exports to the model as a reference image and asks
for a character in the same pose, with 3 seeds. Writes sheet.png (mannequin, then
each seed) and overlay.png (the spec's true joints drawn on each result, to judge
limb by limb) to ~/homeai-bench/pose-probe-<time>/. The coder is stopped meanwhile.

Usage:
    .venv/bin/python tools/art/probe_pose_control.py [--exports DIR] [--model klein9] [--seeds 3] [--prompt TEXT] [--who TEXT]
"""
from __future__ import annotations

import argparse
import json
import math
import shutil
import sys
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from comfy_common import ROOT, Comfy, CoderStopped  # noqa: E402

EXPORTS = Path.home() / "ani/pose-exports/2026-11-06"
# What home assumes when the spec leaves a field out (contract: "Fields at a glance").
DEFAULTS = {"gender": "female", "handedness": "right", "style": "anime", "background": "plain soft background", "framing": "full-body"}
FRAMING = {"full-body": "The whole figure is in frame from the top of the head to the soles of the feet, with space around it.",
           "waist-up": "Show the figure from the waist up.", "portrait": "Show the head and shoulders."}
PROMPT = ("Redraw this figure as {who}, a {gender} character, in clean {style} style: a real person with natural skin, "
          "a real face and modest, age-appropriate clothes covering the body, with no wood, no ball joints and no mannequin "
          "parts anywhere. Natural {style} body proportions, about seven heads tall, with long legs. Match the figure's pose "
          "exactly. {pose} {feeling} {face} Add nothing to the hands and carry nothing that the description doesn't name. "
          "{held} Shoes that suit the outfit, unless the description says barefoot. "
          "{framing} {style} illustration, clean line art, {background}.")
BONES = (("head", "neckTop"), ("neckTop", "chestTop"), ("shoulderL", "shoulderR"), ("shoulderL", "elbowL"),
         ("elbowL", "handL"), ("shoulderR", "elbowR"), ("elbowR", "handR"), ("chestTop", "hips"), ("hipL", "hipR"),
         ("hipL", "kneeL"), ("kneeL", "ankleL"), ("hipR", "kneeR"), ("kneeR", "ankleR"))


def near(a: dict, b: dict, reach: float) -> bool:
    return (a["x"] - b["x"]) ** 2 + (a["y"] - b["y"]) ** 2 < reach ** 2


def elbow_angle(shoulder: dict, elbow: dict, hand: dict) -> float:
    """Degrees at the elbow: 180 is a straight arm."""
    ux, uy = shoulder["x"] - elbow["x"], shoulder["y"] - elbow["y"]
    vx, vy = hand["x"] - elbow["x"], hand["y"] - elbow["y"]
    return abs(math.degrees(math.atan2(ux * vy - uy * vx, ux * vx + uy * vy)))


def arm_words(side: str, hand: dict, elbow: dict, shoulder: dict, head: dict, hips: dict, chest: dict) -> str:
    """What one arm is doing, or "" when it's nothing a model tends to lose."""
    if hand["y"] < head["y"]:
        return f"The arm on the {side} of the picture is raised straight up, the hand above the head."
    if abs(hand["y"] - shoulder["y"]) < 30 and abs(hand["x"] - shoulder["x"]) > 80:
        return f"The arm on the {side} of the picture is stretched straight out to the side."
    if elbow_angle(shoulder, elbow, hand) > 130:          # a hanging arm also ends near the hips
        return ""
    if near(hand, hips, 75):
        return f"The arm on the {side} of the picture is bent sharply, the elbow out and the hand at the waist."
    if near(hand, chest, 90):
        return f"The arm on the {side} of the picture is bent sharply, the hand up in front of the chest."
    return ""


def pose_words(spec: dict) -> str:
    """The parts of a pose that a picture alone doesn't hold, in words, from the spec's joints.

    Her rig 1 is ``mirror``, so ``L`` is the left of the picture in the front view.
    """
    j = spec["joints"]
    words = [arm_words("left", j["handL"], j["elbowL"], j["shoulderL"], j["head"], j["hips"], j["chestTop"]),
             arm_words("right", j["handR"], j["elbowR"], j["shoulderR"], j["head"], j["hips"], j["chestTop"])]
    lean = j["chestTop"]["x"] - j["hips"]["x"]
    if abs(lean) > 15:
        words.append(f"The upper body leans toward the {'right' if lean > 0 else 'left'} of the picture from the waist, "
                     "while the hips stay level and centred over both feet.")
    tilt = j["head"]["x"] - j["neckTop"]["x"]
    if abs(tilt) > 6:
        words.append(f"The head tilts toward the {'right' if tilt > 0 else 'left'} of the picture.")
    return " ".join(w for w in words if w)


def feeling_words(spec: dict) -> str:
    """The spec's feeling as an instruction, so the face shows it and not only the limbs."""
    feeling = spec.get("feeling")
    if not feeling:
        return ""
    strength = spec.get("strength", 100)
    how = "a little" if strength < 75 else "very" if strength <= 110 else "overwhelmingly"
    return (f"The character is {how} {feeling}: her face, eyes, eyebrows and mouth must clearly show that she is {feeling}, "
            f"as an anime character would show it, and so must her whole posture.")


def face_words(spec: dict) -> str:
    """Ani's optional hint about the face, as guidance. Left out, the feeling alone sets the face."""
    face = spec.get("face")
    return f"Her face: {face}." if face else ""


def held_words(handedness: str, turn: int = 0) -> str:
    """Which hand holds things. Facing the viewer, the figure's own right hand is on the left of the picture."""
    hand = "left" if handedness == "left" else "right"
    if turn:
        return f"Anything held is in the character's own {hand} hand."
    other = "right" if hand == "left" else "left"
    return f"Anything held is in the character's own {hand} hand, the hand on the {other} of the picture, gripped firmly."


def prompt_for(spec: dict, template: str = PROMPT, who: str | None = None) -> str:
    """The prompt for ``spec``, with each left-out field taking its default."""
    fields = {**DEFAULTS, **{k: spec[k] for k in DEFAULTS if spec.get(k)}}
    framing = FRAMING.get(fields.pop("framing"), FRAMING[DEFAULTS["framing"]])
    held = held_words(fields.pop("handedness"), spec.get("turn", 0))
    return template.format(who=who or spec["description"], framing=framing, held=held, pose=pose_words(spec),
                           feeling=feeling_words(spec), face=face_words(spec), **fields)


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
    ap.add_argument("--who", help="use this description instead of the spec's")
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
            prompt = prompt_for(spec, args.prompt, args.who)
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
