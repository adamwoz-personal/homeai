#!/usr/bin/env python3
"""Does a chimera LoRA beat the glossary? Plain "chimera" on klein 4B with and without the LoRA, vs klein9 with the
glossary's description (ART_FEATURES_PLAN F9 spike S4).

Why: the glossary (homeai/art/wording.py) gets a right chimera about 5/8 by describing it at length. If a LoRA
trained on right pictures (tools/art/collect_lora_set.py + tools/art/spike_lora.py) gets the plain word right as
often or more, practice nights can teach klein creatures instead of hand-writing descriptions. One scene was in
the training captions, one wasn't (does it learn the creature or the picture?).

Every picture is asked the glossary's chimera questions by the voice model (as the live checker does) and goes
on a contact sheet for Claude's eye. Needs art mode (starts it if off, voice after; leaves it on if it was on).
The LoRA is copied into ComfyUI's models/loras. Writes ~/homeai-bench/lora-bench-<time>/.

Usage:
    .venv/bin/python tools/art/bench_lora.py LORA.safetensors [--seeds 8] [--strength 1.0]
"""
from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

from homeai import gpu_mode as G  # noqa: E402
from homeai.art import checker as CK  # noqa: E402
from homeai.art import comfy as C  # noqa: E402
from homeai.art import wording as W  # noqa: E402
from homeai.art import workflows  # noqa: E402
from homeai.benchio import bench_path  # noqa: E402
from tools.art.bench_rider import sheet  # noqa: E402

LORAS = Path.home() / ".local/share/homeai/comfyui/app/models/loras"
KLEIN4_UNET = "flux-2-klein-4b.safetensors"        # distilled, 4 steps, like klein9 in art mode
KLEIN4_CLIP = "qwen_3_4b.safetensors"
SCENES = {
    "seen": "An anime girl wielding a katana fighting a chimera.",        # in the training captions
    "unseen": "A wizard casting a spell at a chimera in a cave.",
}
CASES = ("4b", "4b+lora", "9b+glossary")


def graph(case: str, plain: str, seed: int, lora: str | None = None, strength: float = 1.0) -> dict:
    """The ComfyUI graph for one case: klein9 with the glossary's words, or klein 4B with the plain words, its
    model passed through the LoRA for "4b+lora"."""
    if case == "9b+glossary":
        return workflows.klein9(W.expand(plain)[0], seed, prefix="bench/lora-9b")
    g = workflows.klein9(plain, seed, prefix=f"bench/lora-{case.replace('+', '-')}")      # a fresh dict each call
    g["unet"]["inputs"]["unet_name"] = KLEIN4_UNET
    g["clip"]["inputs"]["clip_name"] = KLEIN4_CLIP
    if case == "4b+lora":
        if not lora:
            raise ValueError("4b+lora needs a LoRA")
        g["lora"] = {"class_type": "LoraLoaderModelOnly",
                     "inputs": {"model": ["unet", 0], "lora_name": lora, "strength_model": strength}}
        g["guider"]["inputs"]["model"] = ["lora", 0]
    return g


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("lora", type=Path)
    ap.add_argument("--seeds", type=int, default=8)
    ap.add_argument("--strength", type=float, default=1.0)
    args = ap.parse_args(argv)
    LORAS.mkdir(parents=True, exist_ok=True)
    name = f"homeai-{args.lora.name}"
    shutil.copyfile(args.lora, LORAS / name)
    asks = W.expand("a chimera")[1]
    switcher = G.ModeSwitcher(G.System(), log=lambda line: print("  ", line, flush=True))
    if switcher.mode() == "coding":
        print("not from coding mode", file=sys.stderr)
        return 1
    started = switcher.mode() != "art"
    out = bench_path("lora-bench")
    out.mkdir(parents=True)
    rows, results = [], []
    try:
        if started:
            switcher.art()
        client = C.ComfyClient()
        for case in CASES:
            for scene, plain in SCENES.items():
                for seed in range(1, args.seeds + 1):
                    result = client.run(graph(case, plain, seed, name, args.strength))
                    tag = f"{case.replace('+', '-')}-{scene}-{seed}"
                    target = out / f"{tag}.png"
                    shutil.copyfile(result.image, target)
                    answers = {a.kind: CK.ollama_ask(target, a.question)[:30] for a in asks}
                    ok = all((CK.parse_yes(answers[a.kind]) == a.expect) for a in asks)
                    results.append({"case": case, "scene": scene, "seed": seed, "ok": ok, "answers": answers})
                    rows.append((f"{tag} {'ok' if ok else 'x'}", target))
                    print(f"{tag}: {'ok' if ok else 'x'} {answers}", flush=True)
    finally:
        if started:
            switcher.voice()
    sheet(rows, out / "sheet.png", args.seeds)
    tally = {f"{c} {s}": sum(r["ok"] for r in results if r["case"] == c and r["scene"] == s)
             for c in CASES for s in SCENES}
    (out / "result.json").write_text(json.dumps({"lora": str(args.lora), "strength": args.strength,
                                                 "scenes": SCENES, "checker_ok": tally, "pictures": results},
                                                indent=1))
    print(json.dumps(tally, indent=1))
    print(out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
