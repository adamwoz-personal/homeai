#!/usr/bin/env python3
"""Pictures for a concept LoRA: klein9 draws with the glossary's description, the checker keeps the right ones,
and each kept picture is captioned with the *plain* words (ART_FEATURES_PLAN F9 spike S4).

Why: a LoRA that works lets "fighting a chimera" draw a right chimera without the glossary's long description.
Training on pictures the glossary got right, captioned with the plain name, is how it learns that. Several
scenes, so it learns the creature rather than one picture. Strict: an unreadable answer drops the picture
(the live checker lets it pass). Claude still looks at the sheet before training.

Needs art mode (starts it if off and goes back to voice after; leaves it on if it was on). Nothing is spoken.
Writes ~/homeai-bench/lora-set-<time>/{<scene>-<seed>.png,data.jsonl,checks.json,kept.png,dropped.png}.

Usage:
    .venv/bin/python tools/art/collect_lora_set.py [--seeds 12] [--first-seed 101]
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

SCENES = {
    "anime": "An anime girl wielding a katana fighting a chimera.",
    "knight": "A knight with a round shield fighting a chimera in a forest.",
    "cliff": "A chimera standing on a rocky cliff at sunset.",
    "ruins": "A chimera prowling through ancient ruins.",
}


def right(answers: list[tuple[W.Ask, str]]) -> bool:
    """Every answer read and as expected; an unreadable one counts as wrong (strict, unlike the live checker)."""
    for ask, answer in answers:
        got = CK.parse_yes(answer) if isinstance(ask.expect, bool) else CK.parse_count(answer)
        if got is None or got != ask.expect:
            return False
    return True


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--seeds", type=int, default=12)
    ap.add_argument("--first-seed", type=int, default=101, help="away from the bench's 1-8")
    args = ap.parse_args(argv)
    switcher = G.ModeSwitcher(G.System(), log=lambda line: print("  ", line, flush=True))
    if switcher.mode() == "coding":
        print("not from coding mode", file=sys.stderr)
        return 1
    started = switcher.mode() != "art"
    out = bench_path("lora-set")
    out.mkdir(parents=True)
    kept, dropped, rows, log = [], [], [], []
    try:
        if started:
            switcher.art()
        client = C.ComfyClient()
        for scene, plain in SCENES.items():
            prompt, asks = W.expand(plain)
            for seed in range(args.first_seed, args.first_seed + args.seeds):
                result = client.run(workflows.klein9(prompt, seed, 1024, 1024, prefix=f"bench/lora-set-{scene}"))
                target = out / f"{scene}-{seed}.png"
                shutil.copyfile(result.image, target)
                answers = [(a, CK.ollama_ask(target, a.question)) for a in asks]
                ok = right(answers)
                log.append({"image": target.name, "ok": ok, "answers": {a.kind: ans[:40] for a, ans in answers}})
                (kept if ok else dropped).append((f"{scene} {seed}", target))
                if ok:
                    rows.append({"image_path": str(target), "caption": plain})
                print(f"{scene} {seed}: {'kept' if ok else 'dropped'} {log[-1]['answers']}", flush=True)
    finally:
        if started:
            switcher.voice()
    (out / "data.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))
    (out / "checks.json").write_text(json.dumps({"scenes": SCENES, "pictures": log}, indent=1))
    if kept:
        sheet(kept, out / "kept.png", 8)
    if dropped:
        sheet(dropped, out / "dropped.png", 8)
    print(f"kept {len(kept)} of {len(log)}: {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
