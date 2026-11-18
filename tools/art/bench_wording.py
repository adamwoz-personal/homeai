#!/usr/bin/env python3
"""Does describing a thing instead of naming it fix chimeras and double-bladed katanas? (ART_FEATURES_PLAN F8 step 1)

Why: on 2026-11-18 Adam's "an anime girl wielding a katana fighting a chimera" came out 10/10 wrong with klein9:
a dragon instead of a chimera (lion + goat head on its back + snake tail) every time, and two blades every time
(two swords, or one hilt with a blade out of each end). Words about counts ("one katana with one blade") didn't
help. This draws the scene with the old prompt and with descriptive wordings, several seeds each, and writes a
contact sheet to count defects by eye (then by tools/art/probe_vision_parts.py).

Needs ComfyUI up (art mode). Starts art mode if it's off and goes back to voice after; leaves it on if it was
already on. Nothing is spoken.

Writes ~/homeai-bench/wording-<time>/{result.json,sheet.png,<case>-<seed>.png}.

Usage:
    .venv/bin/python tools/art/bench_wording.py [--seeds 8] [--case old --case v1 ...]
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
from homeai.art import workflows  # noqa: E402
from homeai.benchio import bench_path  # noqa: E402
from tools.art.bench_rider import sheet  # noqa: E402

CHIMERA = ("a chimera: a lion with a golden mane, a goat's head with curved horns growing from the middle of its "
           "back, and a living snake as its tail")
CASES = {
    # As sent in Adam's session (node 004).
    "old": "An anime girl wielding a katana. Fighting a chimera.",
    # Describe the chimera; one katana, one hilt, one blade.
    "v1": f"An anime girl holding a single katana: one hilt and one long curved blade. Fighting {CHIMERA}.",
    # As v1, plus where her hands go (the two-handed grip is where the second blade came from).
    "v2": f"An anime girl holding a single katana with both hands on its one hilt, its one long curved blade "
          f"pointing at the monster. Fighting {CHIMERA}.",
    # v2's sword; the chimera without a colon (v1/v2 came out as plain white character sheets) and its goat's
    # head named as a second head (v1/v2 mostly drew goat horns on the lion's head instead).
    "v3": "An anime girl holding a single katana with both hands on its one hilt, its one long curved blade "
          "pointing at the monster. Fighting a chimera, a lion with a golden mane that has a second head, a "
          "goat's head, rising from the middle of its back, and a living snake for a tail.",
    # Adam 2026-11-18: a katana can be held in one hand or two, so the words mustn't say "both hands".
    "v4": "An anime girl holding a single katana by its one hilt, its one long curved blade pointing forward. "
          "Fighting a chimera, a lion with a golden mane that has a second head, a goat's head, rising from the "
          "middle of its back, and a living snake for a tail.",
}


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--seeds", type=int, default=8)
    ap.add_argument("--case", action="append", choices=sorted(CASES), help="only these cases (default all)")
    args = ap.parse_args(argv)
    cases = {k: v for k, v in CASES.items() if not args.case or k in args.case}
    switcher = G.ModeSwitcher(G.System(), log=lambda line: print("  ", line, flush=True))
    if switcher.mode() == "coding":
        print("not from coding mode", file=sys.stderr)
        return 1
    started = switcher.mode() != "art"
    out = bench_path("wording")
    out.mkdir(parents=True)
    rows, results = [], []
    try:
        if started:
            switcher.art()
        client = C.ComfyClient()
        for name, prompt in cases.items():
            for seed in range(1, args.seeds + 1):
                start = time.monotonic()
                result = client.run(workflows.klein9(prompt, seed, 1024, 1024, prefix=f"bench/wording-{name}"))
                target = out / f"{name}-{seed}.png"
                shutil.copyfile(result.image, target)
                rows.append((f"{name} seed {seed}", target))
                results.append({"case": name, "seed": seed, "seconds": round(time.monotonic() - start, 1)})
                print(f"{name} seed {seed}: {results[-1]['seconds']}s", flush=True)
    finally:
        if started:
            switcher.voice()
    sheet(rows, out / "sheet.png", args.seeds)
    (out / "result.json").write_text(json.dumps({"prompts": cases, "runs": results,
                                                 "defects": "counted by eye on sheet.png"}, indent=1))
    print(out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
