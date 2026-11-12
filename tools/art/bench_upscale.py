"""Time the art-mode 2x upscale (ART_FEATURES_PLAN F7d) on this box, with whatever else is loaded.

Why: the upscale goes through the same one-at-a-time art queue as drawing, so its time and
memory decide whether it can share the card with klein 9B and Jarvis's voice model.
Runs workflows.upscale on one picture N times through the real ComfyUI and records seconds,
output size and VRAM used (rocm-smi) before and after.

Usage: .venv/bin/python tools/art/bench_upscale.py PICTURE.png [--runs 3]
Results: ~/homeai-bench/upscale-<stamp>.json
"""
from __future__ import annotations

import argparse
import json
import re
import subprocess
import time
from pathlib import Path

from PIL import Image

from homeai.art import workflows
from homeai.art.comfy import ComfyClient


def vram_used_mb() -> int | None:
    try:
        out = subprocess.run(["rocm-smi", "--showmeminfo", "vram"], capture_output=True, text=True, timeout=10).stdout
    except (OSError, subprocess.SubprocessError):
        return None
    used = re.findall(r"Used Memory \(B\):\s*(\d+)", out)
    return round(sum(int(u) for u in used) / 2**20) if used else None


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("picture", type=Path)
    ap.add_argument("--runs", type=int, default=3)
    args = ap.parse_args()
    client = ComfyClient()
    staged = workflows.stage(args.picture, f"bench-{args.picture.name}")
    runs = []
    for i in range(args.runs):
        before = vram_used_mb()
        start = time.monotonic()
        result = client.run(workflows.upscale(staged, prefix="bench/upscale"))
        seconds = round(time.monotonic() - start, 2)
        after = vram_used_mb()
        with Image.open(result.image) as im:
            size = list(im.size)
        runs.append({"run": i + 1, "seconds": seconds, "comfy_seconds": result.seconds, "size": size,
                     "vram_before_mb": before, "vram_after_mb": after, "image": str(result.image)})
        print(runs[-1])
    with Image.open(args.picture) as im:
        source = list(im.size)
    out = Path.home() / "homeai-bench" / f"upscale-{time.strftime('%Y%m%d-%H%M%S')}.json"
    out.write_text(json.dumps({"picture": str(args.picture), "source_size": source, "model": workflows.UPSCALER,
                               "runs": runs}, indent=1))
    print(out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
