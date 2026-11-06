#!/usr/bin/env python3
"""Add up bake-off runs per model: warm speed, failures and images a safety filter blanked.

Why: a bake-off is split over several folders (voice on, voice off, reruns), and a
blanked image still counts as a success in results.json. Ideogram returned grey
"Image blocked by safety filter" cards for battle maps, which the timings hide.
Warm means every image but the first of each run on a model, because the first one
includes loading it. Prints a markdown table.

Usage:
    .venv/bin/python tools/art/report_bakeoff.py ~/homeai-bench/bakeoff-<time> [...]
"""
from __future__ import annotations

import argparse
import json
import statistics
from collections import defaultdict
from pathlib import Path

import numpy as np

FLAT_SHARE = 0.8  # a filter card is one grey; art never puts this much in one colour
NEAR = 10


def is_blank(pixels: np.ndarray) -> bool:
    """True when one mid grey covers most of the picture. White paper is fine: floor plans are mostly white.

    The card comes out of the VAE like any picture, so its grey wobbles a little.
    """
    flat = pixels.reshape(-1, pixels.shape[-1]).astype(np.int32)
    middle = np.median(flat, axis=0)
    grey = middle.max() - middle.min() <= NEAR and 40 <= middle.mean() <= 200
    return bool(grey and (np.abs(flat - middle).max(axis=1) <= NEAR).mean() >= FLAT_SHARE)


def summarise(runs: list[dict], blank=lambda image: False) -> dict[str, dict]:
    """Per model, from results.json dicts of ``{scenario: {model: [{seconds, image} or {error}]}}``."""
    models: dict[str, dict] = defaultdict(lambda: {"images": 0, "failed": 0, "blanked": defaultdict(int), "warm": []})
    for run in runs:
        seen = set()
        for scenario, by_model in run.items():
            for model, shots in by_model.items():
                row = models[model]
                for shot in shots:
                    if "seconds" not in shot or shot.get("error"):
                        row["failed"] += 1
                        continue
                    row["images"] += 1
                    if model in seen:
                        row["warm"].append(shot["seconds"])
                    seen.add(model)
                    if shot.get("image") and blank(shot["image"]):
                        row["blanked"][scenario] += 1
    return dict(models)


def table(models: dict[str, dict]) -> str:
    lines = ["| model | images | warm s (median) | failed | blanked |", "|---|---|---|---|---|"]
    for name, row in sorted(models.items(), key=lambda kv: statistics.median(kv[1]["warm"] or [1e9])):
        warm = f"{statistics.median(row['warm']):.1f}" if row["warm"] else "-"
        blanked = ", ".join(f"{k} {v}" for k, v in row["blanked"].items()) or "0"
        lines.append(f"| {name} | {row['images']} | {warm} | {row['failed']} | {blanked} |")
    return "\n".join(lines)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("folders", nargs="+", type=Path)
    args = ap.parse_args()
    from PIL import Image

    def blank(image: str) -> bool:
        path = Path(image)
        return path.exists() and is_blank(np.asarray(Image.open(path).convert("RGB").resize((256, 256), Image.NEAREST)))

    runs = [json.loads((folder / "results.json").read_text()) for folder in args.folders]
    print(table(summarise(runs, blank)))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
