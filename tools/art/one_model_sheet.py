#!/usr/bin/env python3
"""Lay one model's bake-off sheets out on a single page, for review beside its rivals.

Why: a run with one model gives a narrow sheet per scenario and seed, too many to open.
This stacks the named scenarios' seed sheets into ~/homeai-bench/<run>/all-<name>.png.

Usage:
    .venv/bin/python tools/art/one_model_sheet.py ~/homeai-bench/bakeoff-<time> [--scenario labels ...]
"""
from __future__ import annotations

import argparse
from pathlib import Path

from PIL import Image

CELL = 420


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("folder", type=Path)
    ap.add_argument("--scenario", action="append", required=True)
    ap.add_argument("--name", default="review")
    args = ap.parse_args()
    rows = []
    for scenario in args.scenario:
        sheets = [Image.open(args.folder / f"{scenario}-seed{n}.png").convert("RGB") for n in range(3)]
        rows.append([s.resize((CELL, CELL * s.height // s.width)) for s in sheets])
    page = Image.new("RGB", (CELL * 3, sum(max(s.height for s in row) for row in rows)))
    y = 0
    for row in rows:
        for x, cell in enumerate(row):
            page.paste(cell, (x * CELL, y))
        y += max(s.height for s in row)
    out = args.folder / f"all-{args.name}.png"
    page.save(out)
    print(out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
