#!/usr/bin/env python3
"""Contact sheets of motion-capture clips as Ani's mannequin, to pick the frames that become pose forms (F10).

Why: a clip has thousands of frames and most are in-between; the forms are chosen by eye. Each clip gets one
sheet of evenly spaced frames, retargeted to her skeleton (homeai/poses/forms.py) and drawn by
homeai/poses/render.py, labelled with the frame number. Optional --turn draws them turned (90 = side view).
Clips must be fetched first (tools/poses/fetch_cmu.py fetch ...).

Usage:
    .venv/bin/python tools/poses/sheet_clips.py 02_07 13_17 [--frames 24] [--turn 0] [--range 400:900]
Output: ~/homeai-bench/poses-<time>/<trial>.png
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

from PIL import Image, ImageDraw

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

from homeai.benchio import bench_path  # noqa: E402
from homeai.poses import forms as FM  # noqa: E402
from homeai.poses import mocap as M  # noqa: E402
from homeai.poses import render as R  # noqa: E402
from tools.poses.fetch_cmu import CACHE, clip_urls  # noqa: E402

CELL = (150, 195)


def picks(total: int, count: int, start: int = 1, end: int | None = None) -> list[int]:
    """``count`` frame numbers spread evenly over start..end (1-based, inclusive)."""
    end = min(end or total, total)
    if count <= 1:
        return [start]
    return sorted({round(start + i * (end - start) / (count - 1)) for i in range(count)})


def sheet(trial: str, numbers: list[int], turn: int, path: Path) -> None:
    urls = clip_urls(trial)
    skeleton, frames = M.load(CACHE / urls["asf"].rsplit("/", 1)[1], CACHE / urls["amc"].rsplit("/", 1)[1])
    columns = 8
    rows = (len(numbers) + columns - 1) // columns
    out = Image.new("RGB", (CELL[0] * columns, (CELL[1] + 16) * rows), "white")
    pen = ImageDraw.Draw(out)
    for i, number in enumerate(numbers):
        joints = FM.turned(FM.from_capture(FM.capture_frame(skeleton, frames, number)), turn)
        cell = R.draw([R.Placed(joints)], *CELL)
        x, y = (i % columns) * CELL[0], (i // columns) * (CELL[1] + 16)
        out.paste(cell, (x, y + 16))
        pen.text((x + 4, y + 2), f"{trial} f{number}", fill="black")
    out.save(path)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("trials", nargs="+")
    ap.add_argument("--frames", type=int, default=24)
    ap.add_argument("--turn", type=int, default=0)
    ap.add_argument("--range", default="", help="first:last frame, e.g. 400:900")
    args = ap.parse_args(argv)
    out = bench_path("poses")
    out.mkdir(parents=True, exist_ok=True)
    for trial in args.trials:
        urls = clip_urls(trial)
        total = len(M.parse_amc((CACHE / urls["amc"].rsplit("/", 1)[1]).read_text()))
        first, _, last = args.range.partition(":")
        numbers = picks(total, args.frames, int(first or 1), int(last) if last else None)
        sheet(trial, numbers, args.turn, out / f"{trial}.png")
        print(out / f"{trial}.png", f"({total} frames)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
