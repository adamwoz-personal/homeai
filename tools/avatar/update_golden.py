#!/usr/bin/env python3
"""Regenerate the avatar golden images (tests/golden/avatar/<state>.png).

Why: tests/test_avatar_golden.py compares a headless render of every shown
state against these PNGs, so a shader or scene change that alters how a state
looks is caught. Run this ONLY after deliberately changing the look, look at
the PNGs (or run tools/avatar/state_tour.py), and commit them with the change.
Prints the difference from the previous images so the change is visible.

Usage:
    .venv/bin/python tools/avatar/update_golden.py [--state thinking ...] [--pose AA ...]
        [--only sphere|face] [--check]
Face goldens (tests/golden/avatar/face/<pose>.png) use a synthetic face built
from the landmark fixture, so they never depend on the chosen artwork.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

import numpy as np  # noqa: E402
from PIL import Image  # noqa: E402

from homeai.avatar.tour import (FACE_TOUR, GOLDEN_POINTS, GOLDEN_SIZE, TOUR, compare,  # noqa: E402
                                render_face_golden, render_states)

GOLDEN_DIR = REPO / "tests" / "golden" / "avatar"
FACE_DIR = GOLDEN_DIR / "face"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--state", action="append", choices=TOUR, help="default: all")
    parser.add_argument("--pose", action="append", choices=FACE_TOUR, help="face poses (default: all)")
    parser.add_argument("--only", choices=("sphere", "face"), help="default: both looks")
    parser.add_argument("--check", action="store_true", help="compare only; write nothing")
    args = parser.parse_args()

    FACE_DIR.mkdir(parents=True, exist_ok=True)
    jobs = []
    if args.only != "face":
        jobs.append((GOLDEN_DIR, render_states(args.state or TOUR, GOLDEN_SIZE, GOLDEN_POINTS)))
    if args.only != "sphere":
        jobs.append((FACE_DIR, render_face_golden(args.pose or FACE_TOUR)))
    failed = 0
    for folder, renders in jobs:
        for name, rgb in renders:
            path = folder / f"{name}.png"
            result = {"golden": str(path.relative_to(REPO))}
            if path.exists():
                result.update(compare(np.asarray(Image.open(path).convert("RGB")), rgb))
            else:
                result["new"] = True
            failed += not result.get("ok", True)
            if not args.check:
                Image.fromarray(rgb).save(path)
                result["wrote"] = True
            print(json.dumps(result))
    return 1 if args.check and failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
