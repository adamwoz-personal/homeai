#!/usr/bin/env python3
"""Regenerate the avatar golden images (tests/golden/avatar/<state>.png).

Why: tests/test_avatar_golden.py compares a headless render of every shown
state against these PNGs, so a shader or scene change that alters how a state
looks is caught. Run this ONLY after deliberately changing the look, look at
the PNGs (or run tools/avatar/state_tour.py), and commit them with the change.
Prints the difference from the previous images so the change is visible.

Usage:
    .venv/bin/python tools/avatar/update_golden.py [--state thinking ...] [--check]
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

from homeai.avatar.tour import (GOLDEN_POINTS, GOLDEN_SIZE, TOUR, compare,  # noqa: E402
                                render_states)

GOLDEN_DIR = REPO / "tests" / "golden" / "avatar"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--state", action="append", choices=TOUR, help="default: all")
    parser.add_argument("--check", action="store_true", help="compare only; write nothing")
    args = parser.parse_args()

    GOLDEN_DIR.mkdir(parents=True, exist_ok=True)
    failed = 0
    for name, rgb in render_states(args.state or TOUR, GOLDEN_SIZE, GOLDEN_POINTS):
        path = GOLDEN_DIR / f"{name}.png"
        result = {"state": name}
        if path.exists():
            result.update(compare(np.asarray(Image.open(path).convert("RGB")), rgb))
        else:
            result["new"] = True
        failed += not result.get("ok", True)
        if not args.check:
            Image.fromarray(rgb).save(path)
            result["wrote"] = str(path.relative_to(REPO))
        print(json.dumps(result))
    return 1 if args.check and failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
