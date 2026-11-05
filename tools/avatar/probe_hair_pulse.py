#!/usr/bin/env python3
"""Show and measure a speech pulse travelling down the aura hair.

Why: Adam asked for speech to flash the roots and travel down the strands
instead of flashing the whole hair. A single render can't show that, so this
renders one burst at several delays, prints how far along the strands the light
sits (brightness only, so sway can't blur the number), and writes a contact sheet
of the full effect to ~/homeai-bench/hair-pulse-<time>/. Arc should grow with the
delay at about HAIR_PULSE_SPEED; a whole-hair flash would hold it constant.
No voice, daemon or window is used.

Usage:
    .venv/bin/python tools/avatar/probe_hair_pulse.py [--version v1] [--size 480]
        [--delays 0.1 0.3 0.6 1.0 1.4]
"""
from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

from homeai.avatar import assets, tour  # noqa: E402

BENCH_DIR = Path.home() / "homeai-bench"


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--version", help="asset version (default: current)")
    ap.add_argument("--size", type=int, default=480)
    ap.add_argument("--delays", type=float, nargs="+", default=[0.1, 0.3, 0.6, 1.0, 1.4])
    args = ap.parse_args()
    asset = assets.load(assets.asset_root() / args.version) if args.version else assets.load_current()
    if asset is None:
        print("no face asset: none chosen (homeai-avatar use vN)", file=sys.stderr)
        return 2
    from PIL import Image

    size = (args.size, args.size)
    silent, lit, rest, arc = tour.render_pulse(asset, args.delays, size, low=0.0)
    rows = []
    for delay, rgb in lit.items():
        where, share = tour.pulse_arc(rgb, silent, rest, arc)
        rows.append({"delay_s": delay, "arc_face_heights": round(where, 3), "changed_share": round(share, 4)})
        print(f"{delay:5.2f} s  arc {where:.2f} face heights  changed {100 * share:5.1f} %")
    silent, shots, _, _ = tour.render_pulse(asset, args.delays, size)
    out = BENCH_DIR / f"hair-pulse-{datetime.now():%Y%m%d-%H%M%S}"
    out.mkdir(parents=True)
    Image.fromarray(np.hstack([silent, *shots.values()])).save(out / "contact.png")
    (out / "results.json").write_text(json.dumps(rows, indent=1))
    print(out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
