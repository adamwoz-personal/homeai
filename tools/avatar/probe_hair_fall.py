#!/usr/bin/env python3
"""Show how the hair's fall changes around the head, and find bumps in its outline.

Why: Adam saw hair at the temples curving down too fast and "horns or cat ears"
on top, and suspected a hard change in gravity between regions. This prints each
strand property by root angle (0 top of the head, 90 brow level, more is lower
down) so a jump is a number, and counts peaks in the top outline of the hair at
rest. A hairline with no horns has one broad peak per side at most.

Usage:
    .venv/bin/python tools/avatar/probe_hair_fall.py [--version v1] [--step 15]
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

from homeai.avatar import assets, facegl, rig  # noqa: E402

BENCH_DIR = Path.home() / "homeai-bench"


def by_angle(strands: np.ndarray, angle: np.ndarray, step: float, h: float) -> list[dict]:
    """Median strand properties per band of root angle; ``strands`` holds one row per point at the strand's root."""
    rows = []
    for lo in np.arange(0.0, 150.0, step):
        pick = (angle >= lo) & (angle < lo + step)
        if not pick.any():
            continue
        s = strands[pick]
        rows.append({"angle": float(lo), "strands": int(pick.sum()), "droop": round(float(np.median(s[:, 9])), 2),
                     "length_h": round(float(np.median(s[:, 3])) / h, 2), "bend": round(float(np.median(s[:, 10])), 2),
                     "aim_x": round(float(np.median(np.abs(s[:, 7]))), 2),
                     "aim_y": round(float(np.median(s[:, 8])), 2)})
    return rows


def outward_before_falling(layout: np.ndarray, angle: np.ndarray, step: float, h: float) -> list[dict]:
    """How far sideways each band's strands get (face heights) by the time they fall back to root height."""
    rest = facegl.hair_rest(layout)
    key = np.round(layout[:, :2], 3)
    _, strand = np.unique(key, axis=0, return_inverse=True)
    strand = strand.ravel()
    rows = {}
    for i in np.unique(strand):
        pts = np.flatnonzero(strand == i)
        pts = pts[np.argsort(layout[pts, 4])]
        below = np.flatnonzero(rest[pts, 1] > layout[pts[0], 1] + 0.02 * h)
        at = pts[below[0]] if len(below) else pts[-1]
        rows.setdefault(float(step * (angle[pts[0]] // step)), []).append(abs(rest[at, 0] - layout[at, 0]) / h)
    return [{"angle": k, "out_h": round(float(np.median(v)), 2)} for k, v in sorted(rows.items())]


def outline_peaks(rest: np.ndarray, centre: float, h: float) -> list[dict]:
    """Bumps in the hair's top outline: where it rises above both neighbours by more than 3% of a face height."""
    q = (rest[:, 0] - centre) / h
    bins = np.arange(-3.0, 3.01, 0.05)
    tops = np.full(len(bins) - 1, np.nan)
    which = np.digitize(q, bins) - 1
    for i in range(len(tops)):
        here = rest[which == i, 1]
        if len(here) > 20:
            tops[i] = np.percentile(here, 1) / h
    x = (bins[:-1] + bins[1:]) / 2
    ok = ~np.isnan(tops)
    x, tops = x[ok], tops[ok]
    tops = np.convolve(tops, np.ones(3) / 3, mode="same")
    peaks = []
    for i in range(6, len(x) - 6):
        if tops[i] == tops[i - 6:i + 7].min():
            drop = min(tops[i - 6:i].max(), tops[i + 1:i + 7].max()) - tops[i]
            if drop > 0.03:
                peaks.append({"x_h": round(float(x[i]), 2), "stands_out_h": round(float(drop), 3)})
    return peaks


def pile_up(rest: np.ndarray, h: float) -> float:
    """How bright the worst ridge is: the busiest cell of a 0.05 face-height grid over the median busy cell."""
    cells = np.unique(np.floor(rest / (0.05 * h)).astype(np.int64), axis=0, return_counts=True)[1]
    return round(float(cells.max() / np.median(cells)), 1)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--version", help="asset version (default: current)")
    ap.add_argument("--step", type=float, default=15.0)
    args = ap.parse_args()
    asset = assets.load(assets.asset_root() / args.version) if args.version else assets.load_current()
    if asset is None:
        print("no face asset: none chosen (homeai-avatar use vN)", file=sys.stderr)
        return 2
    layout = facegl.asset_hair(asset, facegl.point_spacing(asset.xy)).astype(np.float64)
    v = np.asarray(asset.verts, dtype=np.float64)
    h = rig.face_height(v)
    top = float(min(v[105, 1], v[334, 1]))
    centre = float(v[list(rig.REGIONS["jaw"] + rig.REGIONS["forehead"]), 0].mean())
    d = layout[:, :2] - (centre, top)
    angle = np.degrees(np.arctan2(np.abs(d[:, 0]), -d[:, 1]))
    first = np.unique(np.round(layout[:, :2], 3), axis=0, return_index=True)[1]
    result = {"properties": by_angle(layout[first], angle[first], args.step, h),
              "out_before_falling": outward_before_falling(layout, angle, args.step, h),
              "outline_peaks": outline_peaks(facegl.hair_rest(layout), centre, h),
              "pile_up": pile_up(facegl.hair_rest(layout), h)}
    for row in result["properties"]:
        print(row)
    print("out before falling:", " ".join(f"{r['angle']:.0f}:{r['out_h']}" for r in result["out_before_falling"]))
    print("outline peaks:", result["outline_peaks"], " pile-up:", result["pile_up"])
    out = BENCH_DIR / f"hair-fall-{datetime.now():%Y%m%d-%H%M%S}.json"
    out.write_text(json.dumps(result, indent=1))
    print(out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
