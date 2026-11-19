#!/usr/bin/env python3
"""The art review queue as a numbered contact sheet plus a labels file for Claude to fill (ART_FEATURES_PLAN F9).

Why: the local checker misses placement and proportion (node 002 passed with horns on the lion), so pictures
Adam complained about, pictures shown with "Best I could do", 1 in 10 passes and practice-night glossary
candidates wait in ~/.local/share/homeai/art/review/ (homeai/art/review.py). In a session Claude looks at the
sheet, fills each item's "label" in labels.json, and applies it:

    label = {"right": true | false,              (the picture as a whole, by eye)
             "parts": {"<check kind>": <what it really shows: true/false or a number>, ...},
             "notes": "what's wrong in words (feeds glossary changes)"}

Applying moves each labelled item to review/done/ and writes vision-labels.json: one row per labelled part in
tools/art/probe_vision_parts.py's format, so the checker's accuracy grows with every review (a glossary question
needs 30 labelled pictures before it may fail a picture). Items left with "right": null stay waiting.

Usage:
    .venv/bin/python tools/art/report_review.py [--max 40]              -> ~/homeai-bench/review-<time>/
    .venv/bin/python tools/art/report_review.py --apply ~/homeai-bench/review-<time>/labels.json
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

from homeai.art import review as R  # noqa: E402
from homeai.benchio import bench_path  # noqa: E402

HOW = ("Fill each item's label: right true/false by eye; parts {check kind: what the picture really shows}; notes "
       "in words. Leave right null to keep it waiting. Then --apply this file.")


def build(review: R.Review, out: Path, most: int = 40) -> tuple[Path, int]:
    """labels.json (+ sheet.png when there is anything) for up to ``most`` waiting items, oldest first."""
    rows, items = [], []
    for folder in review.waiting()[:most]:
        item = R.read(folder)
        if item is None:
            continue
        n = len(items) + 1
        items.append({"n": n, "folder": str(folder), "reason": item.get("reason"), "source": item.get("source"),
                      "said": item.get("said", ""), "prompt": item.get("prompt", ""),
                      "issues": item.get("issues", []),
                      "checks": [{k: c.get(k) for k in ("kind", "question", "expect")} for c in item.get("checks", [])],
                      **({"candidate": item["candidate"]} if "candidate" in item else {}),
                      "label": {"right": None, "parts": {}, "notes": ""}})
        rows.append((f"{n} {item.get('reason', '')}", folder / "picture.png"))
    out.mkdir(parents=True, exist_ok=True)
    (out / "labels.json").write_text(json.dumps({"how": HOW, "items": items}, indent=1))
    if rows:
        from tools.art.bench_rider import sheet
        sheet(rows, out / "sheet.png", 5)
    return out, len(items)


def apply(labels: Path, review: R.Review) -> tuple[int, int]:
    """(items moved to done, vision label rows written)."""
    data = json.loads(Path(labels).read_text())
    vision = []
    moved = 0
    for entry in data.get("items", []):
        label = entry.get("label") or {}
        if label.get("right") is None:
            continue
        folder = Path(entry["folder"])
        if not (folder / "item.json").exists():
            continue                            # applied before
        questions = {c["kind"]: c["question"] for c in entry.get("checks", [])}
        done = review.done(folder, label)
        moved += 1
        for kind, expect in (label.get("parts") or {}).items():
            if kind in questions and isinstance(expect, (bool, int)):
                vision.append({"image": str(done / "picture.png"), "question": questions[kind], "expect": expect,
                               "type": kind})
    target = Path(labels).with_name("vision-labels.json")
    old = json.loads(target.read_text()).get("labels", []) if target.exists() else []
    target.write_text(json.dumps({"labels": old + vision}, indent=1))
    return moved, len(vision)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--apply", type=Path, help="a filled labels.json")
    ap.add_argument("--max", type=int, default=40)
    ap.add_argument("--root", type=Path, default=R.ROOT)
    args = ap.parse_args(argv)
    review = R.Review(args.root)
    if args.apply:
        moved, rows = apply(args.apply, review)
        print(f"{moved} labelled items moved to {review.root / 'done'}; {rows} checker labels in "
              f"{args.apply.with_name('vision-labels.json')}")
        return 0
    out, n = build(review, bench_path("review"), args.max)
    print(f"{n} of {len(review.waiting())} waiting items: {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
