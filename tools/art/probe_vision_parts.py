#!/usr/bin/env python3
"""Can a local vision model check blades and body parts, not just heads and riders? (ART_FEATURES_PLAN F8 step 2)

Why: on 2026-11-18 klein9 drew a dragon for "a chimera" and two swords for "a katana", and the F4 checker passed
them all, because it only counts heads and riders. Before the checklist asks "How many sword blades are
visible?" or "Is there a goat's head on the creature's back?", this measures how often the vision model agrees
with labels given by eye.

Labels file (JSON): {"labels": [{"image": "path.png", "question": "...", "expect": 1 | true | false,
"type": "blades"}, ...]}. A number is compared with the first number in the answer; true/false with yes/no.

Ollama keeps one model loaded, so this unloads Jarvis's voice model while it runs (as each art check does).
Read-only otherwise; nothing is spoken. Writes ~/homeai-bench/vision-parts-<time>/result.json.

Usage:
    .venv/bin/python tools/art/probe_vision_parts.py --sheet DIR_WITH_PNGS     (numbered contact sheet, to label)
    .venv/bin/python tools/art/probe_vision_parts.py LABELS.json [--model minicpm-v] [--repeat 2]
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from collections import defaultdict
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

from homeai.art import checker as CK  # noqa: E402

BENCH = Path.home() / "homeai-bench"


def score(answer: str, expect) -> bool | None:
    """True if the answer matches, False if it doesn't, None if it can't be read."""
    if isinstance(expect, bool):
        got = CK.parse_yes(answer)
    else:
        got = CK.parse_count(answer)
    return None if got is None else got == expect


def expand(data: dict) -> list[dict]:
    """The labels file's rows; the compact form {"sets": {name: {"dir", "names", <type>: [expect...]}},
    "questions": {<type>: question}} becomes one row per picture and type."""
    rows = list(data.get("labels") or [])
    for spec in (data.get("sets") or {}).values():
        for kind, question in data["questions"].items():
            for name, expect in zip(spec["names"], spec.get(kind) or []):
                rows.append({"image": f"{spec['dir']}/{name}.png", "question": question, "expect": expect,
                             "type": kind})
    return rows


def sheet(folder: Path, columns: int = 5, cell: int = 320, pattern: str = "*.png") -> Path:
    from PIL import Image, ImageDraw
    pngs = sorted(p for p in folder.glob(pattern) if p.name != "sheet.png")
    rows = (len(pngs) + columns - 1) // columns
    out = Image.new("RGB", (columns * cell, rows * cell), "white")
    draw = ImageDraw.Draw(out)
    for i, png in enumerate(pngs):
        x, y = (i % columns) * cell, (i // columns) * cell
        out.paste(Image.open(png).convert("RGB").resize((cell, cell)), (x, y))
        draw.rectangle([x, y, x + 60, y + 28], fill="black")
        draw.text((x + 6, y + 6), png.stem, fill="white")
    tag = pattern.replace("*", "").replace(".png", "")
    path = BENCH / f"vision-parts-sheet-{folder.name[:40]}{'-' + tag if tag else ''}.png"
    out.save(path)
    return path


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n", 1)[0])
    ap.add_argument("labels", type=Path, nargs="?")
    ap.add_argument("--sheet", type=Path)
    ap.add_argument("--glob", default="*.png", help="with --sheet: which pictures (e.g. 'v1-*.png')")
    ap.add_argument("--columns", type=int, default=5)
    ap.add_argument("--cell", type=int, default=320)
    ap.add_argument("--model", default=CK.MODEL)
    ap.add_argument("--repeat", type=int, default=1, help="ask each question this many times (temperature 0)")
    args = ap.parse_args(argv)
    if args.sheet:
        print(sheet(args.sheet, args.columns, args.cell, args.glob))
        return 0
    labels = expand(json.loads(args.labels.read_text()))
    rows, by_type = [], defaultdict(lambda: [0, 0, 0])      # right, wrong, unreadable
    for label in labels:
        for _ in range(args.repeat):
            t0 = time.monotonic()
            answer = CK.ollama_ask(Path(label["image"]).expanduser(), label["question"], args.model)
            ok = score(answer, label["expect"])
            rows.append({**label, "answer": answer, "ok": ok, "s": round(time.monotonic() - t0, 2)})
            by_type[label.get("type", "?")][0 if ok else 1 if ok is False else 2] += 1
            print(f"{Path(label['image']).stem} {label.get('type')}: {ok} ({answer[:60]!r})", flush=True)
    out = BENCH / f"vision-parts-{time.strftime('%Y%m%d-%H%M%S')}"
    out.mkdir(parents=True)
    summary = {t: {"right": r, "wrong": w, "unreadable": u} for t, (r, w, u) in by_type.items()}
    (out / "result.json").write_text(json.dumps({"model": args.model, "summary": summary, "rows": rows}, indent=1))
    for t, s in summary.items():
        print(f"{t:12} right {s['right']}  wrong {s['wrong']}  unreadable {s['unreadable']}")
    print(out)
    from homeai.art.voice import _warm_voice_model     # put Jarvis's voice model back
    _warm_voice_model()
    return 0


if __name__ == "__main__":
    sys.exit(main())
