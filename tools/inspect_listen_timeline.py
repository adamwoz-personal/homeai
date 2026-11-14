"""When was the room loud, and what with? A timeline of the listen probe's clips, in time buckets.

Why: interactive mode step 3 (TV detection) needs labelled TV stretches. Adam says when the TV was on ("a show
earlier this evening"); this shows per bucket how many clips, seconds of sound, how many hit the 30 s maximum
(continuous sound, typical of TV), share that were noise, and a sample of the words, so the TV window can be
pinned down and written to ~/homeai-bench/tv-labels.jsonl by hand or with --label.

Usage: .venv/bin/python tools/inspect_listen_timeline.py [YYYYMMDD] [--minutes 10]
       .venv/bin/python tools/inspect_listen_timeline.py --label tv "2026-11-13 19:00" "2026-11-13 20:10" [note]
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
import sys
import time
from collections import defaultdict
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from homeai import listen_probe  # noqa: E402

LABELS = Path.home() / "homeai-bench" / "tv-labels.jsonl"


def clips(folder: Path, day: str) -> list[dict]:
    out = []
    for path in sorted(folder.glob(f"{day}-*.json")):
        try:
            meta = json.loads(path.read_text())
        except (OSError, ValueError):
            continue
        out.append(meta)
    return out


def timeline(rows: list[dict], minutes: int, max_s: float = listen_probe.Segmenter.max_s) -> list[dict]:
    buckets: dict[int, list[dict]] = defaultdict(list)
    for r in rows:
        if "start" in r:
            buckets[int(r["start"] // (minutes * 60))].append(r)
    out = []
    for key in sorted(buckets):
        group = buckets[key]
        stt = [g.get("stt") or {} for g in group]
        words = " / ".join(s["text"] for s in stt if s.get("ok"))
        out.append({"at": key * minutes * 60, "clips": len(group),
                    "sound_s": round(sum(float(g.get("duration") or 0) for g in group)),
                    "full": sum(1 for g in group if float(g.get("duration") or 0) >= max_s - 0.5),
                    "noise": sum(1 for s in stt if s and not s.get("ok")), "words": words})
    return out


def main(argv: list[str]) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n", 1)[0])
    ap.add_argument("day", nargs="?", default=time.strftime("%Y%m%d"))
    ap.add_argument("--minutes", type=int, default=10)
    ap.add_argument("--label", nargs="+", metavar=("KIND START END", "NOTE"))
    ap.add_argument("--folder", type=Path, default=listen_probe.DIR)
    args = ap.parse_args(argv)
    if args.label:
        kind, start, end, *note = args.label
        row = {"kind": kind, "start": dt.datetime.fromisoformat(start).timestamp(),
               "end": dt.datetime.fromisoformat(end).timestamp(), "note": " ".join(note), "labelled": time.time()}
        LABELS.parent.mkdir(exist_ok=True)
        with LABELS.open("a") as f:
            f.write(json.dumps(row) + "\n")
        print(f"labelled {kind} {start} - {end} -> {LABELS}")
        return 0
    for b in timeline(clips(args.folder, args.day), args.minutes):
        when = time.strftime("%H:%M", time.localtime(b["at"]))
        print(f"{when} clips={b['clips']:3} sound={b['sound_s']:4}s full30={b['full']:2} noise={b['noise']:3}  "
              f"{b['words'][:110]}")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
