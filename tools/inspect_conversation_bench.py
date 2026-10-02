#!/usr/bin/env python3
# Copyright (c) 2026 Adam Wosotowsky
# SPDX-License-Identifier: BUSL-1.1
# Licensed under the Business Source License 1.1; see LICENSE at the
# repository root. Converts to Apache-2.0 on 2030-09-15.
"""Read results written by ``bench_conversation.py --out``.

Two jobs:

* **compare** several runs side by side, so a change is judged on the same
  metrics as the baseline;
* **show** the actual conversations behind bad numbers. A metric says a reply
  repeated itself; only the text says *why* (that is how ZeroClaw's
  memory_recall leaking old answers was found on 2026-10-02).

Usage
-----
    inspect_conversation_bench.py compare ~/homeai-bench/conv-baseline-*.json ~/homeai-bench/conv-after5-*.json
    inspect_conversation_bench.py show RUN.json                 # flagged conversations
    inspect_conversation_bench.py show RUN.json --echo 0.4      # stricter/looser echo flag
    inspect_conversation_bench.py show RUN.json --all --script free_will
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
from pathlib import Path


def load(path: str) -> list[dict]:
    return json.loads(Path(path).expanduser().read_text(encoding="utf-8"))


def summary(rows: list[dict]) -> dict[str, str]:
    ok = [r for r in rows if r.get("ok")]
    later = [r for r in ok if r["turn"] > 1]
    if not ok:
        return {"turns": f"{len(rows)} (all failed)"}
    out = {
        "turns": str(len(rows)),
        "median spoken s": f"{statistics.median(r['spoken_s'] for r in ok):.1f}",
        "max spoken s": f"{max(r['spoken_s'] for r in ok):.1f}",
        "over 60s": f"{sum(r['spoken_s'] > 60 for r in ok)}/{len(ok)}",
        "sourced": f"{sum(bool(r['sourced']) for r in ok)}/{len(ok)}",
        "hedged": f"{sum(bool(r['hedges']) for r in ok)}/{len(ok)}",
        "clarify": f"{sum(bool(r['clarify']) for r in ok)}/{len(ok)}",
    }
    if later:
        out["median echo"] = f"{statistics.median(r['echo'] for r in later):.2f}"
        out["max echo"] = f"{max(r['echo'] for r in later):.2f}"
    if ok and "budget_s" in ok[0]:
        out["max budgeted s"] = f"{max(r['budget_s'] for r in ok):.1f}"
    return out


def cmd_compare(paths: list[str]) -> int:
    runs = [(Path(p).stem, summary(load(p))) for p in paths]
    keys = list(dict.fromkeys(k for _, s in runs for k in s))
    width = max(14, *(len(name) for name, _ in runs))
    print(f"{'':16}" + "".join(f"{name[:width]:>{width + 2}}" for name, _ in runs))
    for key in keys:
        print(f"{key:16}" + "".join(f"{s.get(key, '-'):>{width + 2}}" for _, s in runs))
    return 0


def flagged(row: dict, echo: float) -> bool:
    return row["echo"] > echo or bool(row["hedges"]) or bool(row["sourced"])


def cmd_show(path: str, echo: float, show_all: bool, script: str | None, width: int) -> int:
    convs: dict[tuple, list[dict]] = {}
    for row in load(path):
        if script and row["script"] != script:
            continue
        convs.setdefault((row["script"], row["rep"]), []).append(row)
    shown = 0
    for (name, rep), rows in convs.items():
        if not show_all and not any(flagged(r, echo) for r in rows):
            continue
        shown += 1
        print(f"===== {name} rep {rep}")
        for r in rows:
            marks = [f"echo={r['echo']}"] + r["hedges"] + (["SOURCED"] if r["sourced"] else [])
            print(f" t{r['turn']} {r['spoken_s']:.0f}s {' '.join(marks)}")
            print(f"  Q: {r['asked']}")
            print(f"  A: {(r['reply'] or r.get('error', ''))[:width]}")
        print()
    print(f"{shown} conversation(s) shown")
    return 0


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    c = sub.add_parser("compare", help="side-by-side metrics for several runs")
    c.add_argument("runs", nargs="+")
    s = sub.add_parser("show", help="print conversations behind the numbers")
    s.add_argument("run")
    s.add_argument("--echo", type=float, default=0.4, help="flag replies above this echo")
    s.add_argument("--all", action="store_true", help="show every conversation")
    s.add_argument("--script", help="only this script, e.g. free_will")
    s.add_argument("--width", type=int, default=500, help="characters of each reply")
    args = ap.parse_args(argv)
    if args.cmd == "compare":
        return cmd_compare(args.runs)
    return cmd_show(args.run, args.echo, args.all, args.script, args.width)


if __name__ == "__main__":
    sys.exit(main())
