#!/usr/bin/env python3
# Copyright (c) 2026 Adam Wosotowsky
# SPDX-License-Identifier: BUSL-1.1
"""Calibrate the recall similarity floor (HOMEAI_LTM_MIN_SIM).

For each fixture query, prints the cosine similarity of its relevant memory
and of the best irrelevant one (fixtures plus real transcript exchanges as
distractors), then suggests a floor that keeps most relevant hits while
dropping most "best wrong answer" noise.

    .venv/bin/python tools/memory/sim_floor.py [--model nomic-embed-text-v1.5]
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from homeai.memory_store.embed import DEFAULT_MODEL, OnnxEmbedder  # noqa: E402

FIXTURES = ROOT / "tests" / "fixtures" / "memory_recall.json"
TRANSCRIPT = Path("~/.local/share/homeai/transcript.jsonl").expanduser()


def real_exchanges(limit: int = 400) -> list[str]:
    out = []
    if not TRANSCRIPT.exists():
        return out
    for line in TRANSCRIPT.read_text(encoding="utf-8").splitlines():
        try:
            t = json.loads(line)
        except ValueError:
            continue
        if t.get("reply") and t.get("verdict") in ("allow", "continued"):
            out.append(f"{t.get('heard', '')}\n{t['reply']}")
    return out[-limit:]


def pct(a: np.ndarray, q: float) -> float:
    return float(np.percentile(a, q)) if len(a) else float("nan")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--model", default=DEFAULT_MODEL)
    args = ap.parse_args()

    fx = json.loads(FIXTURES.read_text(encoding="utf-8"))
    mems = {m["id"]: m["text"] for m in fx["memories"]}
    ids = list(mems)
    distract = real_exchanges()
    emb = OnnxEmbedder(args.model)
    docs = emb.embed([mems[i] for i in ids] + distract, kind="doc")
    queries = emb.embed([q["q"] for q in fx["queries"]], kind="query")
    sims = queries @ docs.T

    rel, wrong = [], []
    for qi, q in enumerate(fx["queries"]):
        relevant = {ids.index(r) for r in q["relevant"]}
        rel.extend(sims[qi, j] for j in relevant)
        wrong.append(max(sims[qi, j] for j in range(docs.shape[0]) if j not in relevant))
    # Off-topic chatter: real exchanges vs fixture queries are almost all unrelated.
    rel, wrong = np.array(rel), np.array(wrong)

    print(f"model {args.model}; {len(ids)} memories + {len(distract)} real exchanges")
    for name, a in (("relevant", rel), ("best irrelevant", wrong)):
        print(f"  {name:16} p5 {pct(a, 5):.3f}  p25 {pct(a, 25):.3f}  p50 {pct(a, 50):.3f}"
              f"  p95 {pct(a, 95):.3f}")
    for floor in np.arange(0.40, 0.76, 0.05):
        keep = float((rel >= floor).mean())
        drop = float((wrong < floor).mean())
        print(f"  floor {floor:.2f}: keeps {keep:5.0%} of relevant, drops {drop:5.0%} of best-wrong")
    return 0


if __name__ == "__main__":
    sys.exit(main())
