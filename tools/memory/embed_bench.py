#!/usr/bin/env python3
# Copyright (c) 2026 Adam Wosotowsky
# SPDX-License-Identifier: BUSL-1.1
# Licensed under the Business Source License 1.1; see LICENSE at the
# repository root. Converts to Apache-2.0 on 2030-09-15.
"""Which sentence-embedding model should long-term memory use?

Why this exists
---------------
The embedding model fixes the vector column's meaning; changing it later means
re-embedding everything (`homeai-memory reembed`). MEMORY_PLAN §5 names
candidates; this measures them on *our* kind of text, on this CPU:

* recall@1, recall@5 and MRR on tests/fixtures/memory_recall.json (50 household
  memories, 50 differently-worded spoken questions, near-miss distractors);
* with ``--distractors``, real transcript exchanges are added to the haystack,
  which is harder and closer to life;
* query latency: one sentence at a time, as the voice loop embeds (p50/p95 ms);
* document throughput (batch of 32) and model load time.

Usage
-----
    embed_bench.py --download                     # fetch all candidates once
    embed_bench.py                                # all models, fixtures only
    embed_bench.py --distractors --threads 4      # add ~150 real exchanges
    embed_bench.py --model bge-small-en-v1.5

Results: ~/homeai-bench/memory-embed-<date>.json

Last run: see the "Phase 0 results" section of plans/MEMORY_PLAN.md.
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from homeai.memory_store.embed import EMBED_SPECS, OnnxEmbedder, download  # noqa: E402

FIXTURES = ROOT / "tests" / "fixtures" / "memory_recall.json"
TRANSCRIPT = Path.home() / ".local/share/homeai/transcript.jsonl"
OUT_DIR = Path.home() / "homeai-bench"


def load_fixtures(distractors: bool) -> tuple[list[str], list[str], list[str], list[set[int]]]:
    data = json.loads(FIXTURES.read_text())
    ids = [m["id"] for m in data["memories"]]
    docs = [m["text"] for m in data["memories"]]
    if distractors and TRANSCRIPT.exists():
        for line in TRANSCRIPT.read_text().splitlines():
            rec = json.loads(line)
            if rec.get("heard") and rec.get("reply"):
                ids.append("t")
                docs.append(f"{rec['heard']} -- {rec['reply'][:400]}")
    index = {m: i for i, m in enumerate(ids) if m != "t"}
    queries = [q["q"] for q in data["queries"]]
    relevant = [{index[r] for r in q["relevant"]} for q in data["queries"]]
    return ids, docs, queries, relevant


def score(doc_vecs: np.ndarray, q_vecs: np.ndarray, relevant: list[set[int]]) -> dict:
    sims = q_vecs @ doc_vecs.T
    r1 = r5 = rr = 0.0
    for i, rel in enumerate(relevant):
        order = np.argsort(-sims[i])
        rank = next(k for k, d in enumerate(order) if d in rel)
        r1 += rank == 0
        r5 += rank < 5
        rr += 1.0 / (rank + 1)
    n = len(relevant)
    return {"recall@1": round(r1 / n, 3), "recall@5": round(r5 / n, 3), "mrr": round(rr / n, 3)}


def bench(name: str, threads: int, distractors: bool) -> dict:
    t0 = time.perf_counter()
    emb = OnnxEmbedder(name, threads=threads)
    load_s = time.perf_counter() - t0
    _, docs, queries, relevant = load_fixtures(distractors)
    emb.embed(["warm up"], kind="query")
    t0 = time.perf_counter()
    doc_vecs = np.vstack([emb.embed(docs[i:i + 32]) for i in range(0, len(docs), 32)])
    doc_ms = (time.perf_counter() - t0) * 1000 / len(docs)
    lat = []
    q_vecs = []
    for _ in range(3):  # three passes: single samples lie
        for q in queries:
            t = time.perf_counter()
            v = emb.embed([q], kind="query")
            lat.append((time.perf_counter() - t) * 1000)
            q_vecs.append(v[0])
    q_vecs = np.array(q_vecs[: len(queries)])
    lat.sort()
    return {
        "model": name, "threads": threads, "docs": len(docs), "load_s": round(load_s, 2),
        **score(doc_vecs, q_vecs, relevant),
        "query_ms_p50": round(statistics.median(lat), 1),
        "query_ms_p95": round(lat[int(len(lat) * 0.95) - 1], 1),
        "doc_ms_each_batched": round(doc_ms, 2),
    }


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", choices=sorted(EMBED_SPECS), action="append")
    ap.add_argument("--download", action="store_true")
    ap.add_argument("--distractors", action="store_true")
    ap.add_argument("--threads", type=int, default=4)
    args = ap.parse_args(argv)
    names = args.model or sorted(EMBED_SPECS)
    if args.download:
        for n in names:
            print(f"downloading {n} ...", flush=True)
            print("  ->", download(EMBED_SPECS[n]))
        return 0
    results = []
    for n in names:
        try:
            r = bench(n, args.threads, args.distractors)
        except Exception as exc:  # noqa: BLE001 - one broken model must not hide the rest
            r = {"model": n, "error": str(exc)}
        print(json.dumps(r), flush=True)
        results.append(r)
    OUT_DIR.mkdir(exist_ok=True)
    out = OUT_DIR / f"memory-embed-{time.strftime('%Y-%m-%d')}.json"
    existing = json.loads(out.read_text()) if out.exists() else []
    out.write_text(json.dumps(existing + [{"distractors": args.distractors, "results": results}],
                              indent=1))
    print(f"wrote {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
