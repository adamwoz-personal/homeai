#!/usr/bin/env python3
# Copyright (c) 2026 Adam Wosotowsky
# SPDX-License-Identifier: BUSL-1.1
"""Recall latency at scale, in a scratch database (never the real one).

Loads synthetic episodes and facts for several people, then times the recall
SQL (recall.fetch, person scope, RLS on) and the full Python fusion. Plan
target: p95 < 40 ms for the SQL at 100k episodes / 10k facts.

    .venv/bin/python tools/memory/recall_bench.py [--episodes 100000] [--facts 10000]
                                                    [--queries 200] [--keep]

Results go to ~/homeai-bench/memory-recall-<date>.json.
"""

from __future__ import annotations

import argparse
import json
import os
import random
import statistics
import subprocess
import sys
import time
from datetime import date, datetime, timedelta, timezone
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from homeai.memory_store import recall  # noqa: E402
from homeai.memory_store.db import Db, migrate  # noqa: E402

PEOPLE = ["adam", "ryan", "sue", "kid1"]
WORDS = ("snake dragon garage door weather pizza homework guitar telescope garden tomato "
         "camera python lizard birthday party movie map dungeon castle wizard rocket "
         "mars moon dinner recipe bike car battery solar fence paint").split()


def vocabulary(size: int) -> list[str]:
    """Real topic words first, then pseudo-words ('zq17'); stems stay distinct."""
    return (WORDS + [f"zq{i}" for i in range(max(0, size - len(WORDS)))])[:size]


class Text:
    """Zipf-ish word draws (s=1.1), like conversation; --vocab 30 is the old worst case."""

    def __init__(self, size: int):
        self.words = vocabulary(size)
        self.weights = [1 / (r + 1) ** 1.1 for r in range(len(self.words))]

    def __call__(self, rng: random.Random, n: int) -> str:
        return " ".join(rng.choices(self.words, self.weights, k=n))


text = Text(5000)


def unit(rng: np.random.Generator, n: int, dim: int = 384) -> np.ndarray:
    v = rng.standard_normal((n, dim)).astype(np.float32)
    return v / np.linalg.norm(v, axis=1, keepdims=True)


def pct(xs: list[float], q: float) -> float:
    return float(np.percentile(xs, q))


def load(dsn: str, episodes: int, facts: int, seed: int = 7) -> None:
    import psycopg
    from pgvector.psycopg import register_vector

    rng, nrng = random.Random(seed), np.random.default_rng(seed)
    now = datetime.now(timezone.utc)
    with psycopg.connect(dsn) as conn:
        register_vector(conn)
        conn.execute("INSERT INTO memory.people (username, display_name) SELECT u, u FROM"
                     " unnest(%s::text[]) u ON CONFLICT DO NOTHING", (PEOPLE,))
        conn.execute("SELECT set_config('app.scope', 'admin', false)")
        # COPY is refused under FORCE RLS. This is a scratch database: lift it
        # for the load and put it back before anything is timed.
        for t in ("episodes", "facts"):
            conn.execute(f"ALTER TABLE memory.{t} NO FORCE ROW LEVEL SECURITY")
        vecs = unit(nrng, episodes)
        with conn.cursor().copy("COPY memory.episodes (occurred_at, username, confidence, channel,"
                                " trust, kind, content, text_for_search, embedding, embed_model)"
                                " FROM STDIN") as cp:
            for i in range(episodes):
                u, a = text(rng, 8), text(rng, 25)
                cp.write_row((now - timedelta(minutes=rng.randrange(0, 60 * 24 * 900)),
                              rng.choice(PEOPLE + ["household"]), 1.0, "voice", "owner",
                              "exchange", json.dumps({"v": 1, "user": u, "assistant": a}),
                              f"{u}\n{a}", vecs[i], "bench"))
        fvecs = unit(nrng, facts)
        with conn.cursor().copy("COPY memory.facts (username, kind, summary, embedding, embed_model,"
                                " importance, trust, source) FROM STDIN") as cp:
            for i in range(facts):
                cp.write_row((rng.choice(PEOPLE), rng.choice(["fact", "preference", "profile"]),
                              text(rng, 10), fvecs[i], "bench", rng.random(), "owner", "import"))
        for t in ("episodes", "facts"):
            conn.execute(f"ALTER TABLE memory.{t} FORCE ROW LEVEL SECURITY")
        conn.commit()
        conn.autocommit = True
        conn.execute("ANALYZE memory.episodes")
        conn.execute("ANALYZE memory.facts")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--episodes", type=int, default=100_000)
    ap.add_argument("--facts", type=int, default=10_000)
    ap.add_argument("--queries", type=int, default=200)
    ap.add_argument("--keep", action="store_true", help="keep the scratch database")
    ap.add_argument("--explain", action="store_true", help="print EXPLAIN ANALYZE for one query")
    ap.add_argument("--vocab", type=int, default=5000,
                    help="distinct words in synthetic text (30 = every query matches ~40%% of rows)")
    args = ap.parse_args()
    global text
    text = Text(args.vocab)

    name = f"homeai_bench_{os.getpid()}"
    subprocess.run(["createdb", name], check=True)
    dsn = f"dbname={name}"
    try:
        migrate(dsn)
        t0 = time.perf_counter()
        load(dsn, args.episodes, args.facts)
        load_s = time.perf_counter() - t0
        print(f"loaded {args.episodes} episodes, {args.facts} facts in {load_s:.0f}s")

        db = Db(dsn).open()
        rng, nrng = random.Random(1), np.random.default_rng(1)
        now = datetime.now().astimezone()
        sql_ms, total_ms, cands = [], [], []
        qv = unit(nrng, args.queries)
        for i in range(args.queries + 10):
            who, q = rng.choice(PEOPLE), text(rng, 6)
            if i % 5 == 0:
                q += " last week"
            s = time.perf_counter()
            with db.scoped(who, timeout_ms=1000) as conn:
                rows = recall.fetch(conn, who, qv[i % args.queries], q, recall.time_window(q, now))
            m = time.perf_counter()
            out = recall.render(recall.keep(recall.fuse(rows, now), 0.0), who, now, 2400)
            e = time.perf_counter()
            if i >= 10:      # first few warm the pool and caches
                sql_ms.append((m - s) * 1000)
                total_ms.append((e - s) * 1000)
                cands.append(len(rows))
                assert {c.username for c in rows} <= {who, "household"}, "RLS leak"
        if args.explain:
            # Plain and time-windowed shapes, with the same text generator the timing loop uses.
            for q in (text(rng, 6), text(rng, 6) + " last week"):
                with db.scoped("adam", timeout_ms=5000) as conn:
                    win = recall.time_window(q, now) or (datetime(1970, 1, 1).astimezone(),
                                                         datetime(9999, 1, 1).astimezone())
                    plan = conn.execute("EXPLAIN (ANALYZE) " + recall._SQL,
                                        {"v": qv[0], "text": q, "owners": ["adam", "household"],
                                         "me": "adam", "t0": win[0], "t1": win[1],
                                         "n": recall.PER_LIST}).fetchall()
                print(f"--- {q!r}")
                print("\n".join(r[0] for r in plan if "Filter:" not in r[0]
                                and "Cond:" not in r[0] and "_20" not in r[0][:60]))
            s = time.perf_counter()
            for _ in range(20):
                with db.scoped("adam", timeout_ms=1000) as conn:
                    conn.execute("SELECT 1").fetchall()
            print(f"scoped() + SELECT 1: {(time.perf_counter() - s) / 20 * 1000:.1f} ms")
        db.close()
        res = {"date": date.today().isoformat(), "episodes": args.episodes, "facts": args.facts,
               "vocab": args.vocab,
               "queries": args.queries, "load_s": round(load_s, 1),
               "sql_ms": {"p50": pct(sql_ms, 50), "p95": pct(sql_ms, 95), "max": max(sql_ms)},
               "total_ms": {"p50": pct(total_ms, 50), "p95": pct(total_ms, 95)},
               "candidates_mean": statistics.mean(cands), "target_p95_ms": 40,
               "pass": pct(sql_ms, 95) < 40}
        print(json.dumps(res, indent=1))
        out_dir = Path.home() / "homeai-bench"
        out_dir.mkdir(exist_ok=True)
        (out_dir / f"memory-recall-{res['date']}.json").write_text(json.dumps(res, indent=1))
        return 0 if res["pass"] else 1
    finally:
        if args.keep:
            print(f"kept {name}")
        else:
            subprocess.run(["dropdb", "--if-exists", "--force", name])


if __name__ == "__main__":
    sys.exit(main())
