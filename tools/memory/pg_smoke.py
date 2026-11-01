#!/usr/bin/env python3
# Copyright (c) 2026 Adam Wosotowsky
# SPDX-License-Identifier: BUSL-1.1
"""Smoke-test the live memory database's security and indexing (plan §11, Phase 0/1).

Why: the guarantees memory rests on - Postgres reachable only over the local
socket, pgvector present, row security forced, one person never reading
another's rows, recall using the HNSW index - are configuration, and
configuration drifts (a package upgrade, a restored dump, a hand edit). Run
this after any of those.

    .venv/bin/python tools/memory/pg_smoke.py [--dsn dbname=homeai] [--json]

Everything it writes (a throwaway person and fact) is inside one transaction
that is always rolled back. Exit 0 if every check passed, 1 otherwise.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

MIGRATIONS = REPO / "homeai" / "memory_store" / "migrations"
ROLES = ("homeai_app", "homeai_mcp", "homeai_ro")
FORCED = ("episodes", "facts")


def checks(conn) -> list[tuple[str, bool, str]]:
    q = conn.execute
    out: list[tuple[str, bool, str]] = []

    def check(label: str, ok: bool, detail: object = "") -> None:
        out.append((label, bool(ok), str(detail)))

    listen = q("SHOW listen_addresses").fetchone()[0]
    check("socket only (listen_addresses '')", listen == "", repr(listen))
    check("this session came over the socket", q("SELECT inet_server_addr() IS NULL").fetchone()[0])
    ver = q("SELECT extversion FROM pg_extension WHERE extname = 'vector'").fetchone()
    check("pgvector installed", ver is not None, ver[0] if ver else "missing")
    want = len(list(MIGRATIONS.glob("[0-9]*.sql")))
    have = q("SELECT max(version) FROM memory.schema_version").fetchone()[0]
    check("schema up to date", have == want, f"db {have}, repo {want}")
    roles = {r[0]: r[1] for r in q("SELECT rolname, rolcanlogin FROM pg_roles"
                                   " WHERE rolname = ANY(%s)", (list(ROLES),))}
    check("app roles exist and cannot log in", set(roles) == set(ROLES)
          and not any(roles.values()), roles)
    forced = dict(q("SELECT relname, relforcerowsecurity FROM pg_class c JOIN pg_namespace n"
                    " ON n.oid = c.relnamespace WHERE nspname = 'memory' AND relname = ANY(%s)",
                    (list(FORCED),)).fetchall())
    check("row security forced on episodes and facts",
          forced == {t: True for t in FORCED}, forced)

    # Throwaway rows, rolled back below.
    who = f"zz_smoke_{os.getpid()}"
    q("SET LOCAL ROLE homeai_app")
    q("SELECT set_config('app.scope', 'admin', true)")
    q("INSERT INTO memory.people (username, display_name) VALUES (%s, 'Smoke')", (who,))
    vec = "[" + ",".join(["0.05"] * 384) + "]"
    q("INSERT INTO memory.facts (username, kind, summary, embedding, embed_model, trust, source)"
      " VALUES (%s, 'fact', 'smoke test secret', %s::vector, 'smoke', 'owner', 'explicit')",
      (who, vec))

    def visible(scope: str, user: str) -> int:
        q("SELECT set_config('app.scope', %s, true), set_config('app.username', %s, true)",
          (scope, user))
        return q("SELECT count(*) FROM memory.facts WHERE username = %s", (who,)).fetchone()[0]

    check("owner sees own fact", visible("person", who) == 1)
    check("another person cannot read it (RLS)", visible("person", "adam") == 0)
    check("no scope reads nothing (RLS)", visible("", "") == 0)
    q("SELECT set_config('app.scope', 'person', true), set_config('app.username', %s, true)",
      (who,))
    q("SET LOCAL enable_seqscan = off")
    plan = "\n".join(r[0] for r in q(
        "EXPLAIN SELECT id FROM memory.facts WHERE state = 'active'"
        " ORDER BY embedding <=> %s::vector LIMIT 5", (vec,)))
    check("vector recall can use the HNSW index", "embedding_idx" in plan
          and "Index Scan" in plan, plan.splitlines()[1].strip() if "\n" in plan else plan)
    hit = q("SELECT summary FROM memory.facts ORDER BY embedding <=> %s::vector LIMIT 1",
            (vec,)).fetchone()
    check("nearest-neighbour query answers", hit and hit[0] == "smoke test secret", hit)
    return out


def main() -> int:
    import psycopg

    from homeai.config import Config

    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--dsn", default=Config().memory.dsn)
    ap.add_argument("--json", action="store_true")
    a = ap.parse_args()
    with psycopg.connect(a.dsn) as conn:
        try:
            results = checks(conn)
        finally:
            conn.rollback()
    if a.json:
        print(json.dumps([{"check": c, "ok": ok, "detail": d} for c, ok, d in results]))
    else:
        for c, ok, d in results:
            print(f"{'PASS' if ok else 'FAIL'}  {c}{'  (' + d + ')' if d else ''}")
    return 0 if all(ok for _, ok, _ in results) else 1


if __name__ == "__main__":
    sys.exit(main())
