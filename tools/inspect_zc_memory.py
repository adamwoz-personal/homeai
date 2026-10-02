#!/usr/bin/env python3
# Copyright (c) 2026 Adam Wosotowsky
# SPDX-License-Identifier: BUSL-1.1
# Licensed under the Business Source License 1.1; see LICENSE at the
# repository root. Converts to Apache-2.0 on 2030-09-15.
"""Read-only view of ZeroClaw's persistent memory store (brain.db).

Why this exists
---------------
ZeroClaw auto-saves every request into ``~/.zeroclaw/data/memory/brain.db``.
For the voice agent that means everything the microphone heard, plus the
context block homeai sends (earlier answers). On 2026-10-02 the voice model
was found recalling those rows and repeating old answers. The voice profile
no longer allows ``memory_recall``, but the rows are still written, so this
shows what is in there and how fast it grows.

The database is opened read-only; this never modifies it.

Usage
-----
    inspect_zc_memory.py                      # row counts per agent/category
    inspect_zc_memory.py --agent local --recent 5
    inspect_zc_memory.py --agent local --search "free will"
"""

from __future__ import annotations

import argparse
import sqlite3
import sys
from pathlib import Path

DEFAULT_DB = Path.home() / ".zeroclaw" / "data" / "memory" / "brain.db"


def connect(path: Path) -> sqlite3.Connection:
    return sqlite3.connect(f"file:{path}?mode=ro", uri=True)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--db", type=Path, default=DEFAULT_DB)
    ap.add_argument("--agent", help="agent name, e.g. local, builder")
    ap.add_argument("--recent", type=int, default=0, help="show the N newest rows")
    ap.add_argument("--search", help="substring to find in content")
    ap.add_argument("--width", type=int, default=300)
    args = ap.parse_args(argv)

    if not args.db.exists():
        print(f"no database at {args.db}", file=sys.stderr)
        return 1
    db = connect(args.db)
    agents = dict(db.execute("SELECT id, alias FROM agents"))

    print("agent       category        rows  oldest                     newest")
    for agent_id, category, n, lo, hi in db.execute(
        "SELECT agent_id, category, count(*), min(created_at), max(created_at) "
        "FROM memories GROUP BY 1, 2 ORDER BY 3 DESC"
    ):
        print(f"{agents.get(agent_id, agent_id)[:10]:11} {category[:14]:14} "
              f"{n:5}  {lo[:25]:26} {hi[:25]}")

    if not (args.recent or args.search):
        return 0
    where, params = [], []
    if args.agent:
        ids = [i for i, name in agents.items() if name == args.agent]
        if not ids:
            print(f"unknown agent {args.agent!r}; known: {sorted(agents.values())}",
                  file=sys.stderr)
            return 1
        where.append("agent_id = ?")
        params.append(ids[0])
    if args.search:
        where.append("content LIKE ?")
        params.append(f"%{args.search}%")
    sql = "SELECT created_at, session_id, content FROM memories"
    if where:
        sql += " WHERE " + " AND ".join(where)
    sql += " ORDER BY created_at DESC LIMIT ?"
    params.append(args.recent or 20)
    print()
    for created, session, content in db.execute(sql, params):
        print(f"--- {created[:19]}  {session}")
        print(content[: args.width].replace("\n", " / "))
    return 0


if __name__ == "__main__":
    sys.exit(main())
