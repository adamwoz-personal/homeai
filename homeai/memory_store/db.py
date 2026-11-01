# Copyright (c) 2026 Adam Wosotowsky
# SPDX-License-Identifier: BUSL-1.1
# Licensed under the Business Source License 1.1; see LICENSE at the
# repository root. Converts to Apache-2.0 on 2030-09-15.
"""Database access: pool, roles, per-request person scope, migrations.

Every query runs inside ``Db.scoped(username, scope)``, which sets
``app.username`` / ``app.scope`` for that transaction only. Row-level security
(migrations/0001_init.sql) reads them, so a code path that forgets to say who
is asking sees nothing. Scopes:

* ``person``: that person's rows plus 'household' (voice, MCP for one speaker)
* ``admin``: everything (Adam via the CLI, background maintenance)

Roles are a backstop against bugs, not a security boundary: every process is
the same Unix user and could ``RESET ROLE``.
"""

from __future__ import annotations

import logging
import re
from contextlib import contextmanager
from importlib import resources
from typing import Iterator

log = logging.getLogger(__name__)

ROLES = ("homeai_app", "homeai_mcp", "homeai_ro")
SCOPES = ("person", "admin")


class DbError(RuntimeError):
    pass


def migrations() -> list[tuple[int, str, str]]:
    """(version, name, sql) in order, from the package's migrations/ folder."""
    out = []
    for f in resources.files("homeai.memory_store").joinpath("migrations").iterdir():
        m = re.match(r"^(\d{4})_(\w+)\.sql$", f.name)
        if m:
            out.append((int(m.group(1)), m.group(2), f.read_text(encoding="utf-8")))
    return sorted(out)


def migrate(dsn: str) -> list[int]:
    """Apply pending migrations as the login user (the owner). Idempotent.

    Each migration runs in its own transaction; a failure rolls it back
    entirely and stops, so the schema is never half-applied.
    """
    import psycopg

    applied = []
    with psycopg.connect(dsn, autocommit=False) as conn:
        conn.execute("CREATE EXTENSION IF NOT EXISTS vector")
        conn.execute("CREATE SCHEMA IF NOT EXISTS memory")
        conn.execute("""CREATE TABLE IF NOT EXISTS memory.schema_version (
                          version int PRIMARY KEY, name text NOT NULL,
                          applied_at timestamptz NOT NULL DEFAULT now())""")
        conn.commit()
        done = {r[0] for r in conn.execute("SELECT version FROM memory.schema_version")}
        conn.commit()
        for version, name, sql in migrations():
            if version in done:
                continue
            try:
                conn.execute(sql)
                conn.execute("INSERT INTO memory.schema_version (version, name) VALUES (%s, %s)",
                             (version, name))
                conn.commit()
            except Exception as exc:
                conn.rollback()
                raise DbError(f"migration {version:04d}_{name} failed: {exc}") from exc
            log.info("applied migration %04d_%s", version, name)
            applied.append(version)
    return applied


class Db:
    """A small pool; connections take ``role`` as soon as they open."""

    def __init__(self, dsn: str, role: str = "homeai_app", min_size: int = 1,
                 max_size: int = 4, connect_timeout_s: float = 2.0) -> None:
        if role not in ROLES:
            raise ValueError(f"unknown role {role!r}")
        from psycopg_pool import ConnectionPool

        self.role = role
        self.pool = ConnectionPool(
            dsn, min_size=min_size, max_size=max_size, open=False,
            kwargs={"connect_timeout": max(1, int(connect_timeout_s))},
            configure=self._configure, timeout=connect_timeout_s,
        )

    def _configure(self, conn) -> None:
        from pgvector.psycopg import register_vector

        register_vector(conn)
        # Owner/time filters are applied after the HNSW walk; iterative scan keeps
        # walking until LIMIT rows survive instead of returning too few.
        conn.execute("SET hnsw.iterative_scan = relaxed_order")
        # psycopg prepares a statement after 5 runs; Postgres then tends to a generic
        # plan that can't use HNSW/partition pruning (bench: 16 ms -> 72 ms).
        # Planning the recall query costs < 1 ms, so always plan with real values.
        conn.execute("SET plan_cache_mode = force_custom_plan")
        conn.execute(f"SET ROLE {self.role}")  # role is validated above
        conn.commit()

    def open(self, wait: bool = True) -> "Db":
        self.pool.open(wait=wait)
        return self

    def close(self) -> None:
        self.pool.close()

    @contextmanager
    def scoped(self, username: str | None, scope: str = "person",
               timeout_ms: int | None = None,
               connect_timeout_s: float | None = None) -> Iterator:
        """One transaction, with RLS settings and an optional statement timeout."""
        if scope not in SCOPES:
            raise ValueError(f"unknown scope {scope!r}")
        if scope == "person" and not username:
            raise ValueError("a person scope needs a username")
        with self.pool.connection(timeout=connect_timeout_s) as conn:
            with conn.transaction():
                conn.execute("SELECT set_config('app.username', %s, true), "
                             "set_config('app.scope', %s, true)",
                             (username or "", scope))
                if timeout_ms:
                    conn.execute("SELECT set_config('statement_timeout', %s, true)",
                                 (f"{int(timeout_ms)}ms",))
                yield conn
