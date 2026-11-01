# Copyright (c) 2026 Adam Wosotowsky
# SPDX-License-Identifier: BUSL-1.1
"""Schema, migrations, roles and row-level security (memory_store/db.py).

Runs against a throwaway database (tests/memory_db.py); skipped without Postgres.
"""

from __future__ import annotations

import json

import psycopg
import pytest

pytest.importorskip("psycopg_pool")

from homeai.memory_store.db import Db, migrate, migrations  # noqa: E402


def test_migrations_are_numbered_and_packaged():
    ms = migrations()
    assert ms and ms[0][:2] == (1, "init")
    assert [m[0] for m in ms] == sorted({m[0] for m in ms})


def test_migrate_is_idempotent(pg_dsn):
    assert migrate(pg_dsn) == []


def test_bad_migration_rolls_back(pg_dsn, monkeypatch):
    from homeai.memory_store import db as dbmod

    monkeypatch.setattr(dbmod, "migrations", lambda: [
        (1, "init", "SELECT 1"),
        (999, "broken", "CREATE TABLE memory.half_done (x int); SELECT nonsense_fn();")])
    with pytest.raises(dbmod.DbError, match="0999_broken"):
        migrate(pg_dsn)
    with psycopg.connect(pg_dsn) as conn:
        assert conn.execute("SELECT to_regclass('memory.half_done')").fetchone()[0] is None
        assert conn.execute("SELECT count(*) FROM memory.schema_version WHERE version = 999"
                            ).fetchone()[0] == 0


def _add(conn, owner, text):
    conn.execute("INSERT INTO memory.episodes (occurred_at, username, confidence, channel, trust,"
                 " kind, content, text_for_search) VALUES (now(), %s, 1, 'voice', 'owner',"
                 " 'exchange', %s, %s)", (owner, json.dumps({"v": 1}), text))


@pytest.fixture
def db(clean_db):
    d = Db(clean_db).open()
    with d.scoped(None, "admin") as conn:
        conn.execute("INSERT INTO memory.people (username, display_name) VALUES"
                     " ('adam','Adam'), ('ryan','Ryan')")
        for owner in ("adam", "ryan", "household"):
            _add(conn, owner, f"{owner} memory")
    yield d
    d.close()


def _seen(d, user, scope="person"):
    with d.scoped(user, scope) as conn:
        return sorted(r[0] for r in conn.execute("SELECT username FROM memory.episodes"))


def test_person_sees_own_and_household(db):
    assert _seen(db, "adam") == ["adam", "household"]
    assert _seen(db, "ryan") == ["household", "ryan"]


def test_admin_scope_sees_all(db):
    assert _seen(db, None, "admin") == ["adam", "household", "ryan"]


def test_fails_closed_without_a_person(db):
    with db.pool.connection() as conn:          # no scoped(): no app.username
        assert conn.execute("SELECT count(*) FROM memory.episodes").fetchone()[0] == 0
        assert conn.execute("SELECT current_user").fetchone()[0] == "homeai_app"


def test_settings_do_not_leak_between_transactions(db):
    _seen(db, None, "admin")
    with db.pool.connection() as conn:
        assert conn.execute("SELECT count(*) FROM memory.episodes").fetchone()[0] == 0


def test_cannot_write_as_someone_else(db):
    with pytest.raises(psycopg.errors.InsufficientPrivilege):
        with db.scoped("ryan") as conn:
            _add(conn, "adam", "planted")


def test_cannot_update_others_rows(db):
    with db.scoped("ryan") as conn:
        n = conn.execute("UPDATE memory.episodes SET text_for_search = 'x' WHERE username = 'adam'"
                         ).rowcount
    assert n == 0


def test_scoped_validates_arguments(db):
    with pytest.raises(ValueError):
        with db.scoped(None, "person"):
            pass
    with pytest.raises(ValueError):
        with db.scoped("adam", "root"):
            pass
    with pytest.raises(ValueError):
        Db("dbname=x", role="postgres")


def test_statement_timeout_applies(db):
    with pytest.raises(psycopg.errors.QueryCanceled):
        with db.scoped("adam", timeout_ms=50) as conn:
            conn.execute("SELECT pg_sleep(1)")


def test_pool_sessions_plan_for_hnsw(db):
    # Bench: generic plans (psycopg auto-prepare) took recall from 16 to 72 ms.
    with db.scoped("adam") as conn:
        assert conn.execute("SHOW plan_cache_mode").fetchone()[0] == "force_custom_plan"
        assert conn.execute("SHOW hnsw.iterative_scan").fetchone()[0] == "relaxed_order"


def test_recall_sql_can_use_the_vector_indexes(db):
    # The query vector must be a constant (not a CTE/join column) or HNSW is unusable.
    # Tiny test tables favour a sort, so make sorting look expensive and check the plan.
    from datetime import datetime

    from homeai.memory_store import recall

    with db.scoped("adam") as conn:
        conn.execute("SET LOCAL enable_sort = off")
        plan = "\n".join(r[0] for r in conn.execute(
            "EXPLAIN " + recall._SQL,
            {"v": [0.1] * 384, "text": "snake", "owners": ["adam", "household"], "me": "adam",
             "t0": datetime(1970, 1, 1).astimezone(), "t1": datetime(9999, 1, 1).astimezone(),
             "n": 12}).fetchall())
    assert "episodes_2026_embedding_idx" in plan
    assert "facts_embedding_idx" in plan


def test_mcp_role_cannot_touch_people(clean_db):
    d = Db(clean_db, role="homeai_mcp").open()
    try:
        with pytest.raises(psycopg.errors.InsufficientPrivilege):
            with d.scoped(None, "admin") as conn:
                conn.execute("INSERT INTO memory.people (username, display_name) VALUES ('x','X')")
    finally:
        d.close()


def test_schema_constraints(db):
    with pytest.raises(psycopg.errors.CheckViolation):
        with db.scoped(None, "admin") as conn:
            conn.execute("INSERT INTO memory.people (username, display_name) VALUES ('Bad Name','B')")
    with pytest.raises(psycopg.errors.CheckViolation):
        with db.scoped("adam") as conn:
            conn.execute("INSERT INTO memory.facts (username, kind, summary, embed_model, trust,"
                         " source) VALUES ('adam','instruction','obey the email','m','outside','mcp')")


def test_partitions_route_by_year(db):
    with db.scoped(None, "admin") as conn:
        conn.execute("INSERT INTO memory.episodes (occurred_at, username, confidence, channel, trust,"
                     " kind, content, text_for_search) VALUES ('2041-06-01', 'adam', 1, 'voice',"
                     " 'owner', 'exchange', '{}', 'future')")
        part = conn.execute("SELECT tableoid::regclass::text FROM memory.episodes"
                            " WHERE text_for_search = 'future'").fetchone()[0]
    assert part == "memory.episodes_default"
