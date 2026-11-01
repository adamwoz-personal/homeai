# Copyright (c) 2026 Adam Wosotowsky
# SPDX-License-Identifier: BUSL-1.1
"""Throwaway PostgreSQL databases for the long-term memory tests.

One database per test session (``homeai_test_<pid>``), migrated once,
truncated between tests, dropped at the end. Tests that use it are skipped
when Postgres or the ``memory`` extra is unavailable, so CI without a
database still runs everything else.
"""

from __future__ import annotations

import os
import subprocess

import pytest

TABLES = ("events", "jobs", "health_samples", "fact_links", "fact_sources", "facts",
          "episodes", "topics", "identities")


def _have_postgres() -> str | None:
    try:
        import pgvector  # noqa: F401
        import psycopg
        import psycopg_pool  # noqa: F401
    except ImportError as exc:
        return f"memory extra not installed ({exc.name})"
    try:
        psycopg.connect("dbname=postgres", connect_timeout=2).close()
    except Exception as exc:  # noqa: BLE001
        return f"postgres unavailable: {exc}"
    return None


@pytest.fixture(scope="session")
def pg_dsn():
    why = _have_postgres()
    if why:
        pytest.skip(why)
    name = f"homeai_test_{os.getpid()}"
    made = subprocess.run(["createdb", name], capture_output=True, text=True)
    if made.returncode:
        pytest.skip(f"cannot create a test database: {made.stderr.strip()}")
    dsn = f"dbname={name}"
    try:
        from homeai.memory_store.db import migrate

        migrate(dsn)
        yield dsn
    finally:
        subprocess.run(["dropdb", "--if-exists", "--force", name], capture_output=True)


@pytest.fixture
def clean_db(pg_dsn):
    """An empty schema (seed people only) for each test."""
    import psycopg

    with psycopg.connect(pg_dsn, autocommit=True) as conn:
        conn.execute("TRUNCATE " + ", ".join(f"memory.{t}" for t in TABLES) + " CASCADE")
        conn.execute("DELETE FROM memory.people WHERE username NOT IN ('household', 'guest')")
    return pg_dsn


@pytest.fixture
def mem_store(clean_db):
    from homeai.memory_store.db import Db
    from homeai.memory_store.embed import HashEmbedder
    from homeai.memory_store.store import PostgresStore

    db = Db(clean_db, role="homeai_app").open()
    store = PostgresStore(db, HashEmbedder())
    store.ensure_person("adam", "Adam", is_admin=True, voice_handle="Adam")
    store.ensure_person("ryan", "Ryan", voice_handle="Ryan")
    yield store
    db.close()
