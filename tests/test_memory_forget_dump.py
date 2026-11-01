# Copyright (c) 2026 Adam Wosotowsky
# SPDX-License-Identifier: BUSL-1.1
"""Forgetting really forgets: after "forget that", a canary word appears
nowhere in a full ``pg_dump`` of the database (plan §11 Phase 2 exit check).

Covers what extraction and reflection spread the words into: facts, the
profile built from them, topic and thread names. pg_dump only sees live rows;
dead tuples linger on disk until VACUUM (see RUNBOOK).
"""

from __future__ import annotations

import os
import shutil
import subprocess

import pytest

from tests.test_memory_worker import Router, add_eps, note, run_extract, run_reflect, worker

CANARY = "zorbleflax"

pytestmark = pytest.mark.skipif(shutil.which("pg_dump") is None, reason="no pg_dump")


def dump(dsn: str) -> str:
    """Everything in the database, as the owner with admin scope (RLS applies
    even to the owner; app.scope=admin is the policy's see-everything case)."""
    env = {**os.environ, "PGOPTIONS": "-c app.scope=admin"}
    out = subprocess.run(["pg_dump", "--data-only", "--enable-row-security", dsn],
                         capture_output=True, text=True, check=True, env=env)
    return out.stdout.lower()


def spread_canary(store):
    """One exchange mentioning the canary, a fact, topic and thread drawn from
    it, and a profile line built from that fact."""
    add_eps(store, (f"my {CANARY} collection keeps growing", "Nice."),
            ("I have a dog", "Lovely."))
    w, _ = worker(store, Router(
        notes=[[note(f"Adam has a growing {CANARY} collection", topics=(CANARY,),
                     thread=f"{CANARY} hunt", sources=[1]),
                note("Adam has a dog", sources=[2])]],
        lines={"lines": [{"summary": f"Adam collects {CANARY}", "sources": [1, 2]}]}))
    run_extract(store, w)
    store.add_fact("adam", "Adam likes tea", source="extractor")
    run_reflect(store, w)
    return w


def test_canary_is_spread_before_forgetting(mem_store, pg_dsn):
    spread_canary(mem_store)
    assert dump(pg_dsn).count(CANARY) >= 4


def test_forgetting_the_exchange_leaves_no_trace(mem_store, pg_dsn):
    spread_canary(mem_store)
    (ep,) = [e for e in mem_store.list_episodes("adam", owner="adam") if CANARY in e["user"]]
    mem_store.forget_items("adam", episodes=[ep["id"]], actor="test")
    assert CANARY not in dump(pg_dsn)
    assert "adam has a dog" in dump(pg_dsn)         # only what was forgotten went


def test_forget_everything_leaves_no_trace(mem_store, pg_dsn):
    spread_canary(mem_store)
    mem_store.forget_person("adam", actor="test")
    assert CANARY not in dump(pg_dsn)


def test_forgetting_one_fact_by_id_leaves_no_trace(mem_store, pg_dsn):
    spread_canary(mem_store)
    (f,) = [f for f in mem_store.facts("adam") if CANARY in f["summary"] and
            f["source"] == "extractor"]
    mem_store.forget(f["id"], actor="adam")
    (ep,) = [e for e in mem_store.list_episodes("adam", owner="adam") if CANARY in e["user"]]
    mem_store.forget(ep["id"], actor="adam")
    assert CANARY not in dump(pg_dsn)


def test_forgetting_a_summary_with_its_sources_in_one_go(mem_store):
    spread_canary(mem_store)
    ids = [f["id"] for f in mem_store.facts("adam", kinds=None)]
    assert mem_store.forget_items("adam", facts=ids, actor="test") == len(ids)


def test_topics_still_used_by_someone_else_are_kept(mem_store):
    spread_canary(mem_store)
    mem_store.add_fact("ryan", "Ryan also has some", source="extractor")
    with mem_store.db.scoped(None, "admin") as conn:
        conn.execute("UPDATE memory.facts SET topics = %s WHERE username = 'ryan'", ([CANARY],))
    mem_store.forget_person("adam", actor="test")
    assert CANARY in mem_store.known_topics()
