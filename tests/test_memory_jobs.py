# Copyright (c) 2026 Adam Wosotowsky
# SPDX-License-Identifier: BUSL-1.1
"""The persistent job queue (memory_store/jobs.py), against real Postgres."""

from __future__ import annotations

import pytest

from homeai.memory_store import jobs

pytestmark = pytest.mark.usefixtures("mem_store")


def admin(store):
    return store.db.scoped(None, "admin")


def rows(store):
    with admin(store) as conn:
        return conn.execute("SELECT kind, dedupe_key, state, attempts,"
                            " extract(epoch FROM not_before - now())::int, last_error"
                            " FROM memory.jobs ORDER BY id").fetchall()


def test_enqueue_dedupes_by_key(mem_store):
    with admin(mem_store) as conn:
        jobs.enqueue(conn, "extract", {"username": "adam"}, key="extract:adam", delay_s=600)
        jobs.enqueue(conn, "extract", {"username": "adam"}, key="extract:adam", delay_s=600)
        jobs.enqueue(conn, "extract", {"username": "ryan"}, key="extract:ryan")
    assert [r[1] for r in rows(mem_store)] == ["extract:adam", "extract:ryan"]


def test_debounce_pushes_back_but_non_debounce_only_brings_forward(mem_store):
    with admin(mem_store) as conn:
        jobs.enqueue(conn, "extract", {"username": "adam"}, key="k", delay_s=100)
        jobs.enqueue(conn, "extract", {"username": "adam"}, key="k", delay_s=600)
    assert rows(mem_store)[0][4] > 500
    with admin(mem_store) as conn:
        jobs.enqueue(conn, "extract", {"username": "adam"}, key="k", delay_s=900, debounce=False)
    assert 500 < rows(mem_store)[0][4] <= 600
    with admin(mem_store) as conn:
        jobs.enqueue(conn, "extract", {"username": "adam"}, key="k", delay_s=0, debounce=False)
    assert rows(mem_store)[0][4] <= 0


def test_run_now(mem_store):
    with admin(mem_store) as conn:
        jobs.enqueue(conn, "extract", {"username": "adam"}, key="k", delay_s=600)
        assert jobs.claim(conn, ("extract",)) is None          # not due yet
        assert jobs.run_now(conn, "k") == 1
        assert jobs.claim(conn, ("extract",)).payload == {"username": "adam"}


def test_running_twin_swallows_enqueue_then_allows_a_new_one_after_finish(mem_store):
    with admin(mem_store) as conn:
        jobs.enqueue(conn, "extract", {"username": "adam"}, key="k")
        job = jobs.claim(conn, ("extract",))
        jobs.enqueue(conn, "extract", {"username": "adam"}, key="k")
    assert len(rows(mem_store)) == 1
    with admin(mem_store) as conn:
        jobs.finish(conn, job.id)
        jobs.enqueue(conn, "extract", {"username": "adam"}, key="k")
    assert [r[2] for r in rows(mem_store)] == ["done", "queued"]


def test_claim_filters_by_kind_and_order(mem_store):
    with admin(mem_store) as conn:
        jobs.enqueue(conn, "reflect", {"username": "adam"}, key="r")
        jobs.enqueue(conn, "extract", {"username": "adam"}, key="e")
        assert jobs.claim(conn, ("extract",)).kind == "extract"
        assert jobs.claim(conn, ("extract",)) is None
        j = jobs.claim(conn, ("extract", "reflect"))
        assert j.kind == "reflect" and j.attempts == 1


def test_two_workers_never_claim_the_same_job(mem_store):
    with admin(mem_store) as conn:
        jobs.enqueue(conn, "extract", {"username": "adam"}, key="a")
        jobs.enqueue(conn, "extract", {"username": "ryan"}, key="b")
    with admin(mem_store) as c1, admin(mem_store) as c2:
        j1 = jobs.claim(c1, ("extract",))          # c1's lock is still held
        j2 = jobs.claim(c2, ("extract",))
        assert j1 and j2 and j1.id != j2.id
    with admin(mem_store) as c1, admin(mem_store) as c2:
        jobs.enqueue(c1, "extract", {"username": "adam"}, key="c")
    with admin(mem_store) as c1, admin(mem_store) as c2:
        assert jobs.claim(c1, ("extract",)) is not None
        assert jobs.claim(c2, ("extract",)) is None


def test_fail_backs_off_then_gives_up(mem_store):
    with admin(mem_store) as conn:
        jobs.enqueue(conn, "extract", {"username": "adam"}, key="k")
    delays = []
    for attempt in range(1, jobs.MAX_ATTEMPTS + 1):
        with admin(mem_store) as conn:
            conn.execute("UPDATE memory.jobs SET not_before = now()")
            job = jobs.claim(conn, ("extract",))
            assert job.attempts == attempt
            state = jobs.fail(conn, job, "boom " * 200)
        r = rows(mem_store)[0]
        delays.append(r[4])
        assert len(r[5]) <= 500
    assert state == "failed" and rows(mem_store)[0][2] == "failed"
    assert 50 < delays[0] <= 60 and 590 < delays[1] <= 600
    with admin(mem_store) as conn:
        assert jobs.claim(conn, ("extract",)) is None


def test_recover_requeues_stale_running_and_tidies_old_done(mem_store):
    with admin(mem_store) as conn:
        for k in ("stale", "fresh", "old", "recent"):
            jobs.enqueue(conn, "extract", {"username": "adam"}, key=k)
        conn.execute("UPDATE memory.jobs SET state = 'running',"
                     " updated_at = now() - interval '1 hour' WHERE dedupe_key = 'stale'")
        conn.execute("UPDATE memory.jobs SET state = 'running' WHERE dedupe_key = 'fresh'")
        conn.execute("UPDATE memory.jobs SET state = 'done',"
                     " updated_at = now() - interval '8 days' WHERE dedupe_key = 'old'")
        conn.execute("UPDATE memory.jobs SET state = 'done' WHERE dedupe_key = 'recent'")
        assert jobs.recover(conn) == 1
        states = dict(conn.execute("SELECT dedupe_key, state FROM memory.jobs").fetchall())
    assert states == {"stale": "queued", "fresh": "running", "recent": "done"}


def test_pending_lists_unfinished(mem_store):
    with admin(mem_store) as conn:
        jobs.enqueue(conn, "extract", {"username": "adam"}, key="a")
        jobs.enqueue(conn, "reflect", {"username": "adam"}, key="b")
        jobs.finish(conn, jobs.claim(conn, ("extract",)).id)
        assert [j["key"] for j in jobs.pending(conn)] == ["b"]
