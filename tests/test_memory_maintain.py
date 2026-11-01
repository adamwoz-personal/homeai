# Copyright (c) 2026 Adam Wosotowsky
# SPDX-License-Identifier: BUSL-1.1
"""Health sampling, the maintenance policy and the actions it runs
(memory_store/maintain.py). The policy is pure and table-tested; sampling,
purge, vacuum and ``maintain()`` run on a real database."""

from __future__ import annotations

import json
import time

import pytest

from homeai.memory_store import maintain as mt
from homeai.memory_store.maintain import Action, Thresholds, decide

T = Thresholds()


# -- pure ------------------------------------------------------------------------------


def test_percentile_nearest_rank():
    assert mt.percentile([], 0.95) is None
    assert mt.percentile([7.0], 0.95) == 7.0
    vals = list(range(100, 0, -1))
    assert mt.percentile(vals, 0.5) == 50 and mt.percentile(vals, 0.95) == 95


def test_pool_recalls_combines_hours():
    pooled = mt.pool_recalls([mt.recall_sample([10, 20], 1), mt.recall_sample([30], 0), {}])
    assert pooled == {"n": 3, "p50": 20.0, "p95": 30.0, "empty_rate": 0.333}
    assert mt.pool_recalls([])["p95"] is None


def person(**kw):
    return {"facts": 10, "stale": 0, "episodes": 10, "old_episodes": 0,
            "redundancy": None, "redundancy_n": 0, **kw}


def metrics(**kw):
    base = {"recall": {"n": 100, "p95": 12.0}, "people": {"adam": person()},
            "tables": {"facts": {"live": 100, "dead": 0}}, "purgeable": 0}
    return {**base, **kw}


@pytest.mark.parametrize("m,want", [
    (metrics(), []),
    (metrics(recall={"n": 20, "p95": 31.0}), [("decay", ""), ("merge", ""), ("reindex", "")]),
    (metrics(recall={"n": 19, "p95": 99.0}), []),                     # too few to judge
    (metrics(recall={"n": 500, "p95": 30.0}), []),
    (metrics(recall={"n": 0, "p95": None}), []),
    (metrics(people={"adam": person(facts=5001)}), [("decay", "adam"), ("merge", "adam")]),
    (metrics(people={"adam": person(facts=5000)}), []),
    (metrics(people={"adam": person(redundancy=0.11, redundancy_n=20)}), [("merge", "adam")]),
    (metrics(people={"adam": person(redundancy=0.11, redundancy_n=19)}), []),
    (metrics(people={"adam": person(redundancy=0.10, redundancy_n=200)}), []),
    (metrics(people={"adam": person(facts=2001, stale=601)}), [("decay", "adam")]),
    (metrics(people={"adam": person(facts=2001, stale=600)}), []),
    (metrics(people={"adam": person(facts=2000, stale=2000)}), []),   # too few to matter
    (metrics(people={"adam": person(old_episodes=20001)}), [("summarise", "adam")]),
    (metrics(people={"adam": person(old_episodes=20000)}), []),
    (metrics(tables={"facts": {"live": 79, "dead": 21}}), [("vacuum", "facts")]),
    (metrics(tables={"facts": {"live": 80, "dead": 20}}), []),
    (metrics(tables={"facts": {"live": 0, "dead": 0}}), []),
    (metrics(purgeable=1), [("purge", "")]),
    (metrics(people={"ryan": person(facts=5001, redundancy=0.5, redundancy_n=200)}),
     [("decay", "ryan"), ("merge", "ryan")]),                          # merge listed once
])
def test_policy_table(m, want):
    assert [(a.kind, a.target) for a in decide(m, T)] == want


def test_duplicate_actions_keep_every_reason():
    [_, merge] = decide(metrics(people={"adam": person(facts=5001, redundancy=0.5,
                                                       redundancy_n=200)}), T)
    assert "5001 active facts" in merge.reason and "near-duplicate" in merge.reason


def test_only_vacuum_and_purge_run():
    assert {k for k in ("vacuum", "purge", "decay", "merge", "summarise", "reindex")
            if Action(k, "", "").runs} == {"vacuum", "purge"}


def test_thresholds_are_configurable():
    assert decide(metrics(), Thresholds(recall_p95_ms=10.0))[0].kind == "decay"


# -- database --------------------------------------------------------------------------


def sql(store, q, args=()):
    with store.db.scoped(None, "admin") as conn:
        cur = conn.execute(q, args)
        return cur.fetchall() if cur.description else cur.rowcount


def test_sample_measures_each_person(mem_store):
    from homeai.memory_store.store import Episode
    from datetime import datetime, timedelta, timezone

    a1 = mem_store.add_fact("adam", "Adam keeps a corn snake called Noodle")
    mem_store.add_fact("adam", "Adam keeps a corn snake called Noodle")   # a duplicate
    mem_store.add_fact("adam", "Adam brews coffee every morning at six", pinned=True)
    mem_store.add_fact("ryan", "Ryan likes trains")
    sql(mem_store, "UPDATE memory.facts SET created_at = now() - interval '200 days',"
                   " importance = 0.3")                    # all old and unimportant ...
    now = datetime.now(timezone.utc)
    mem_store.add_episode(Episode("adam", "hello there", "hi", occurred_at=now))
    mem_store.add_episode(Episode("adam", "old chat", "yes", occurred_at=now - timedelta(days=95)))
    with mem_store.db.scoped(None, "admin") as conn:
        m = mt.sample(conn)
    adam, ryan = m["people"]["adam"], m["people"]["ryan"]
    assert adam["facts"] == 3 and adam["stale"] == 2      # ... but pinned is never stale
    assert adam["episodes"] == 2 and adam["old_episodes"] == 1 and adam["bytes"] > 0
    assert adam["redundancy_n"] == 3 and adam["redundancy"] == pytest.approx(2 / 3, abs=0.001)
    assert ryan["facts"] == 1 and ryan["redundancy"] is None   # no neighbour to compare
    assert "facts" in m["tables"] and m["db_bytes"] > 0 and m["purgeable"] == 0
    assert a1 is not None and json.loads(json.dumps(m)) == m   # storable as jsonb


def test_sample_pools_only_recent_daemon_timings(mem_store):
    with mem_store.db.scoped(None, "admin") as conn:
        mt.record(conn, mt.recall_sample([5.0, 6.0], 0))
        conn.execute("INSERT INTO memory.health_samples (at, metrics) VALUES"
                     " (now() - interval '8 days', %s)",
                     (json.dumps(mt.recall_sample([900.0], 0)),))
        mt.record(conn, {"source": "maintain", "recall_ms": [800.0]})
        assert mt.sample(conn)["recall"]["n"] == 2 and mt.sample(conn)["recall"]["p95"] == 6.0


def archived(store, summary, *, days, accessed_days=None, pinned=False, state="archived"):
    fid = store.add_fact("adam", summary, pinned=pinned)
    sql(store, "UPDATE memory.facts SET state = %s, updated_at = now() - make_interval(days => %s),"
               " last_accessed = CASE WHEN %s::int IS NULL THEN NULL"
               " ELSE now() - make_interval(days => %s) END WHERE id = %s",
        (state, days, accessed_days, accessed_days or 0, fid))
    return fid


def test_purge_deletes_only_long_unused_archived_facts(mem_store):
    gone = archived(mem_store, "Adam once owned a red bicycle", days=400)
    keep = [archived(mem_store, "Adam owned a blue kayak", days=364),
            archived(mem_store, "Adam owned a green tent", days=400, accessed_days=10),
            archived(mem_store, "Adam owned a yellow canoe", days=400, pinned=True),
            archived(mem_store, "Adam owns a black dog", days=400, state="active")]
    with mem_store.db.scoped(None, "admin") as conn:
        assert mt.sample(conn)["purgeable"] == 1
    assert mt.purge(mem_store) == 1
    ids = {r[0] for r in sql(mem_store, "SELECT id FROM memory.facts")}
    assert gone not in ids and set(keep) <= ids
    [(op, detail)] = sql(mem_store, "SELECT op, detail FROM memory.events WHERE object_id = %s"
                                    " AND op <> 'fact.add'", (gone,))
    assert op == "fact.purge" and "bicycle" not in json.dumps(detail)   # audit without content
    assert mt.purge(mem_store) == 0


def test_vacuum_runs_as_owner_outside_a_transaction(mem_store, clean_db):
    assert mt.vacuum(clean_db, ["facts", "episodes_2026"]) == ["facts", "episodes_2026"]
    for _ in range(50):
        rows = sql(mem_store, "SELECT relname FROM pg_stat_user_tables WHERE schemaname = 'memory'"
                              " AND relname IN ('facts', 'episodes_2026')"
                              " AND last_vacuum IS NOT NULL")
        if len(rows) == 2:
            break
        time.sleep(0.1)
    assert len(rows) == 2


def fake_metrics(**kw):
    def sample(conn, t=T):
        return metrics(**kw)
    return sample


def test_maintain_healthy_does_nothing_but_record(mem_store, monkeypatch):
    monkeypatch.setattr(mt, "sample", fake_metrics())
    calls = []
    r = mt.maintain(mem_store, "unused", vacuum_fn=lambda *a: calls.append(a),
                    purge_fn=lambda *a: calls.append(a))
    assert r["actions"] == [] and r["done"] == [] and calls == []
    assert sql(mem_store, "SELECT count(*) FROM memory.health_samples")[0][0] == 1


def test_maintain_dry_run_writes_nothing(mem_store, monkeypatch):
    monkeypatch.setattr(mt, "sample", fake_metrics(purgeable=3,
                                                   tables={"facts": {"live": 1, "dead": 9}}))
    sql(mem_store, "INSERT INTO memory.health_samples (at, metrics) VALUES"
                   " (now() - interval '200 days', '{}')")
    r = mt.maintain(mem_store, "unused", dry_run=True,
                    vacuum_fn=lambda *a: pytest.fail("vacuum ran"),
                    purge_fn=lambda *a: pytest.fail("purge ran"))
    assert [a["kind"] for a in r["actions"]] == ["vacuum", "purge"] and r["done"] == []
    # Still exactly the old row: nothing recorded, nothing housekept.
    assert sql(mem_store, "SELECT metrics FROM memory.health_samples") == [({},)]


def test_maintain_runs_due_work_and_housekeeps(mem_store, monkeypatch):
    monkeypatch.setattr(mt, "sample", fake_metrics(
        purgeable=1, people={"adam": person(facts=6000)},
        tables={"facts": {"live": 1, "dead": 9}, "jobs": {"live": 9, "dead": 1}}))
    sql(mem_store, "INSERT INTO memory.health_samples (at, metrics) VALUES"
                   " (now() - interval '91 days', '{}'), (now() - interval '89 days', '{}')")
    sql(mem_store, "INSERT INTO memory.jobs (kind, state, updated_at) VALUES"
                   " ('extract', 'done', now() - interval '31 days'),"
                   " ('extract', 'done', now() - interval '29 days'),"
                   " ('extract', 'failed', now() - interval '99 days')")
    vac = []
    r = mt.maintain(mem_store, "dsn", vacuum_fn=lambda dsn, t: vac.append((dsn, t)) or t,
                    purge_fn=lambda store, t: 1)
    assert vac == [("dsn", ["facts"])]
    assert r["done"] == [{"kind": "purge", "result": 1}, {"kind": "vacuum", "result": ["facts"]}]
    assert r["housekeeping"] == {"old_samples": 1, "old_jobs": 1}
    assert {(a["kind"], a["runs"]) for a in r["actions"]} >= {("decay", False), ("merge", False)}
    assert sql(mem_store, "SELECT count(*) FROM memory.health_samples")[0][0] == 2


def test_maintain_time_limit_defers_and_errors_do_not_stop_the_rest(mem_store, monkeypatch):
    monkeypatch.setattr(mt, "sample", fake_metrics(purgeable=1,
                                                   tables={"facts": {"live": 1, "dead": 9}}))

    def boom(store, t):
        raise RuntimeError("disk on fire")

    r = mt.maintain(mem_store, "dsn", purge_fn=boom, vacuum_fn=lambda dsn, t: t)
    assert r["errors"][0]["kind"] == "purge" and "disk on fire" in r["errors"][0]["error"]
    assert r["done"] == [{"kind": "vacuum", "result": ["facts"]}]
    ticks = iter([0.0, 700.0, 700.0])
    r = mt.maintain(mem_store, "dsn", time_limit_s=600, monotonic=lambda: next(ticks),
                    purge_fn=lambda *a: pytest.fail("ran past the limit"),
                    vacuum_fn=lambda *a: pytest.fail("ran past the limit"))
    assert r["deferred"] == ["purge", "vacuum"] and r["done"] == []
