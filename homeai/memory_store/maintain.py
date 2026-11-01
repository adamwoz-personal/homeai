# Copyright (c) 2026 Adam Wosotowsky
# SPDX-License-Identifier: BUSL-1.1
# Licensed under the Business Source License 1.1; see LICENSE at the
# repository root. Converts to Apache-2.0 on 2030-09-15.
"""Health sampling and maintenance (plan §8, "Phase 4 lite").

* ``sample(conn)`` measures the database: sizes, dead tuples, per-person
  counts, redundancy, staleness, the job queue, and the recall timings the
  daemon records (``recall_sample``).
* ``decide(metrics)`` is a pure function from metrics to ``Action``\\ s.
  Nothing to do is the normal answer: "healthy, no work".
* ``maintain()`` records the sample, does the housekeeping that is always
  due, runs the cheap actions (VACUUM, purge), and only *reports* the rest
  (decay, merge, summarise, reindex): those are not built until an advisory
  says they are needed.

VACUUM can't run in a transaction or as ``homeai_app``; it runs on its own
autocommit connection as the login user (the database owner), like backups.
"""

from __future__ import annotations

import json
import logging
import time
from dataclasses import asdict, dataclass
from typing import Any, Callable, Sequence

log = logging.getLogger(__name__)

RUNNABLE = frozenset({"vacuum", "purge"})
DAEMON = "daemon"
MAINTAIN = "maintain"


@dataclass(frozen=True)
class Thresholds:
    """Plan §8.2 defaults. Counts are per person unless named otherwise."""

    recall_p95_ms: float = 30.0
    min_recalls: int = 20              # fewer timings than this say nothing about p95
    recall_window_days: int = 7
    facts_per_person: int = 5000
    redundancy: float = 0.10
    redundancy_sim: float = 0.92
    redundancy_sample: int = 200
    redundancy_min_facts: int = 20     # below this a share is noise
    staleness: float = 0.30
    staleness_min_facts: int = 2000
    stale_days: int = 180
    stale_importance: float = 0.5
    old_episodes: int = 20000
    old_episode_days: int = 90
    dead_ratio: float = 0.20
    purge_days: int = 365
    keep_samples_days: int = 90
    keep_jobs_days: int = 30


@dataclass(frozen=True)
class Action:
    kind: str       # vacuum | purge | decay | merge | summarise | reindex
    target: str     # a table, a username, or "" for everything
    reason: str

    @property
    def runs(self) -> bool:
        return self.kind in RUNNABLE


# -- timings -----------------------------------------------------------------------------


def percentile(values: Sequence[float], q: float) -> float | None:
    """Nearest-rank percentile; None for no values."""
    if not values:
        return None
    s = sorted(values)
    k = max(0, min(len(s) - 1, -(-len(s) * q // 1) - 1))
    return float(s[int(k)])


def recall_sample(ms: Sequence[float], empty: int) -> dict[str, Any]:
    """What the daemon stores each hour: raw timings, so p95 can be pooled."""
    return {"source": DAEMON, "recall_ms": [round(float(m), 1) for m in ms], "empty": int(empty)}


def pool_recalls(samples: Sequence[dict[str, Any]]) -> dict[str, Any]:
    ms = [m for s in samples for m in s.get("recall_ms", [])]
    empty = sum(int(s.get("empty", 0)) for s in samples)
    return {"n": len(ms), "p50": percentile(ms, 0.5), "p95": percentile(ms, 0.95),
            "empty_rate": round(empty / len(ms), 3) if ms else None}


# -- policy ------------------------------------------------------------------------------


def decide(m: dict[str, Any], t: Thresholds = Thresholds()) -> list[Action]:
    """What needs doing, in a stable order; [] means healthy."""
    out: list[Action] = []
    r = m.get("recall") or {}
    if (r.get("n") or 0) >= t.min_recalls and r.get("p95") is not None \
            and r["p95"] > t.recall_p95_ms:
        why = f"recall p95 {r['p95']:.0f} ms > {t.recall_p95_ms:.0f} ms over {r['n']} recalls"
        out += [Action("decay", "", why), Action("merge", "", why), Action("reindex", "", why)]
    for user, p in sorted((m.get("people") or {}).items()):
        facts = p.get("facts", 0)
        if facts > t.facts_per_person:
            why = f"{facts} active facts > {t.facts_per_person}"
            out += [Action("decay", user, why), Action("merge", user, why)]
        red, n = p.get("redundancy"), p.get("redundancy_n", 0)
        if red is not None and n >= t.redundancy_min_facts and red > t.redundancy:
            out.append(Action("merge", user, f"{red:.0%} of {n} sampled facts have a"
                                             f" near-duplicate (> {t.redundancy:.0%})"))
        stale = p.get("stale", 0)
        if facts > t.staleness_min_facts and stale / facts > t.staleness:
            out.append(Action("decay", user, f"{stale / facts:.0%} of {facts} facts stale"
                                             f" (> {t.staleness:.0%})"))
        old = p.get("old_episodes", 0)
        if old > t.old_episodes:
            out.append(Action("summarise", user, f"{old} active episodes older than"
                                                 f" {t.old_episode_days} days > {t.old_episodes}"))
    for table, s in sorted((m.get("tables") or {}).items()):
        live, dead = s.get("live", 0), s.get("dead", 0)
        if dead and dead / (live + dead) > t.dead_ratio:
            out.append(Action("vacuum", table, f"{dead} dead of {live + dead} rows"
                                               f" (> {t.dead_ratio:.0%})"))
    if m.get("purgeable", 0) > 0:
        out.append(Action("purge", "", f"{m['purgeable']} archived facts unaccessed for"
                                       f" {t.purge_days} days"))
    return _dedupe(out)


def _dedupe(actions: list[Action]) -> list[Action]:
    seen: dict[tuple[str, str], Action] = {}
    for a in actions:
        k = (a.kind, a.target)
        seen[k] = Action(a.kind, a.target, f"{seen[k].reason}; {a.reason}") if k in seen else a
    return list(seen.values())


# -- sampling ----------------------------------------------------------------------------


def sample(conn, t: Thresholds = Thresholds()) -> dict[str, Any]:
    """Measure everything ``decide`` needs. ``conn`` must be in admin scope."""
    q = conn.execute
    people: dict[str, dict[str, Any]] = {}

    def put(user: str, **kw) -> None:
        people.setdefault(user, {"facts": 0, "stale": 0, "episodes": 0, "old_episodes": 0,
                                 "bytes": 0, "redundancy": None, "redundancy_n": 0}).update(kw)

    for user, facts, stale, nbytes in q(
            "SELECT username, count(*),"
            " count(*) FILTER (WHERE NOT pinned AND importance < %s"
            "   AND coalesce(last_accessed, created_at) < now() - make_interval(days => %s)),"
            " sum(pg_column_size(f.*)) FROM memory.facts f WHERE state = 'active'"
            " GROUP BY username", (t.stale_importance, t.stale_days)):
        put(user, facts=facts, stale=stale, bytes=int(nbytes or 0))
    for user, eps, old, nbytes in q(
            "SELECT username, count(*),"
            " count(*) FILTER (WHERE occurred_at < now() - make_interval(days => %s)),"
            " sum(pg_column_size(e.*)) FROM memory.episodes e WHERE state = 'active'"
            " GROUP BY username", (t.old_episode_days,)):
        prev = people.get(user, {}).get("bytes", 0)
        put(user, episodes=eps, old_episodes=old, bytes=prev + int(nbytes or 0))
    for user, n, dup in q(
            "SELECT s.username, count(*), count(*) FILTER (WHERE nn.sim >= %s) FROM"
            " (SELECT id, username, embedding, row_number() OVER"
            "   (PARTITION BY username ORDER BY random()) AS rn"
            "  FROM memory.facts WHERE state = 'active' AND embedding IS NOT NULL) s"
            " CROSS JOIN LATERAL (SELECT 1 - (g.embedding <=> s.embedding) AS sim"
            "  FROM memory.facts g WHERE g.username = s.username AND g.state = 'active'"
            "  AND g.id <> s.id AND g.embedding IS NOT NULL"
            "  ORDER BY g.embedding <=> s.embedding LIMIT 1) nn"
            " WHERE s.rn <= %s GROUP BY s.username", (t.redundancy_sim, t.redundancy_sample)):
        put(user, redundancy=round(dup / n, 3), redundancy_n=n)

    tables = {name: {"live": live, "dead": dead, "bytes": total, "index_bytes": idx}
              for name, live, dead, total, idx in q(
                  "SELECT relname, n_live_tup, n_dead_tup, pg_total_relation_size(relid),"
                  " pg_indexes_size(relid) FROM pg_stat_user_tables WHERE schemaname = 'memory'")}
    jobs = dict(q("SELECT state, count(*) FROM memory.jobs GROUP BY state").fetchall())
    purgeable = q("SELECT count(*) FROM memory.facts WHERE state = 'archived' AND NOT pinned"
                  " AND coalesce(last_accessed, updated_at) < now() - make_interval(days => %s)",
                  (t.purge_days,)).fetchone()[0]
    daemon = [r[0] for r in q("SELECT metrics FROM memory.health_samples"
                              " WHERE metrics->>'source' = %s"
                              " AND at > now() - make_interval(days => %s)",
                              (DAEMON, t.recall_window_days))]
    return {"source": MAINTAIN,
            "db_bytes": q("SELECT pg_database_size(current_database())").fetchone()[0],
            "people": people, "tables": tables, "jobs": jobs, "purgeable": purgeable,
            "recall": pool_recalls(daemon)}


def record(conn, metrics: dict[str, Any]) -> None:
    conn.execute("INSERT INTO memory.health_samples (at, metrics) VALUES (clock_timestamp(), %s)",
                 (json.dumps(metrics),))


def housekeeping(conn, t: Thresholds = Thresholds()) -> dict[str, int]:
    """Always due: drop old health samples and long-finished jobs."""
    samples = conn.execute("DELETE FROM memory.health_samples"
                           " WHERE at < now() - make_interval(days => %s)",
                           (t.keep_samples_days,)).rowcount
    done = conn.execute("DELETE FROM memory.jobs WHERE state = 'done'"
                        " AND updated_at < now() - make_interval(days => %s)",
                        (t.keep_jobs_days,)).rowcount
    return {"old_samples": samples, "old_jobs": done}


# -- actions -----------------------------------------------------------------------------


def purge(store, t: Thresholds = Thresholds(), actor: str = MAINTAIN) -> int:
    """Delete archived, unpinned facts nobody touched for ``purge_days``.
    The audit row says a fact was purged, never what it said."""
    with store.db.scoped(None, "admin") as conn:
        rows = conn.execute(
            "SELECT id, username FROM memory.facts WHERE state = 'archived' AND NOT pinned"
            " AND coalesce(last_accessed, updated_at) < now() - make_interval(days => %s)",
            (t.purge_days,)).fetchall()
        if not rows:
            return 0
        store._delete_facts(conn, "f.id = ANY(%s)", ([r[0] for r in rows],))
        for fid, user in rows:
            store._audit(conn, fid, user, "fact.purge", actor,
                         {"reason": f"archived, unaccessed {t.purge_days} days"})
    store._prune_topics()
    return len(rows)


def vacuum(dsn: str, tables: Sequence[str], connect: Callable | None = None) -> list[str]:
    """``VACUUM (ANALYZE)`` each memory table, as the owner, outside a transaction."""
    from psycopg import connect as pg_connect, sql

    done = []
    with (connect or pg_connect)(dsn, autocommit=True) as conn:
        for name in tables:
            conn.execute(sql.SQL("VACUUM (ANALYZE) {}").format(sql.Identifier("memory", name)))
            done.append(name)
    return done


def maintain(store, dsn: str, *, t: Thresholds = Thresholds(), dry_run: bool = False,
             time_limit_s: float = 600.0, monotonic: Callable[[], float] = time.monotonic,
             vacuum_fn: Callable = vacuum, purge_fn: Callable = purge) -> dict[str, Any]:
    """Sample, record, housekeep, decide, run what runs. Returns a report.

    ``dry_run`` measures and decides but writes nothing. Actions still due
    when ``time_limit_s`` runs out are left for the next run (they are
    decided afresh from new metrics, so nothing needs resuming).
    """
    start = monotonic()
    with store.db.scoped(None, "admin") as conn:
        metrics = sample(conn, t)
        if not dry_run:
            record(conn, metrics)
            house = housekeeping(conn, t)
        else:
            house = {}
    actions = decide(metrics, t)
    report: dict[str, Any] = {"metrics": metrics, "housekeeping": house,
                              "actions": [{**asdict(a), "runs": a.runs} for a in actions],
                              "done": [], "deferred": [], "errors": []}
    if not actions:
        log.info("memory health: healthy, no work")
        return report
    for a in actions:
        if not a.runs:
            log.warning("memory health: needs %s%s: %s", a.kind,
                        f" for {a.target}" if a.target else "", a.reason)
    if dry_run:
        return report
    vac = [a.target for a in actions if a.kind == "vacuum"]
    for kind, run in (("purge", lambda: purge_fn(store, t)),
                      ("vacuum", lambda: vacuum_fn(dsn, vac))):
        if not any(a.kind == kind for a in actions):
            continue
        if monotonic() - start > time_limit_s:
            report["deferred"].append(kind)
            continue
        try:
            report["done"].append({"kind": kind, "result": run()})
        except Exception as exc:  # noqa: BLE001 - one failed action must not stop the rest
            log.warning("memory maintenance %s failed: %s", kind, exc)
            report["errors"].append({"kind": kind, "error": f"{type(exc).__name__}: {exc}"})
    return report
