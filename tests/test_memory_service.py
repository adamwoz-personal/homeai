# Copyright (c) 2026 Adam Wosotowsky
# SPDX-License-Identifier: BUSL-1.1
"""The daemon-facing memory service (memory_store/service.py).

The rule under test throughout: memory never blocks, slows or breaks a reply.
"""

from __future__ import annotations

import time
from dataclasses import replace
from datetime import datetime, timedelta, timezone

import pytest

from homeai.config import MemoryConfig
from homeai.memory_store.service import MemoryService
from homeai.transcript import Turn

CFG = MemoryConfig(enabled=True, dsn="unused", recall_timeout_ms=2000, budget_tokens=600,
                   min_similarity=0.3, ignore_speakers=("ryan",), admins=("adam",),
                   extract=False, helper_url="http://127.0.0.1:9")


def turn(**kw):
    base = dict(heard="hey jarvis what's a good beginner snake", reply="A corn snake is gentle.",
                verdict="allow", speaker="Adam", speaker_score=0.8)
    return Turn(**{**base, **kw})


class Exploding:
    """A store that fails the test if the service touches it."""

    def __getattr__(self, name):
        raise AssertionError(f"store.{name} used")


def ready_service(store, cfg=CFG):
    svc = MemoryService(cfg, store_factory=lambda: store).start()
    assert svc.wait_ready(5)
    return svc


# -- no database needed ----------------------------------------------------------------

def test_recall_is_empty_before_ready():
    svc = MemoryService(CFG, store_factory=lambda: Exploding())
    assert svc.recall("Adam", "what's my favourite snake") == ""


@pytest.mark.parametrize("speaker,utterance", [
    (None, "what's my favourite snake"), ("Adam", "thank you"), ("Ryan", "what's my favourite snake"),
    ("Adam", ""),
])
def test_recall_gating_never_touches_the_store(speaker, utterance):
    svc = ready_service(Exploding())
    try:
        assert svc.recall(speaker, utterance) == ""
    finally:
        svc._stop.set()


def test_recall_swallows_errors_and_warns_once(caplog):
    class Broken:
        class embedder:
            @staticmethod
            def embed(texts, kind):
                raise RuntimeError("model gone")

    svc = ready_service(Broken())
    with caplog.at_level("WARNING"):
        assert svc.recall("Adam", "what's my favourite snake") == ""
        assert svc.recall("Adam", "what's my favourite snake") == ""
    assert sum("recall failed" in r.message for r in caplog.records) == 1
    svc._stop.set()


def test_recall_skips_sql_when_embedding_used_the_budget():
    class Slow:
        class embedder:
            @staticmethod
            def embed(texts, kind):
                time.sleep(0.05)
                return [[0.0]]
        db = Exploding()

    svc = ready_service(Slow(), replace(CFG, recall_timeout_ms=20))
    assert svc.recall("Adam", "what's my favourite snake") == ""
    assert svc.last_recall_ms >= 50
    svc._stop.set()


def test_on_turn_filters_before_queueing():
    svc = MemoryService(CFG, store_factory=lambda: Exploding())
    for t in (turn(speaker=None), turn(verdict="unverified"), turn(speaker="Ryan"),
              turn(verdict="home-lights"), turn(reply="")):
        svc.on_turn(t)
    assert svc._q.qsize() == 0
    svc.on_turn(turn())
    kind, ep, display = svc._q.get_nowait()
    assert (kind, ep.username, ep.user, display) == ("episode", "adam", "what's a good beginner snake",
                                                     "Adam")
    assert ep.confidence == pytest.approx(0.8)


def test_on_turn_never_blocks_when_full(monkeypatch):
    from homeai.memory_store import service

    monkeypatch.setattr(service, "QUEUE_MAX", 2)
    svc = MemoryService(CFG, store_factory=lambda: Exploding())
    start = time.perf_counter()
    for _ in range(10):
        svc.on_turn(turn())
    assert time.perf_counter() - start < 0.5
    assert svc._q.qsize() == 2


def test_unreachable_database_retries_quietly_and_closes_fast(caplog):
    calls = []

    def factory():
        calls.append(1)
        raise OSError("connection refused")

    with caplog.at_level("WARNING"):
        svc = MemoryService(CFG, store_factory=factory).start()
        time.sleep(0.1)
        assert not svc.ready
        assert svc.recall("Adam", "what's my favourite snake") == ""
        start = time.perf_counter()
        svc.close()
    assert time.perf_counter() - start < 1
    assert calls == [1]
    assert any("unavailable" in r.message for r in caplog.records)


def test_warn_throttle(caplog):
    svc = MemoryService(CFG)
    with caplog.at_level("WARNING"):
        for _ in range(5):
            svc._warn("k", "same problem")
        svc._warn("other", "different problem")
    assert [r.message for r in caplog.records] == ["same problem", "different problem"]


# -- with a throwaway database --------------------------------------------------------------

def test_records_then_recalls_end_to_end(mem_store):
    svc = ready_service(mem_store)
    try:
        svc.on_turn(turn())
        svc.on_turn(turn(speaker="Ryan", heard="I like king cobras"))   # test voice: ignored
        assert svc.flush()
        [e] = mem_store.list_episodes("adam", "admin")
        assert e["user"] == "what's a good beginner snake" and e["username"] == "adam"
        block = svc.recall("Adam", "which beginner snake did you suggest")
        assert "with Adam" in block and "corn snake" in block
        assert svc.recall("Joe", "which beginner snake did you suggest") == ""
    finally:
        svc.close()


def test_new_speaker_becomes_a_person(mem_store):
    svc = ready_service(mem_store)
    try:
        svc.on_turn(turn(speaker="Mary Jane"))
        assert svc.flush()
        p = mem_store.person("mary_jane")
        assert p["display_name"] == "Mary Jane" and not p["is_admin"]
    finally:
        svc.close()


def test_recalled_facts_are_touched(mem_store):
    fid = mem_store.add_fact("adam", "Adam keeps a corn snake called Noodle", kind="profile")
    svc = ready_service(mem_store)
    try:
        assert "Noodle" in svc.recall("Adam", "tell me about my snake")
        assert svc.flush()
        with mem_store.db.scoped("adam") as conn:
            count, last = conn.execute("SELECT access_count, last_accessed FROM memory.facts"
                                       " WHERE id = %s", (fid,)).fetchone()
        assert count == 1 and last is not None
    finally:
        svc.close()


def test_people_never_recall_each_other(mem_store):
    mem_store.add_fact("ryan", "Ryan's secret hideout is behind the shed", importance=1.0)
    svc = ready_service(mem_store)
    try:
        assert "hideout" not in svc.recall("Adam", "where is the secret hideout behind the shed")
    finally:
        svc.close()


def test_time_phrases_use_the_local_clock(mem_store):
    now = datetime(2026, 11, 4, 15, tzinfo=timezone(timedelta(hours=-5)))
    from homeai.memory_store.store import Episode

    # Equally similar to the question, so only the time window can separate them.
    mem_store.add_episode(Episode("adam", "beginner snake question", "corn",
                                  occurred_at=now - timedelta(days=1)))
    mem_store.add_episode(Episode("adam", "beginner snake question", "python",
                                  occurred_at=now - timedelta(days=20)))
    svc = MemoryService(CFG, store_factory=lambda: mem_store, clock=lambda: now).start()
    assert svc.wait_ready(5)
    try:
        block = svc.recall("Adam", "what beginner snake did you say yesterday")
        assert "corn" in block and "python" not in block
    finally:
        svc.close()
