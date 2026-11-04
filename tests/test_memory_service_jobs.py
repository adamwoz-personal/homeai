# Copyright (c) 2026 Adam Wosotowsky
# SPDX-License-Identifier: BUSL-1.1
"""Phase 2 in the memory service: queueing extraction, the idle-only worker,
and spoken commands. Memory still never breaks a reply."""

from __future__ import annotations

import time
from dataclasses import replace

from homeai.memory_store import jobs, service as svc_mod
from homeai.memory_store.helper import ScriptedHelper
from homeai.memory_store.service import MSG_NOT_READY, MemoryService
from homeai.memory_store.spoken import MSG_BROKEN, MSG_UNKNOWN
from tests.test_memory_service import CFG, Exploding, turn

XCFG = replace(CFG, extract=True, extract_idle_s=600, extract_batch=3)


class Mono:
    def __init__(self, t=10_000.0):
        self.t = t

    def __call__(self):
        return self.t


def start(store, cfg=XCFG, helper=None, mono=None):
    s = MemoryService(cfg, store_factory=lambda: store,
                      helper_factory=lambda: helper or ScriptedHelper(respond=lambda *a: {"notes": []}),
                      monotonic=mono or Mono()).start()
    assert s.wait_ready(5)
    return s


def due(store):
    with store.db.scoped(None, "admin") as conn:
        return conn.execute("SELECT dedupe_key, not_before <= now() FROM memory.jobs"
                            " WHERE state = 'queued'").fetchall()


# -- commands ------------------------------------------------------------------------------

def test_commands_before_ready():
    s = MemoryService(CFG, store_factory=lambda: Exploding())
    assert s.handle_command("Adam", "remember that my sister is Kate") == (MSG_NOT_READY, False)
    assert s.handle_command("Adam", "what's the weather") is None


def test_command_errors_are_spoken_not_raised(caplog):
    class Broken:
        def person(self, u):
            raise RuntimeError("db gone")

    s = MemoryService(CFG, store_factory=lambda: Broken()).start()
    assert s.wait_ready(5)
    try:
        assert s.handle_command("Adam", "hey jarvis remember that my sister is Kate") == \
            (MSG_BROKEN, False)
        assert s.handle_command("Adam", "what's the weather") is None
    finally:
        s._stop.set()


def test_remember_through_the_service(mem_store):
    s = start(mem_store, CFG)
    try:
        assert s.handle_command("Adam", "Hey Jarvis, remember that my sister is Kate") == \
            ("Okay, I'll remember that.", False)
        assert [f["summary"] for f in mem_store.facts("adam")] == ["Adam's sister is Kate"]
        assert s.handle_command("Ryan", "remember that I'm Adam") == (MSG_UNKNOWN, False)
        assert s.handle_command(None, "remember that I'm Adam") == (MSG_UNKNOWN, False)
    finally:
        s.close()


def test_forget_last_waits_for_queued_writes(mem_store):
    s = start(mem_store, CFG)
    try:
        s.on_turn(turn(heard="hey jarvis my pin is 4321", reply="Okay."))
        reply, _ = s.handle_command("Adam", "forget what I just said")
        assert reply == "Okay, I've forgotten that."
        assert mem_store.list_episodes("adam", owner="adam") == []
    finally:
        s.close()


# -- extraction scheduling ---------------------------------------------------------------

def test_a_new_episode_queues_extraction_for_when_the_person_goes_quiet(mem_store):
    s = start(mem_store, mono=Mono())
    try:
        s.on_turn(turn())
        assert s.flush()
        assert due(mem_store) == [("extract:adam", False)]
    finally:
        s.close()


def test_a_full_batch_is_due_at_once(mem_store):
    s = start(mem_store)
    try:
        for i in range(3):
            s.on_turn(turn(heard=f"hey jarvis question {i}"))
        assert s.flush()
        assert due(mem_store) == [("extract:adam", True)]
    finally:
        s.close()


def test_no_extraction_queued_when_disabled(mem_store):
    s = start(mem_store, CFG)
    try:
        s.on_turn(turn())
        assert s.flush()
        assert due(mem_store) == [] and s.worker is None
    finally:
        s.close()


def test_worker_only_runs_when_quiet(mem_store, monkeypatch):
    monkeypatch.setattr(svc_mod, "WORKER_POLL_S", 3600)      # drive it by hand
    mono = Mono()
    helper = ScriptedHelper(respond=lambda *a: {"notes": []})
    released = []
    helper.release = lambda restore=None: released.append(restore) or True
    s = start(mem_store, helper=helper, mono=mono)
    try:
        s.on_turn(turn())
        assert s.flush()
        with mem_store.db.scoped(None, "admin") as conn:
            jobs.run_now(conn, "extract:adam")
        assert not s.quiet() and s.work_once() == 0           # someone is talking
        assert released == []                                  # nothing ran: the voice model is untouched
        mono.t += svc_mod.QUIET_S
        assert s.quiet() and s.work_once() == 1
        assert mem_store.count_unextracted("adam") == 0 and len(helper.calls) == 1
        assert released == [XCFG.restore_model]                # the voice model comes back after jobs
    finally:
        s.close()


def test_a_failing_release_never_breaks_the_worker(mem_store, monkeypatch, caplog):
    monkeypatch.setattr(svc_mod, "WORKER_POLL_S", 3600)
    mono = Mono()
    helper = ScriptedHelper(respond=lambda *a: {"notes": []})
    helper.release = lambda restore=None: (_ for _ in ()).throw(RuntimeError("ollama gone"))
    s = start(mem_store, helper=helper, mono=mono)
    try:
        s.on_turn(turn())
        assert s.flush()
        with mem_store.db.scoped(None, "admin") as conn:
            jobs.run_now(conn, "extract:adam")
        mono.t += svc_mod.QUIET_S
        assert s.work_once() == 1
        assert any("release failed" in r.message for r in caplog.records)
    finally:
        s.close()


def test_worker_survives_a_broken_step(mem_store, caplog):
    s = start(mem_store)
    try:
        s.worker.step = lambda: (_ for _ in ()).throw(RuntimeError("db gone"))
        assert s.work_once() == 0
        assert any("job loop failed" in r.message for r in caplog.records)
    finally:
        s.close()


def test_helper_construction_failure_leaves_recall_working(mem_store, caplog):
    def bad():
        raise RuntimeError("no helper")

    s = MemoryService(XCFG, store_factory=lambda: mem_store, helper_factory=bad).start()
    assert s.wait_ready(5)
    try:
        assert s.worker is None
        assert any("extraction disabled" in r.message for r in caplog.records)
        s.on_turn(turn())
        assert s.flush() and mem_store.list_episodes("adam", owner="adam")
    finally:
        s.close()


def test_worker_thread_recovers_on_start_and_close_is_fast(mem_store):
    with mem_store.db.scoped(None, "admin") as conn:
        jobs.enqueue(conn, "extract", {"username": "adam"}, key="extract:adam")
        conn.execute("UPDATE memory.jobs SET state = 'running',"
                     " updated_at = now() - interval '2 hours'")
    s = start(mem_store)

    def state():
        with mem_store.db.scoped(None, "admin") as conn:
            return conn.execute("SELECT state FROM memory.jobs").fetchone()[0]

    deadline = time.monotonic() + 5
    while time.monotonic() < deadline and state() != "done":
        time.sleep(0.02)
    assert state() == "done"                    # recovered, then run while quiet
    t0 = time.perf_counter()
    s.close()
    assert time.perf_counter() - t0 < 2
