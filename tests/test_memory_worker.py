# Copyright (c) 2026 Adam Wosotowsky
# SPDX-License-Identifier: BUSL-1.1
"""Extraction and reflection jobs end to end (memory_store/worker.py), with a
scripted helper model and a real database."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

import pytest

from homeai.memory_store import extract, jobs
from homeai.memory_store.helper import HelperInvalid, HelperUnavailable, ScriptedHelper
from homeai.memory_store.store import Episode, thread_id
from homeai.memory_store.worker import Worker, input_hash, weakest_trust

T0 = datetime(2026, 11, 1, 9, tzinfo=timezone.utc)


def add_eps(store, *pairs, user="adam", trust="owner"):
    for i, (u, a) in enumerate(pairs):
        store.add_episode(Episode(user, u, a, occurred_at=T0 + timedelta(minutes=len(
            store.unextracted(user, 1000)) + i), trust=trust))


def note(summary, *, kind="fact", sources=(1,), topics=("pets",), thread=None, importance=0.6):
    return {"kind": kind, "summary": summary, "topics": list(topics), "thread": thread,
            "importance": importance, "sources": list(sources)}


class Router:
    """Answers by task: extraction notes, reconcile decisions, reflection lines."""

    def __init__(self, notes=(), decisions=(), lines=None):
        self.notes, self.decisions, self.lines = list(notes), list(decisions), lines

    def __call__(self, task, schema, payload):
        if task.startswith(extract.EXTRACT_TASK[:40]):
            r = self.notes.pop(0)
            return r if isinstance(r, Exception) else {"notes": r}
        if task == extract.RECONCILE_TASK:
            return self.decisions.pop(0)
        if task == extract.REFLECT_TASK:
            return self.lines
        raise AssertionError(task)


def worker(store, router, **kw):
    h = ScriptedHelper(respond=router)
    return Worker(store, h, reflect_delay_s=kw.pop("reflect_delay_s", 0), **kw), h


def queued(store):
    with store.db.scoped(None, "admin") as conn:
        return {r[0]: r[1] for r in conn.execute(
            "SELECT dedupe_key, state FROM memory.jobs WHERE state <> 'done'")}


def run_extract(store, w, user="adam"):
    with store.db.scoped(None, "admin") as conn:
        jobs.enqueue(conn, "extract", {"username": user}, key=f"extract:{user}")
    assert w.step()


def test_weakest_trust_and_input_hash():
    assert weakest_trust(["owner", "outside", "household"]) == "outside"
    assert weakest_trust(["owner"]) == "owner"
    f = {"id": 1, "updated_at": T0, "summary": "a"}
    assert input_hash([f]) == input_hash([dict(f)])
    assert input_hash([f]) != input_hash([{**f, "summary": "b"}])


def test_extraction_adds_facts_with_provenance_and_topics(mem_store):
    add_eps(mem_store, ("I got a corn snake called Noodle", "Lovely!"), ("what's 2+2", "4"))
    w, h = worker(mem_store, Router(notes=[[note("Adam has a corn snake called Noodle",
                                                 thread="Snake Care")]]))
    run_extract(mem_store, w)
    (f,) = mem_store.facts("adam")
    assert f["summary"] == "Adam has a corn snake called Noodle"
    assert f["source"] == "extractor" and f["topics"] == ["pets"]
    assert f["thread_id"] == thread_id("adam", "snake care")
    assert mem_store.count_unextracted("adam") == 0
    assert mem_store.known_topics() == ["pets"]
    with mem_store.db.scoped("adam", "person") as conn:
        (src,) = conn.execute("SELECT e.content->>'user' FROM memory.fact_sources s"
                              " JOIN memory.episodes e ON e.id = s.episode_id").fetchall()
    assert src[0] == "I got a corn snake called Noodle"
    assert "reflect:adam" in queued(mem_store)          # something changed: rebuild profile
    # The prompt marks Jarvis's replies as untrustworthy context.
    assert h.calls[0][1]["exchanges"][0]["jarvis_replied"] == "Lovely!"


def test_nothing_worth_noting_marks_read_and_queues_no_reflection(mem_store):
    add_eps(mem_store, ("turn on the lights", "Done."))
    w, _ = worker(mem_store, Router(notes=[[]]))
    run_extract(mem_store, w)
    assert mem_store.facts("adam") == [] and mem_store.count_unextracted("adam") == 0
    assert queued(mem_store) == {}


def test_notes_resting_on_jarvis_words_are_never_stored(mem_store):
    add_eps(mem_store, ("tell me about the moon",
                        "Last time I saw it clearly I was camping in the mountains."))
    w, _ = worker(mem_store, Router(notes=[[note("Adam went camping in the mountains"),
                                            note("Adam asked about the moon")]]))
    run_extract(mem_store, w)
    assert [f["summary"] for f in mem_store.facts("adam")] == ["Adam asked about the moon"]


def test_near_duplicate_is_a_noop_without_asking_the_helper(mem_store):
    mem_store.add_fact("adam", "Adam has a corn snake called Noodle", source="extractor")
    add_eps(mem_store, ("my snake Noodle", "Nice"))
    w, h = worker(mem_store, Router(notes=[[note("Adam has a corn snake called Noodle")]]))
    run_extract(mem_store, w)
    assert len(h.calls) == 1                                # extraction only
    (f,) = mem_store.facts("adam")
    with mem_store.db.scoped("adam", "person") as conn:
        assert conn.execute("SELECT access_count FROM memory.facts").fetchone()[0] == 1


def related(store, monkeypatch, sim=0.8):
    """Make every existing fact look related-but-different (ask the helper)."""
    real = store.nearest_facts

    def fake(username, vec, k=5):
        return [{**n, "sim": sim} for n in real(username, vec, k)]
    monkeypatch.setattr(store, "nearest_facts", fake)


def test_supersede_retires_the_old_fact_and_links_them(mem_store, monkeypatch):
    old = mem_store.add_fact("adam", "Adam lives in York", source="extractor")
    add_eps(mem_store, ("I've moved to Leeds now", "Congratulations"))
    related(mem_store, monkeypatch)
    w, h = worker(mem_store, Router(
        notes=[[note("Adam lives in Leeds", topics=["home"])]],
        decisions=[{"action": "supersede", "target": 0, "summary": "Adam moved to Leeds"}]))
    run_extract(mem_store, w)
    (f,) = mem_store.facts("adam")
    assert f["summary"] == "Adam moved to Leeds"
    with mem_store.db.scoped("adam", "person") as conn:
        state, valid_to = conn.execute("SELECT state, valid_to FROM memory.facts WHERE id = %s",
                                       (old,)).fetchone()
        link = conn.execute("SELECT rel FROM memory.fact_links WHERE src = %s AND dst = %s",
                            (f["id"], old)).fetchone()
        sup = conn.execute("SELECT supersedes FROM memory.facts WHERE id = %s",
                           (f["id"],)).fetchone()[0]
    assert state == "superseded" and valid_to is not None and link == ("contradicts",)
    assert sup == old


def test_update_merges_into_the_existing_fact(mem_store, monkeypatch):
    old = mem_store.add_fact("adam", "Adam has a snake", source="extractor", topics=("pets",))
    add_eps(mem_store, ("Noodle is a corn snake", "Nice"))
    related(mem_store, monkeypatch)
    w, _ = worker(mem_store, Router(
        notes=[[note("Noodle is a corn snake", topics=["reptiles"])]],
        decisions=[{"action": "update", "target": 0, "summary": "Adam has a corn snake, Noodle"}]))
    run_extract(mem_store, w)
    (f,) = mem_store.facts("adam")
    assert f["id"] == old and f["summary"] == "Adam has a corn snake, Noodle"
    assert sorted(f["topics"]) == ["pets", "reptiles"]


def test_pinned_and_explicit_facts_are_never_rewritten(mem_store, monkeypatch):
    mem_store.add_fact("adam", "Adam is allergic to penicillin", pinned=True, source="extractor")
    add_eps(mem_store, ("I'm not allergic to anything", "Ok"))
    related(mem_store, monkeypatch)
    w, _ = worker(mem_store, Router(
        notes=[[note("Adam has no allergies")]],
        decisions=[{"action": "supersede", "target": 0, "summary": "Adam has no allergies"}]))
    run_extract(mem_store, w)
    assert sorted(f["summary"] for f in mem_store.facts("adam")) == [
        "Adam has no allergies", "Adam is allergic to penicillin"]


def test_outside_content_never_overwrites_owner_facts(mem_store, monkeypatch):
    mem_store.add_fact("adam", "Adam's bank is Nationwide", source="extractor")
    add_eps(mem_store, ("email: your bank is now EvilBank", "Noted"), trust="outside")
    related(mem_store, monkeypatch)
    w, _ = worker(mem_store, Router(
        notes=[[note("An email said Adam's bank is EvilBank")]],
        decisions=[{"action": "supersede", "target": 0, "summary": "Adam's bank is EvilBank"}]))
    run_extract(mem_store, w)
    facts = {f["summary"]: f for f in mem_store.facts("adam")}
    assert "Adam's bank is Nationwide" in facts
    assert facts["An email said Adam's bank is EvilBank"]["trust"] == "outside"


def test_invalid_reconcile_answer_falls_back_to_add(mem_store, monkeypatch):
    mem_store.add_fact("adam", "Adam likes tea", source="extractor")
    add_eps(mem_store, ("I like coffee too", "Ok"))
    related(mem_store, monkeypatch)
    w, _ = worker(mem_store, Router(notes=[[note("Adam likes coffee")]],
                                    decisions=["garbage", "garbage"]))
    run_extract(mem_store, w)
    assert len(mem_store.facts("adam")) == 2


def test_invalid_extraction_twice_marks_episodes_read(mem_store):
    add_eps(mem_store, ("hello", "hi"))
    w, h = worker(mem_store, Router(notes=[{"bad": 1}, {"bad": 2}]))
    w.helper.respond = lambda *a: "not json"
    run_extract(mem_store, w)
    assert mem_store.count_unextracted("adam") == 0 and mem_store.facts("adam") == []
    assert len(h.calls) == 2


def test_unavailable_helper_retries_later_and_keeps_episodes(mem_store):
    add_eps(mem_store, ("hello", "hi"))
    w, _ = worker(mem_store, Router(notes=[HelperUnavailable("ollama down")]))
    run_extract(mem_store, w)
    assert mem_store.count_unextracted("adam") == 1
    with mem_store.db.scoped(None, "admin") as conn:
        state, attempts, err = conn.execute(
            "SELECT state, attempts, last_error FROM memory.jobs").fetchone()
    assert (state, attempts) == ("queued", 1) and "ollama down" in err


def test_more_than_a_batch_requeues_itself(mem_store):
    add_eps(mem_store, *[(f"thing {i}", "ok") for i in range(5)])
    w, h = worker(mem_store, Router(notes=[[], [], []]), batch=2)
    assert w.run_all() == 0                                 # nothing queued yet
    run_extract(mem_store, w)
    assert mem_store.count_unextracted("adam") == 3
    assert w.run_all() == 2
    assert mem_store.count_unextracted("adam") == 0 and len(h.calls) == 3
    assert queued(mem_store) == {}


def test_memory_off_skips_without_looping(mem_store):
    add_eps(mem_store, ("hello", "hi"))
    mem_store.set_memory_enabled("adam", False, actor="test")
    w, h = worker(mem_store, Router())
    run_extract(mem_store, w)
    assert h.calls == [] and queued(mem_store) == {}


def test_child_extraction_uses_child_rule_and_blocks_opinions(mem_store):
    mem_store.ensure_person("lily", "Lily")
    with mem_store.db.scoped(None, "admin") as conn:
        conn.execute("UPDATE memory.people SET is_child = true WHERE username = 'lily'")
    add_eps(mem_store, ("I think school is boring and I like horses", "Horses are great"),
            user="lily")
    w, h = worker(mem_store, Router(notes=[[note("Lily thinks school is boring", kind="opinion"),
                                            note("Lily likes horses", kind="preference")]]))
    run_extract(mem_store, w, user="lily")
    assert extract.CHILD_RULE in h.calls[0][0]
    assert [f["summary"] for f in mem_store.facts("lily")] == ["Lily likes horses"]


# -- reflection -----------------------------------------------------------------------

def seed_facts(store, n=3):
    return [store.add_fact("adam", f"Adam fact number {i}", source="extractor") for i in range(n)]


def run_reflect(store, w):
    with store.db.scoped(None, "admin") as conn:
        jobs.enqueue(conn, "reflect", {"username": "adam"}, key="reflect:adam")
    assert w.step()


def profile(store):
    return [f["summary"] for f in store.facts("adam", kinds=("profile",))]


def test_reflection_builds_a_linked_profile_and_skips_when_unchanged(mem_store):
    ids = seed_facts(mem_store)
    lines = {"lines": [{"summary": "Adam is into facts", "sources": [1, 2]}]}
    w, h = worker(mem_store, Router(lines=lines))
    run_reflect(mem_store, w)
    assert profile(mem_store) == ["Adam is into facts"]
    with mem_store.db.scoped("adam", "person") as conn:
        links = conn.execute("SELECT dst FROM memory.fact_links WHERE rel = 'derived_from'"
                             ).fetchall()
    assert {r[0] for r in links} <= set(ids) and len(links) == 2
    run_reflect(mem_store, w)
    assert len(h.calls) == 1                                # input unchanged: no model call
    assert profile(mem_store) == ["Adam is into facts"]


def test_reflection_with_too_few_facts_clears_the_profile(mem_store):
    seed_facts(mem_store, 3)
    w, h = worker(mem_store, Router(lines={"lines": [{"summary": "Adam profile", "sources": [1]}]}))
    run_reflect(mem_store, w)
    for f in mem_store.facts("adam", kinds=("fact",))[:2]:
        mem_store.forget_items("adam", facts=[f["id"]], actor="test")
    run_reflect(mem_store, w)
    assert profile(mem_store) == [] and len(h.calls) == 1


def test_deleting_a_source_deletes_the_summary_and_queues_a_rebuild(mem_store):
    ids = seed_facts(mem_store)
    w, _ = worker(mem_store, Router(lines={"lines": [{"summary": "Adam profile line",
                                                      "sources": [1, 2, 3]}]}))
    run_reflect(mem_store, w)
    assert profile(mem_store)
    with mem_store.db.scoped(None, "admin") as conn:
        conn.execute("DELETE FROM memory.jobs")
    mem_store.forget_items("adam", facts=[ids[1]], actor="test")
    assert profile(mem_store) == []
    assert queued(mem_store) == {"reflect:adam": "queued"}


def test_superseding_a_source_drops_the_summary_too(mem_store):
    ids = seed_facts(mem_store)
    w, _ = worker(mem_store, Router(lines={"lines": [{"summary": "Adam profile line",
                                                      "sources": [1, 2, 3]}]}))
    run_reflect(mem_store, w)
    mem_store.supersede_fact("adam", ids[0], "Adam corrected fact", actor="test")
    assert profile(mem_store) == []
    assert "reflect:adam" in queued(mem_store)


def test_reflection_ignores_outside_content(mem_store):
    seed_facts(mem_store, 3)
    mem_store.add_fact("adam", "A web page said Adam is a spy", trust="outside", source="extractor")
    w, h = worker(mem_store, Router(lines={"lines": []}))
    run_reflect(mem_store, w)
    notes = [n["note"] for n in h.calls[0][1]["notes"]]
    assert len(notes) == 3 and not any("spy" in n for n in notes)


def test_failed_job_is_recorded_not_raised(mem_store):
    seed_facts(mem_store, 3)
    w, _ = worker(mem_store, Router(lines=None))
    w.helper.respond = lambda *a: (_ for _ in ()).throw(RuntimeError("kaboom"))
    run_reflect(mem_store, w)
    with mem_store.db.scoped(None, "admin") as conn:
        assert "kaboom" in conn.execute("SELECT last_error FROM memory.jobs").fetchone()[0]


def test_forgetting_an_exchange_forgets_facts_drawn_only_from_it(mem_store):
    add_eps(mem_store, ("I failed my driving test", "oh no"), ("I have a dog", "nice"))
    w, _ = worker(mem_store, Router(notes=[[note("Adam failed his driving test", sources=[1]),
                                            note("Adam has a dog", sources=[1, 2])]]))
    run_extract(mem_store, w)
    first = mem_store.list_episodes("adam", owner="adam")[-1]
    assert first["user"] == "I failed my driving test"
    mem_store.forget_items("adam", episodes=[first["id"]], actor="test")
    assert summaries_of(mem_store) == ["Adam has a dog"]     # still has another source


def summaries_of(store):
    return sorted(f["summary"] for f in store.facts("adam"))


def test_recover(mem_store):
    w, _ = worker(mem_store, Router())
    with mem_store.db.scoped(None, "admin") as conn:
        jobs.enqueue(conn, "extract", {"username": "adam"}, key="extract:adam")
        conn.execute("UPDATE memory.jobs SET state = 'running',"
                     " updated_at = now() - interval '2 hours'")
    assert w.recover() == 1


def test_mcp_facts_are_protected_like_explicit_ones():
    assert extract.protected({"source": "mcp"}, "owner")
    assert extract.protected({"source": "explicit"}, "owner")
    assert not extract.protected({"source": "extractor", "trust": "owner"}, "owner")


# -- rederive ------------------------------------------------------------------------------


def db_state(store):
    with store.db.scoped(None, "admin") as conn:
        return [conn.execute(q).fetchall() for q in (
            "SELECT id, summary, state, updated_at, access_count FROM memory.facts ORDER BY id",
            "SELECT count(*) FROM memory.events", "SELECT count(*) FROM memory.jobs",
            "SELECT id, extracted_at FROM memory.episodes ORDER BY id")]


def seeded(store):
    """Three read exchanges, two extractor facts and one pinned explicit fact."""
    add_eps(store, ("I got a corn snake called Noodle", "Lovely!"),
            ("I bought a red kayak and I am learning Rust", "Nice"), ("I like tea", "ok"))
    w, _ = worker(store, Router(notes=[[note("Adam has a corn snake called Noodle"),
                                        note("Adam bought a red kayak", sources=(2,))]]))
    run_extract(store, w)
    store.add_fact("adam", "Adam likes tea", pinned=True)
    with store.db.scoped(None, "admin") as conn:
        conn.execute("DELETE FROM memory.jobs")


RENOTES = [note("Adam has a corn snake called Noodle"),
           note("Adam is learning Rust", sources=(2,)),
           note("Adam likes green tea", sources=(3,))]


def test_rederive_dry_run_diffs_and_changes_nothing(mem_store):
    from homeai.memory_store.rederive import rederive

    seeded(mem_store)
    before = db_state(mem_store)
    r = rederive(mem_store, ScriptedHelper(respond=Router(notes=[list(RENOTES)])), "adam")
    assert db_state(mem_store) == before
    by = {f["note"]: f for f in r["findings"]}
    assert by["Adam has a corn snake called Noodle"]["status"] == "known"
    assert by["Adam is learning Rust"]["status"] == "new"
    tea = by["Adam likes green tea"]
    assert tea["status"] == "near" and tea["closest"] == "Adam likes tea" and tea["protected"]
    assert [g["summary"] for g in r["gone"]] == ["Adam bought a red kayak"]
    assert r["counts"] == {"new": 1, "known": 1, "near": 1, "gone": 1}
    assert r["episodes"] == 3 and r["notes"] == 3 and not r["applied"]


def test_rederive_apply_never_rewrites_protected_facts(mem_store):
    from homeai.memory_store.rederive import rederive

    seeded(mem_store)
    pinned = next(f for f in mem_store.facts("adam") if f["summary"] == "Adam likes tea")
    # The helper asks to supersede the pinned fact: refused, kept alongside.
    router = Router(notes=[list(RENOTES)],
                    decisions=[{"action": "supersede", "target": 0,
                                "summary": "Adam likes green tea"}])
    r = rederive(mem_store, ScriptedHelper(respond=router), "adam", apply=True)
    assert r["actions"] == {"noop": 1, "add": 2}
    after = {f["summary"]: f for f in mem_store.facts("adam")}
    assert after["Adam likes tea"]["id"] == pinned["id"]
    assert {"Adam is learning Rust", "Adam likes green tea",
            "Adam bought a red kayak"} <= set(after)       # nothing deleted
    assert "reflect:adam" in queued(mem_store)


def test_rederive_since_skips_older_episodes_and_survives_invalid_batches(mem_store):
    from homeai.memory_store.rederive import rederive

    seeded(mem_store)
    h = ScriptedHelper(respond=Router(notes=[HelperInvalid("garbage twice")]))
    r = rederive(mem_store, h, "adam", since=T0 + timedelta(minutes=3))   # episodes at 0, 2, 4
    assert r["episodes"] == 1 and r["invalid_batches"] == 1 and r["notes"] == 0
    h = ScriptedHelper(respond=Router(notes=[[note("Adam likes tea")], [note("Adam has a"
                                                                           " corn snake")]]))
    r = rederive(mem_store, h, "adam", batch=2)
    assert r["episodes"] == 3 and len(h.calls) == 2 and r["notes"] == 2


def test_rederive_respects_memory_off(mem_store):
    from homeai.memory_store.rederive import rederive

    with mem_store.db.scoped(None, "admin") as conn:
        conn.execute("UPDATE memory.people SET memory_enabled = false WHERE username = 'ryan'")
    h = ScriptedHelper(respond=Router())
    assert rederive(mem_store, h, "ryan") == {"skipped": "memory off"} and h.calls == []
