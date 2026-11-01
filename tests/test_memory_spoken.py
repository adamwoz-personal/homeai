# Copyright (c) 2026 Adam Wosotowsky
# SPDX-License-Identifier: BUSL-1.1
"""Spoken memory commands carried out (memory_store/spoken.py), real database."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

import pytest

from homeai.memory_store import spoken as sp
from homeai.memory_store.spoken import SpokenCommands
from homeai.memory_store.store import Episode


class Clock:
    def __init__(self):
        self.t = 1000.0

    def __call__(self):
        return self.t


@pytest.fixture
def cmds(mem_store):
    clock = Clock()
    c = SpokenCommands(mem_store, min_similarity=0.0, monotonic=clock)
    c.clock = clock
    return c


def say(c, text, who="adam", display="Adam"):
    return c.handle(who, display, text)


def summaries(store, user="adam"):
    return sorted(f["summary"] for f in store.facts(user))


def test_not_a_command(cmds):
    assert say(cmds, "what's the weather") is None


@pytest.mark.parametrize("who", [None, "stranger"])
def test_unknown_voice_is_refused(cmds, who):
    assert say(cmds, "remember that my code is 1234", who=who) == (sp.MSG_UNKNOWN, False)


def test_remember_and_always_remember(cmds, mem_store):
    assert say(cmds, "remember that my sister is called Kate") == ("Okay, I'll remember that.",
                                                                   False)
    reply, _ = say(cmds, "always remember that I'm allergic to penicillin")
    assert reply == "I'll always remember that."
    facts = {f["summary"]: f for f in mem_store.facts("adam")}
    assert facts["Adam's sister is called Kate"]["source"] == "explicit"
    assert facts["Adam is allergic to penicillin"]["pinned"] is True


def test_remember_respects_memory_off(cmds, mem_store):
    mem_store.set_memory_enabled("adam", False, actor="t")
    assert say(cmds, "remember that my sister is Kate") == (sp.MSG_OFF, False)
    assert mem_store.facts("adam") == []


def test_recall_me_lists_profile_first_and_counts_the_rest(cmds, mem_store):
    for i in range(7):
        mem_store.add_fact("adam", f"Adam fact {i}", importance=i / 10)
    mem_store.add_fact("adam", "Adam is a builder", kind="profile")
    reply, listen = say(cmds, "what do you remember about me")
    assert reply.startswith("Here's what I remember about you. Adam is a builder.")
    assert "Adam fact 6." in reply and "And 3 more things." in reply and not listen


def test_recall_me_with_nothing(cmds):
    assert say(cmds, "what do you remember about me")[0].startswith("I don't have anything")


def test_recall_about_never_shows_someone_elses_memories(cmds, mem_store):
    mem_store.add_fact("ryan", "Ryan's snake is called Biscuit")
    mem_store.add_fact("household", "The household snake is called Mango")
    mem_store.add_fact("adam", "Adam's snake is called Noodle")
    reply, _ = say(cmds, "what do you remember about the snake")
    assert "Noodle" in reply and "Biscuit" not in reply and "Mango" not in reply


def test_forget_about_never_offers_household_memories(cmds, mem_store):
    mem_store.add_fact("household", "The household snake is called Mango")
    assert say(cmds, "forget what I said about the snake")[0] == \
        "I don't have anything about the snake."


def test_recall_about_nothing(cmds):
    assert say(cmds, "what do you remember about volcanoes")[0] == \
        "I don't remember anything about volcanoes."


def test_correction_after_a_listing_supersedes_the_fact(cmds, mem_store):
    old = mem_store.add_fact("adam", "Adam's snake is called Noodle")
    say(cmds, "what do you remember about me")
    assert say(cmds, "that's wrong, my snake is called Pickle") == \
        ("Thanks, I've corrected that.", False)
    assert summaries(mem_store) == ["Adam's snake is called Pickle"]
    with mem_store.db.scoped("adam", "person") as conn:
        assert conn.execute("SELECT state FROM memory.facts WHERE id = %s",
                            (old,)).fetchone()[0] == "superseded"


def test_correction_picks_the_closest_listed_fact(cmds, mem_store):
    mem_store.add_fact("adam", "Adam lives in York", importance=0.9)
    mem_store.add_fact("adam", "Adam's snake is called Noodle", importance=0.8)
    say(cmds, "what do you remember about me")
    say(cmds, "that's wrong, Adam's snake is called Pickle")
    assert summaries(mem_store) == ["Adam lives in York", "Adam's snake is called Pickle"]


def test_correction_without_a_listing_is_not_a_command(cmds, mem_store):
    mem_store.add_fact("adam", "Adam's snake is called Noodle")
    assert say(cmds, "that's wrong, my snake is called Pickle") is None
    assert summaries(mem_store) == ["Adam's snake is called Noodle"]


def test_correction_expires(cmds, mem_store):
    mem_store.add_fact("adam", "Adam's snake is called Noodle")
    say(cmds, "what do you remember about me")
    cmds.clock.t += sp.LISTED_S + 1
    assert say(cmds, "that's wrong, my snake is called Pickle") is None


def test_forget_last_removes_the_note_just_made(cmds, mem_store):
    say(cmds, "remember that my pin is the year I was born")
    assert say(cmds, "forget that") == ("Okay, I've forgotten that.", False)
    assert mem_store.facts("adam") == []


def test_forget_last_removes_the_latest_exchange(cmds, mem_store):
    mem_store.add_episode(Episode("adam", "older thing", "ok",
                                  occurred_at=datetime.now(timezone.utc) - timedelta(minutes=5)))
    mem_store.add_episode(Episode("adam", "embarrassing thing", "ok"))
    assert say(cmds, "forget what I just said")[0] == "Okay, I've forgotten that."
    eps = [e["user"] for e in mem_store.list_episodes("adam", owner="adam")]
    assert eps == ["older thing"]


def test_forget_last_with_nothing_recent(cmds, mem_store):
    mem_store.add_episode(Episode("adam", "ancient", "ok",
                                  occurred_at=datetime.now(timezone.utc) - timedelta(days=2)))
    assert say(cmds, "forget that")[0].startswith("I'm not sure what to forget")
    assert len(mem_store.list_episodes("adam", owner="adam")) == 1


def test_forget_about_asks_then_deletes_facts_and_their_sources(cmds, mem_store):
    mem_store.add_episode(Episode("adam", "the party is on Saturday", "Fun"))
    mem_store.add_fact("adam", "Adam's party is on Saturday")
    mem_store.add_fact("adam", "Adam builds sheds")
    reply, listen = say(cmds, "forget what I said about the party")
    assert listen and "Shall I forget" in reply and "party" in reply.lower()
    assert len(mem_store.facts("adam")) == 2                 # nothing yet
    assert say(cmds, "yes") == ("Done, I've forgotten that.", False)
    assert summaries(mem_store) == ["Adam builds sheds"]
    assert not any("party" in e["user"] for e in mem_store.list_episodes("adam", owner="adam"))


def test_forget_about_no_keeps_everything(cmds, mem_store):
    mem_store.add_fact("adam", "Adam's party is on Saturday")
    say(cmds, "forget what I said about the party")
    assert say(cmds, "no") == ("Okay, I'll keep them.", False)
    assert len(mem_store.facts("adam")) == 1


def test_unrelated_answer_lets_the_question_lapse(cmds, mem_store):
    mem_store.add_fact("adam", "Adam's party is on Saturday")
    say(cmds, "forget what I said about the party")
    assert say(cmds, "what's the weather") is None
    assert say(cmds, "yes") is None                          # the question is gone
    assert len(mem_store.facts("adam")) == 1


def test_confirmation_expires(cmds, mem_store):
    mem_store.add_fact("adam", "Adam's party is on Saturday")
    say(cmds, "forget what I said about the party")
    cmds.clock.t += sp.PENDING_S + 1
    assert say(cmds, "yes") is None
    assert len(mem_store.facts("adam")) == 1


def test_someone_else_cannot_confirm(cmds, mem_store):
    mem_store.add_fact("adam", "Adam's party is on Saturday")
    say(cmds, "forget what I said about the party")
    assert say(cmds, "yes", who="ryan", display="Ryan") is None
    assert len(mem_store.facts("adam")) == 1


def test_forget_thread_deletes_the_whole_idea(cmds, mem_store, monkeypatch):
    from homeai.memory_store.store import thread_id

    tid = thread_id("adam", "shed")
    a = mem_store.add_fact("adam", "Adam plans a shed")
    b = mem_store.add_fact("adam", "The shed will be cedar")
    mem_store.add_fact("adam", "Adam has a snake")
    with mem_store.db.scoped("adam", "person") as conn:
        conn.execute("UPDATE memory.facts SET thread_id = %s WHERE id = ANY(%s)", (tid, [a, b]))
    reply, listen = say(cmds, "forget that whole idea")
    assert listen and "2 memories" in reply
    say(cmds, "yes")
    assert summaries(mem_store) == ["Adam has a snake"]


def test_forget_all_asks_with_a_count_and_keeps_the_person(cmds, mem_store):
    mem_store.add_fact("adam", "Adam has a snake")
    mem_store.add_episode(Episode("adam", "hello", "hi"))
    reply, listen = say(cmds, "forget everything about me")
    assert listen and "all 2 of my memories" in reply and "can't undo" in reply
    assert say(cmds, "yes")[0].startswith("Done. I've forgotten all")
    assert mem_store.count_for("adam") == 0 and mem_store.person("adam") is not None


def test_forget_all_with_nothing(cmds):
    assert say(cmds, "forget everything about me")[0] == "I don't have any memories about you."


def test_stop_and_start(cmds, mem_store):
    mem_store.add_fact("adam", "Adam has a snake")
    reply, _ = say(cmds, "stop remembering things about me")
    assert "stopped" in reply and "1 older memories" in reply
    assert mem_store.person("adam")["memory_enabled"] is False
    assert mem_store.add_episode(Episode("adam", "secret", "ok")) is None
    say(cmds, "start remembering things about me again")
    assert mem_store.person("adam")["memory_enabled"] is True
