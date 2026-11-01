# Copyright (c) 2026 Adam Wosotowsky
# SPDX-License-Identifier: BUSL-1.1
"""Extraction rules that don't trust the helper (memory_store/extract.py)."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

import pytest

from homeai.memory_store import extract as x

NOW = datetime(2026, 11, 2, 12, tzinfo=timezone.utc)


def note(**kw):
    base = {"kind": "fact", "summary": "Adam keeps a corn snake called Noodle",
            "topics": ["Pets"], "thread": None, "importance": 0.8, "sources": [1]}
    return {**base, **kw}


def parse(*notes, n=3, child=False):
    return x.parse_notes({"notes": list(notes)}, n, is_child=child)


# -- parse_notes ---------------------------------------------------------------------

def test_good_note_is_kept_with_zero_based_sources_and_slugged_topics():
    (n,) = parse(note(sources=[2, 1, 2], topics=["Pets!", "pets", " Reptile Care "],
                      thread="Snake Plans"))
    assert n.sources == (0, 1)
    assert n.topics == ("pets", "reptile care")
    assert n.thread == "snake plans"
    assert n.importance == 0.8


@pytest.mark.parametrize("bad", [
    {"kind": "instruction"}, {"kind": "profile"}, {"kind": "nonsense"},
    {"sources": []}, {"sources": [0]}, {"sources": [4]}, {"sources": ["1"]},
    {"summary": ""}, {"summary": "  a "},
])
def test_doubtful_notes_are_dropped(bad):
    assert parse(note(**bad)) == []


@pytest.mark.parametrize("kind", sorted(x.CHILD_BLOCKED))
def test_children_get_no_opinions_or_thoughts(kind):
    assert parse(note(kind=kind), child=True) == []
    assert len(parse(note(kind=kind), child=False)) == 1


def test_child_facts_are_still_kept():
    assert len(parse(note(kind="preference"), child=True)) == 1


def test_secrets_are_redacted_from_summaries():
    (n,) = parse(note(summary="Adam's wifi password is hunter2hunter2 and key sk-abcdefghijklmnopqrstuvwx"))
    assert "sk-abcdefghijklmnopqrstuvwx" not in n.summary


@pytest.mark.parametrize("raw,want", [(5, 1.0), (-1, 0.0), ("high", 0.5), (None, 0.5)])
def test_importance_is_clamped(raw, want):
    assert parse(note(importance=raw))[0].importance == want


def test_at_most_max_notes_and_topics():
    notes = parse(*[note(summary=f"fact number {i}") for i in range(20)])
    assert len(notes) == x.MAX_NOTES
    (n,) = parse(note(topics=[f"t{i}" for i in range(9)]))
    assert len(n.topics) == x.MAX_TOPICS


# -- grounding: notes must rest on what the person said, not on Jarvis ----------------

MOON = [{"user": "tell me about the moon",
         "assistant": "Last time I saw it clearly I was camping in the mountains."},
        {"user": "I just got a corn snake, his name is Noodle.",
         "assistant": "Corn snakes make great pets. Ball pythons love warm hides."}]


def grounded(summary, sources=(1,), exchanges=MOON):
    return x.parse_notes({"notes": [note(summary=summary, sources=list(sources))]},
                         len(exchanges), is_child=False, exchanges=exchanges, person="Adam")


def test_note_built_from_jarvis_reply_is_dropped():
    assert grounded("Adam remembers camping in the mountains to see the moon") == []


def test_note_paraphrasing_the_person_is_kept():
    assert len(grounded("Adam has a corn snake named Noodle", sources=[2])) == 1


def test_a_word_only_jarvis_used_sinks_the_note_even_a_harmless_one():
    # precision over recall: "pet" came from Jarvis's "pets"
    assert grounded("Adam has a pet corn snake named Noodle", sources=[2]) == []


def test_jarvis_words_from_any_exchange_in_the_batch_count():
    assert grounded("Adam keeps ball pythons", sources=[1]) == []
    assert grounded("Adam keeps ball pythons", sources=[2]) == []


def test_numbers_only_jarvis_said_are_borrowed():
    eps = [{"user": "turn the fan red", "assistant": "Set the fan to 50 percent."}]
    assert grounded("Adam wants the fan at 50", exchanges=eps) == []
    assert x.borrowed("Adam wants the fan at 50", eps) == {"50"}


def test_persons_own_name_and_plurals_are_not_borrowed():
    eps = [{"user": "I bought snakes", "assistant": "Adam, snake care matters."}]
    assert x.borrowed("Adam bought a snake", eps, person="Adam") == set()
    assert x.borrowed("Adam bought a snake", eps) == {"adam"}


def test_without_exchanges_grounding_is_not_checked():
    assert len(parse(note(summary="Adam remembers camping in the mountains"))) == 1


def test_extract_payload_numbers_from_one_and_clips_replies():
    eps = [{"at": NOW, "user": "hi", "assistant": "y" * 900, "trust": "owner"}]
    p = x.extract_payload("Adam", eps, today=NOW, known_topics=["pets"])
    assert p["exchanges"][0]["n"] == 1 and len(p["exchanges"][0]["jarvis_replied"]) == 600
    assert p["known_topics"] == ["pets"] and p["today"] == "2026-11-02"


def test_extract_task_adds_child_rule_only_for_children():
    assert x.CHILD_RULE in x.extract_task(True)
    assert x.CHILD_RULE not in x.extract_task(False)
    assert "never note something only Jarvis said" in x.EXTRACT_TASK


@pytest.mark.parametrize("text,want", [
    ("D&D Campaign", "d&d campaign"), ("  --hi--  ", "hi"), ("!!!", None), (None, None),
    ("x" * 60, "x" * 48),
])
def test_slug(text, want):
    assert x.slug(text) == want


# -- triage / reconcile -----------------------------------------------------------------

def test_triage():
    assert x.triage([]).action == "add"
    assert x.triage([0.3, 0.64]).action == "add"
    d = x.triage([0.5, 0.95])
    assert d.action == "noop" and d.target == 1
    assert x.triage([0.7, 0.8]) is None          # related: ask the helper


@pytest.mark.parametrize("target,trust,want", [
    ({"pinned": True, "source": "extractor", "trust": "owner"}, "owner", True),
    ({"pinned": False, "source": "explicit", "trust": "owner"}, "owner", True),
    ({"pinned": False, "source": "import", "trust": "owner"}, "owner", True),
    ({"pinned": False, "source": "reflection", "trust": "owner"}, "owner", True),
    ({"pinned": False, "source": "extractor", "trust": "owner"}, "owner", False),
    ({"pinned": False, "source": "extractor", "trust": "owner"}, "outside", True),
    ({"pinned": False, "source": "extractor", "trust": "outside"}, "outside", False),
])
def test_protected(target, trust, want):
    assert x.protected(target, trust) is want


N = x.Note("fact", "Adam's snake is called Pickle", ("pets",), None, 0.5, (0,))
NB = [{"summary": "Adam's snake is called Noodle", "pinned": False, "source": "extractor",
       "trust": "owner", "created_at": NOW - timedelta(days=3)},
      {"summary": "Adam likes snakes", "pinned": True, "source": "extractor",
       "trust": "owner", "created_at": NOW}]


@pytest.mark.parametrize("raw,want", [
    ({"action": "noop", "target": 0, "summary": ""}, ("noop", 0)),
    ({"action": "noop", "target": None, "summary": ""}, ("skip", None)),
    ({"action": "supersede", "target": 0, "summary": "Adam's snake is Pickle"}, ("supersede", 0)),
    ({"action": "update", "target": 0, "summary": "merged"}, ("update", 0)),
    ({"action": "update", "target": 1, "summary": "merged"}, ("add", None)),     # pinned
    ({"action": "supersede", "target": 7, "summary": "x"}, ("add", None)),       # bad target
    ({"action": "add", "target": None, "summary": "new one"}, ("add", None)),
])
def test_parse_decision(raw, want):
    d = x.parse_decision(raw, N, NB, "owner")
    assert (d.action, d.target) == want


def test_rejected_overwrite_keeps_the_notes_own_words():
    d = x.parse_decision({"action": "update", "target": 1, "summary": "rewritten"}, N, NB, "owner")
    assert d.summary == N.summary


def test_outside_content_cannot_overwrite_owner_facts():
    d = x.parse_decision({"action": "supersede", "target": 0, "summary": "x y z"}, N, NB, "outside")
    assert d.action == "add"


def test_reconcile_payload_is_zero_based():
    p = x.reconcile_payload(N, NB)
    assert [e["n"] for e in p["existing"]] == [0, 1] and p["new"] == N.summary


# -- reflection --------------------------------------------------------------------------

def fact(i, **kw):
    base = {"id": i, "kind": "fact", "summary": f"fact {i}", "topics": [], "trust": "owner",
            "source": "extractor", "importance": 0.5, "updated_at": NOW}
    return {**base, **kw}


def test_reflection_inputs_exclude_outside_profiles_and_instructions():
    facts = [fact(1), fact(2, trust="outside"), fact(3, source="reflection", kind="profile"),
             fact(4, kind="instruction"), fact(5, importance=0.9), fact(6, kind="profile",
                                                                        source="explicit")]
    assert [f["id"] for f in x.reflection_inputs(facts)] == [5, 1]


def test_reflection_inputs_limit():
    assert len(x.reflection_inputs([fact(i) for i in range(80)], limit=60)) == 60


def test_parse_reflection():
    raw = {"lines": [{"summary": "Adam keeps snakes", "sources": [1, 2, 9]},
                     {"summary": "Adam builds things", "sources": [0]},
                     {"summary": "x", "sources": [1]}]}
    assert x.parse_reflection(raw, 3) == [("Adam keeps snakes", (0, 1))]


def test_reflect_payload():
    p = x.reflect_payload("Adam", [fact(1, topics=["pets"])])
    assert p["notes"] == [{"n": 1, "kind": "fact", "note": "fact 1", "topics": ["pets"]}]
