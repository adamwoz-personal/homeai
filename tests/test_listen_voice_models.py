# Copyright (c) 2026 Adam Wosotowsky
# SPDX-License-Identifier: BUSL-1.1
# Licensed under the Business Source License 1.1; see LICENSE at the
# repository root. Converts to Apache-2.0 on 2030-09-15.
"""tools/listen_voice_models.py: blind, Ollama-only, and says what Jarvis would say (LLM_UPGRADE_PLAN step 2).

Why: Adam picks the voice model by ear from this. The test must stay blind
(labels shuffled, the key kept out of blind.json), must never reach anything
but Ollama (no tool runs), must carry the conversation history, and must speak
each answer through the daemon's own speech chain and budget.
"""
from __future__ import annotations

import contextlib
import importlib.util
import json
import random
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("listen_voice_models", REPO / "tools" / "listen_voice_models.py")
L = importlib.util.module_from_spec(spec)
spec.loader.exec_module(L)

CAPTURE = {"body": {"temperature": 0.45, "tool_choice": "auto",
                    "tools": [{"type": "function", "function": {"name": "homeai__research"}}],
                    "messages": [{"role": "system", "content": "You are Jarvis."},
                                 {"role": "user", "content": "captured question"}]}}


class FakeOllama:
    """Model 'quiet' answers briefly; 'chatty' answers at length; 'looker' calls research on 'joke'."""

    def __init__(self):
        self.calls = []

    def __call__(self, url, payload=None, timeout=30.0):
        self.calls.append((url, payload))
        q = payload["messages"][-1]["content"]
        if payload["model"] == "looker" and "joke" in q:
            return {"choices": [{"message": {"content": None,
                                             "tool_calls": [{"function": {"name": "homeai__research"}}]}}]}
        n = 3 if payload["model"] != "chatty" else 60
        text = " ".join(f"Sentence number {i} is about {q.split()[0].lower()} things." for i in range(n))
        return {"choices": [{"message": {"content": f"**{payload['model']}** {text}", "reasoning": "hmm"}}]}


@pytest.fixture
def two_convs(monkeypatch):
    monkeypatch.setattr(L, "CONVERSATIONS", [("A joke", ["Tell me a joke."]),
                                             ("Eggs", ["How long for a soft egg?", "And hard?"])])


def test_generate_is_blind_and_only_talks_to_ollama(tmp_path, two_convs):
    fake = FakeOllama()
    L.generate(CAPTURE, ["quiet", "chatty"], tmp_path, 110, fake, random.Random(1), log=lambda line: None)
    assert {url for url, _ in fake.calls} == {f"{L.OLLAMA}/v1/chat/completions"}
    blind = (tmp_path / "blind.json").read_text()
    for item in json.loads(blind):
        assert set(item) == {"title", "A", "B"}
    key = json.loads((tmp_path / "key.json").read_text())["key"]
    assert all(sorted(k.values()) == ["chatty", "quiet"] for k in key)


def test_labels_are_shuffled_per_conversation(tmp_path, monkeypatch):
    monkeypatch.setattr(L, "CONVERSATIONS", [(f"c{i}", ["Hi."]) for i in range(12)])
    data = L.generate(CAPTURE, ["quiet", "chatty"], tmp_path, 110, FakeOllama(), random.Random(3),
                      log=lambda line: None)
    assert {k["A"] for k in data["key"]} == {"quiet", "chatty"}


def test_follow_ups_carry_the_history_and_the_real_request(tmp_path, two_convs):
    fake = FakeOllama()
    L.generate(CAPTURE, ["quiet", "chatty"], tmp_path, 110, fake, random.Random(1), log=lambda line: None)
    body = [p for _, p in fake.calls if p["model"] == "quiet" and p["messages"][-1]["content"] == "And hard?"][0]
    roles = [m["role"] for m in body["messages"]]
    assert roles == ["system", "user", "assistant", "user"]           # captured user turn dropped, history kept
    assert body["messages"][2]["content"].startswith("**quiet**")
    assert body["tools"] == CAPTURE["body"]["tools"] and body["tool_choice"] == "auto"
    assert body["temperature"] == 0.45 and body["stream"] is False and "think" not in body


def test_a_tool_call_is_heard_as_a_lookup_and_never_run(tmp_path, two_convs):
    data = L.generate(CAPTURE, ["looker", "quiet"], tmp_path, 110, FakeOllama(), random.Random(1),
                      log=lambda line: None)
    joke = data["blind"][0]
    looked = [joke[x][0] for x in "AB" if joke[x][0]["tool"]]
    assert len(looked) == 1 and looked[0]["spoken"] == L.LOOKUP


def test_answers_are_spoken_the_way_the_daemon_speaks_them(tmp_path, two_convs):
    data = L.generate(CAPTURE, ["quiet", "chatty"], tmp_path, 110, FakeOllama(), random.Random(1),
                      log=lambda line: None)
    answers = json.loads((tmp_path / "key.json").read_text())["answers"]
    quiet, chatty = answers["quiet"][0][0], answers["chatty"][0][0]
    assert "**" not in quiet["spoken"] and quiet["held_words"] == 0
    assert chatty["spoken"].endswith(L.CONTINUE_PROMPT) and chatty["held_words"] > 0
    assert len(chatty["spoken"].split()) < 130 and chatty["words"] > 400
    assert quiet["reasoning_chars"] == len("hmm")
    assert data["blind"]


def test_generate_wants_exactly_two_models(tmp_path):
    with pytest.raises(ValueError):
        L.generate(CAPTURE, ["one"], tmp_path, 110, FakeOllama())


def test_play_speaks_a_s_conversation_then_b_s_with_jarvis_stopped(tmp_path, two_convs):
    L.generate(CAPTURE, ["quiet", "chatty"], tmp_path, 110, FakeOllama(), random.Random(1), log=lambda line: None)
    said, events = [], []

    @contextlib.contextmanager
    def pause():
        events.append("stop")
        yield
        events.append("start")

    def say(line):
        events.append("say")
        said.append(line)
    assert L.play(tmp_path, 2, say, pause, gap=lambda s: None) == 1
    assert events[0] == "stop" and events[-1] == "start" and events.count("say") == len(said)
    assert said[0] == "Conversation 2: Eggs." and said[1] == "Answer A."
    assert said.index("Answer B.") > said.index("You asked: And hard?")
    with pytest.raises(SystemExit):
        L.play(tmp_path, 9, say, pause, gap=lambda s: None)


def test_jarvis_comes_back_when_playing_fails(tmp_path, two_convs):
    L.generate(CAPTURE, ["quiet", "chatty"], tmp_path, 110, FakeOllama(), random.Random(1), log=lambda line: None)
    events = []

    @contextlib.contextmanager
    def pause():
        events.append("stop")
        try:
            yield
        finally:
            events.append("start")

    def say(line):
        raise SystemExit("speech failed")
    with pytest.raises(SystemExit):
        L.play(tmp_path, None, say, pause, gap=lambda s: None)
    assert events == ["stop", "start"]


def test_reveal_names_both_models(tmp_path, two_convs):
    L.generate(CAPTURE, ["quiet", "chatty"], tmp_path, 110, FakeOllama(), random.Random(1), log=lambda line: None)
    text = L.reveal(tmp_path)
    assert text.count("quiet") == 2 and text.count("chatty") == 2 and text.startswith("A joke: A = ")


def test_jarvis_is_really_stopped_and_started(monkeypatch):
    monkeypatch.setenv("HOMEAI_SERVICE_SCOPE", "system")
    ran = []
    with L._jarvis_stopped(lambda argv, check: ran.append((argv, check)))():
        assert ran == [(["systemctl", "stop", "homeai.service"], True)]
    assert ran[-1] == (["systemctl", "start", "homeai.service"], True)


def test_a_markdown_list_is_spoken_as_sentences():
    spoken, held = L.spoken_form("## Soft egg\n- Boil six minutes\n- Cool in water", [], 110)
    assert "#" not in spoken and "- " not in spoken and held == 0
    assert "Soft egg." in spoken and "Boil six minutes." in spoken
