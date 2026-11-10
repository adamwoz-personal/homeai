# Copyright (c) 2026 Adam Wosotowsky
# SPDX-License-Identifier: BUSL-1.1
# Licensed under the Business Source License 1.1; see LICENSE at the
# repository root. Converts to Apache-2.0 on 2030-09-15.
"""tools/eval_tool_misfire.py scores tool calls right and never leaves Ollama (LLM_UPGRADE_PLAN step 1).

Why: the model choice rests on these counts, and the eval must never run a
tool: it may only talk to Ollama's /api/chat and /api/ps.
"""
from __future__ import annotations

import importlib.util
import io
import urllib.error
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("eval_tool_misfire", REPO / "tools" / "eval_tool_misfire.py")
E = importlib.util.module_from_spec(spec)
spec.loader.exec_module(E)

CAPTURE = {"path": "/v1/chat/completions",
           "body": {"model": "llama31-voice", "temperature": 0.45, "tool_choice": "auto",
                    "tools": [{"type": "function", "function": {"name": "homeai__weather"}}],
                    "messages": [{"role": "system", "content": "You are Jarvis."},
                                 {"role": "user", "content": "What is two plus two?"}]}}


class FakeOllama:
    """Weather questions get the weather tool, 'yourself' gets a misfire; one model rejects think."""

    def __init__(self, no_think=()):
        self.calls, self.no_think = [], set(no_think)

    def __call__(self, url, payload=None, timeout=30.0):
        self.calls.append((url, payload))
        if url.endswith("/api/ps"):
            return {"models": [{"name": "m:latest", "size": 100, "size_vram": 80}]}
        if "think" in payload and payload["model"] in self.no_think:
            raise urllib.error.HTTPError(url, 400, "does not support thinking", {}, io.BytesIO())
        q = payload["messages"][-1]["content"]
        if "weather" in q or "yourself" in q:
            return {"message": {"content": "", "tool_calls": [{"function": {"name": "homeai__weather"}}]}}
        return {"message": {"content": "A short spoken answer here."}}


@pytest.mark.parametrize("called, wanted, verdict", [
    (None, E.NONE, "ok"), ("homeai__research", E.NONE, "misfire"), (None, frozenset({"homeai__door"}), "miss"),
    ("homeai__timer", frozenset({"homeai__door"}), "wrong"), ("calculator", E.CALC_OK, "ok"), (None, E.CALC_OK, "ok"),
])
def test_classify(called, wanted, verdict):
    assert E.classify(called, wanted) == verdict


def test_the_request_is_jarvis_s_prompt_and_tools_with_the_question_swapped():
    req = E.request_for(CAPTURE, "qwen3.5:9b", "Tell me a joke.", False)
    assert req["model"] == "qwen3.5:9b" and req["tools"] == CAPTURE["body"]["tools"]
    assert req["messages"] == [{"role": "system", "content": "You are Jarvis."},
                               {"role": "user", "content": "Tell me a joke."}]
    assert req["think"] is False and req["options"] == {"temperature": 0.45, "num_ctx": 16384}
    assert "think" not in E.request_for(CAPTURE, "m", "q", None)


def test_a_model_without_thinking_is_asked_again_without_the_flag(tmp_path):
    fake = FakeOllama(no_think={"old"})
    state = {}
    answer = E.ask(CAPTURE, "old", "Tell me a joke.", state, fake)
    assert answer["text"] and state["old"] is None
    assert "think" in fake.calls[0][1] and "think" not in fake.calls[1][1]
    E.ask(CAPTURE, "old", "Tell me a joke.", state, fake)
    assert len(fake.calls) == 3   # remembered: no second rejected try


def test_a_run_scores_each_model_and_only_talks_to_ollama(tmp_path, monkeypatch):
    monkeypatch.setattr(E, "QUESTIONS", [("Tell me about yourself.", E.NONE),
                                         ("What's the weather like right now?", frozenset({"homeai__weather"})),
                                         ("Who was at the front door?", frozenset({"homeai__door"}))])
    fake = FakeOllama()
    data = E.run(CAPTURE, ["m"], 2, tmp_path, fake, log=lambda line: None)
    s = data["models"]["m"]["summary"]
    assert (s["ok"], s["misfire"], s["miss"], s["wrong"], s["n"]) == (2, 2, 2, 0, 6)
    assert s["median_words"] == 5 and data["models"]["m"]["gpu_percent"] == 80.0
    assert (tmp_path / "result.json").exists()
    assert {url for url, _ in fake.calls} == {f"{E.OLLAMA}/api/chat", f"{E.OLLAMA}/api/ps"}
    assert all(p["think"] is False for url, p in fake.calls if url.endswith("/api/chat"))   # spoken answers, no thinking
    assert "| m | 2/6 | 2 | 2 | 0 | 5.0 |" in E.table(data)
    assert s["median_reasoning"] == 0


class FakeOpenAI:
    def __init__(self):
        self.calls = []

    def __call__(self, url, payload=None, timeout=30.0):
        self.calls.append((url, payload))
        if url.endswith("/api/ps"):
            return {"models": []}
        q = payload["messages"][-1]["content"]
        if "weather" in q:
            return {"choices": [{"message": {"content": None,
                                             "tool_calls": [{"function": {"name": "homeai__weather"}}]}}]}
        return {"choices": [{"message": {"content": "Paris.", "reasoning": "" if payload.get("think") is False
                                         else "Let me think."}}]}


def test_the_openai_route_sends_the_captured_body_as_zeroclaw_does(tmp_path, monkeypatch):
    monkeypatch.setattr(E, "QUESTIONS", [("What's the capital of France?", E.NONE),
                                         ("What's the weather like right now?", frozenset({"homeai__weather"}))])
    fake = FakeOpenAI()
    data = E.run(CAPTURE, ["qwen3.5:9b"], 1, tmp_path, fake, log=lambda line: None, endpoint="openai")
    chats = [p for url, p in fake.calls if url == f"{E.OLLAMA}/v1/chat/completions"]
    assert len(chats) == 2 and data["models"]["qwen3.5:9b"]["summary"]["ok"] == 2
    body = chats[0]
    assert body["model"] == "qwen3.5:9b" and body["stream"] is False and "think" not in body
    assert body["tool_choice"] == "auto" and body["temperature"] == 0.45 and body["tools"] == CAPTURE["body"]["tools"]
    assert body["messages"] == [{"role": "system", "content": "You are Jarvis."},
                                {"role": "user", "content": "What's the capital of France?"}]


def test_extra_fields_reach_the_request_and_thinking_is_measured(tmp_path, monkeypatch):
    monkeypatch.setattr(E, "QUESTIONS", [("What's the capital of France?", E.NONE)])
    fake = FakeOpenAI()
    plain = E.run(CAPTURE, ["m"], 2, tmp_path, fake, log=lambda line: None, endpoint="openai")
    off = E.run(CAPTURE, ["m"], 2, tmp_path, fake, log=lambda line: None, endpoint="openai", extra={"think": False})
    assert plain["models"]["m"]["summary"]["median_reasoning"] == len("Let me think.")
    assert off["models"]["m"]["summary"]["median_reasoning"] == 0 and off["extra"] == {"think": False}
    assert [p.get("think") for url, p in fake.calls if p] == [None, None, False, False]
    assert E.table(plain).splitlines()[-1].endswith("| 13.0 |")


@pytest.mark.parametrize("message, args", [
    ({"tool_calls": [{"function": {"name": "homeai__weather", "arguments": '{"day": "tomorrow"}'}}]}, {"day": "tomorrow"}),
    ({"tool_calls": [{"function": {"name": "homeai__weather", "arguments": {"day": "saturday"}}}]}, {"day": "saturday"}),
    ({"tool_calls": [{"function": {"name": "x", "arguments": "not json"}}]}, {}),
    ({"content": "Paris."}, {}),
])
def test_tool_arguments_are_recorded(message, args):
    assert E.first_args(message) == args
