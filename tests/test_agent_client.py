# Copyright (c) 2026 Adam Wosotowsky
# SPDX-License-Identifier: BUSL-1.1
# Licensed under the Business Source License 1.1; see LICENSE at the
# repository root. Converts to Apache-2.0 on 2030-09-15.
"""Tests for the ZeroClaw gateway client.

All network interaction is mocked. These tests exist to prove the failure
paths behave, because those are what keep the voice service alive.
"""

from __future__ import annotations

import json
from dataclasses import replace

import pytest
import requests

from homeai.agent_client import AgentClient, AgentReply
from homeai.config import AgentConfig


class FakeResponse:
    def __init__(self, status_code: int = 200, payload: object = None, text: str = "") -> None:
        self.status_code = status_code
        self._payload = payload
        self.text = text

    def json(self) -> object:
        if self._payload is None:
            raise ValueError("no json")
        return self._payload


class FakeSession:
    """Replays a scripted list of responses or exceptions."""

    def __init__(self, responses: list) -> None:
        self._responses = list(responses)
        self.calls: list[dict] = []

    def post(self, url, json=None, headers=None, timeout=None):  # noqa: A002
        self.calls.append({"url": url, "json": json, "headers": headers})
        if not self._responses:
            raise AssertionError("more POSTs than scripted responses")
        item = self._responses.pop(0)
        if isinstance(item, Exception):
            raise item
        return item

    def get(self, url, timeout=None):
        if not self._responses:
            raise AssertionError("no scripted response for GET")
        item = self._responses.pop(0)
        if isinstance(item, Exception):
            raise item
        return item


@pytest.fixture
def cfg() -> AgentConfig:
    return replace(AgentConfig(), token="test-token", max_retries=2, timeout_s=1.0)


class TestSuccess:
    def test_returns_reply_text(self, cfg: AgentConfig) -> None:
        session = FakeSession([FakeResponse(200, {"reply": "It is sunny."})])
        result = AgentClient(cfg, session).ask("what is the weather")
        assert result.ok is True
        assert result.text == "It is sunny."
        assert result.attempts == 1

    @pytest.mark.parametrize(
        "key", ["reply", "response", "message", "text", "content", "output", "result"]
    )
    def test_accepts_alternative_response_keys(self, cfg: AgentConfig, key: str) -> None:
        session = FakeSession([FakeResponse(200, {key: "hello"})])
        assert AgentClient(cfg, session).ask("hi").text == "hello"

    def test_accepts_nested_payload(self, cfg: AgentConfig) -> None:
        session = FakeSession([FakeResponse(200, {"result": {"text": "nested reply"}})])
        assert AgentClient(cfg, session).ask("hi").text == "nested reply"

    def test_sends_bearer_token(self, cfg: AgentConfig) -> None:
        session = FakeSession([FakeResponse(200, {"reply": "ok"})])
        AgentClient(cfg, session).ask("hi")
        assert session.calls[0]["headers"]["Authorization"] == "Bearer test-token"

    def test_uses_fresh_session_id_each_attempt(self, cfg: AgentConfig) -> None:
        """Session reuse is what makes markup poisoning permanent."""
        session = FakeSession(
            [
                FakeResponse(200, {"reply": "<function=shell>"}),
                FakeResponse(200, {"reply": "clean answer"}),
            ]
        )
        result = AgentClient(cfg, session).ask("hi")
        assert result.ok is True
        ids = [call["json"]["session_id"] for call in session.calls]
        assert len(ids) == 2
        assert ids[0] != ids[1], "each attempt must use a new session id"


class TestLeakedMarkup:
    def test_retries_then_succeeds(self, cfg: AgentConfig) -> None:
        session = FakeSession(
            [
                FakeResponse(200, {"reply": "<function=file_write>"}),
                FakeResponse(200, {"reply": "the real answer"}),
            ]
        )
        result = AgentClient(cfg, session).ask("hi")
        assert result.ok is True
        assert result.text == "the real answer"
        assert result.attempts == 2

    def test_never_returns_leaked_text(self, cfg: AgentConfig) -> None:
        """Even when every attempt leaks, the markup must not escape."""
        session = FakeSession([FakeResponse(200, {"reply": "</tool_call>"}) for _ in range(3)])
        result = AgentClient(cfg, session).ask("hi")
        assert result.ok is False
        assert "tool_call" not in result.text
        assert result.text == ""
        assert result.leaked is True


class TestFailurePaths:
    def test_401_does_not_retry(self, cfg: AgentConfig) -> None:
        session = FakeSession([FakeResponse(401)])
        result = AgentClient(cfg, session).ask("hi")
        assert result.ok is False
        assert "401" in result.error
        assert len(session.calls) == 1, "bad credentials must not be retried"

    def test_timeout_is_reported_not_raised(self, cfg: AgentConfig) -> None:
        session = FakeSession([requests.Timeout() for _ in range(3)])
        result = AgentClient(cfg, session).ask("hi")
        assert result.ok is False
        assert "timed out" in result.error

    def test_connection_error_is_reported_not_raised(self, cfg: AgentConfig) -> None:
        session = FakeSession([requests.ConnectionError("refused") for _ in range(3)])
        result = AgentClient(cfg, session).ask("hi")
        assert result.ok is False
        assert "unreachable" in result.error

    def test_server_error_retries(self, cfg: AgentConfig) -> None:
        session = FakeSession([FakeResponse(500), FakeResponse(200, {"reply": "recovered"})])
        result = AgentClient(cfg, session).ask("hi")
        assert result.ok is True
        assert result.text == "recovered"

    def test_client_error_does_not_retry(self, cfg: AgentConfig) -> None:
        session = FakeSession([FakeResponse(404)])
        result = AgentClient(cfg, session).ask("hi")
        assert result.ok is False
        assert len(session.calls) == 1

    def test_empty_reply_retried_then_fails(self, cfg: AgentConfig) -> None:
        session = FakeSession([FakeResponse(200, {"reply": "   "}) for _ in range(3)])
        result = AgentClient(cfg, session).ask("hi")
        assert result.ok is False
        assert "empty" in result.error

    def test_empty_utterance_short_circuits(self, cfg: AgentConfig) -> None:
        session = FakeSession([])
        result = AgentClient(cfg, session).ask("")
        assert result.ok is False
        assert session.calls == [], "must not contact the gateway with nothing to say"

    def test_non_json_response_handled(self, cfg: AgentConfig) -> None:
        session = FakeSession([FakeResponse(200, None, text="plain text reply")])
        result = AgentClient(cfg, session).ask("hi")
        assert result.ok is True
        assert result.text == "plain text reply"


class TestHealth:
    def test_health_ok(self, cfg: AgentConfig) -> None:
        assert AgentClient(cfg, FakeSession([FakeResponse(200)])).health() is True

    def test_health_failure_is_false_not_raise(self, cfg: AgentConfig) -> None:
        session = FakeSession([requests.ConnectionError("down")])
        assert AgentClient(cfg, session).health() is False
