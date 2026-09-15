# Copyright (c) 2026 Adam Wosotowsky
# SPDX-License-Identifier: BUSL-1.1
# Licensed under the Business Source License 1.1; see LICENSE at the
# repository root. Converts to Apache-2.0 on 2030-09-15.
"""Tests for the CLI agent transport.

No test here spawns a real process; the runner is injected.
"""

from __future__ import annotations

import subprocess

import pytest

from homeai.agent_cli import CliAgentClient, build_agent_client
from homeai.config import AgentConfig


class FakeProc:
    def __init__(self, returncode=0, stdout="", stderr=""):
        self.returncode = returncode
        self.stdout = stdout
        self.stderr = stderr


def make_runner(*results):
    """Runner returning each result in turn; raises if called too often."""
    calls = []
    seq = list(results)

    def runner(cmd, **kwargs):
        calls.append((cmd, kwargs))
        if not seq:
            raise AssertionError("runner called more times than expected")
        item = seq.pop(0)
        if isinstance(item, Exception):
            raise item
        return item

    runner.calls = calls
    return runner


def cfg(**kw):
    base = AgentConfig()
    for k, v in kw.items():
        object.__setattr__(base, k, v)
    return base


# -- happy path ------------------------------------------------------------


def test_successful_reply_is_returned_trimmed():
    runner = make_runner(FakeProc(0, "  The capital of France is Paris.\n"))
    reply = CliAgentClient(cfg(), runner=runner).ask("capital of france")
    assert reply.ok
    assert reply.text == "The capital of France is Paris."
    assert reply.attempts == 1


def test_command_targets_configured_agent_and_binary():
    runner = make_runner(FakeProc(0, "hi"))
    c = cfg(agent_name="local", cli_binary="/usr/bin/zeroclaw")
    CliAgentClient(c, runner=runner).ask("hello")
    cmd = runner.calls[0][0]
    assert cmd[0] == "/usr/bin/zeroclaw"
    assert "-a" in cmd and cmd[cmd.index("-a") + 1] == "local"
    assert cmd[-2] == "-m" and cmd[-1] == "hello"


def test_timeout_is_passed_to_runner():
    runner = make_runner(FakeProc(0, "hi"))
    CliAgentClient(cfg(timeout_s=12.0), runner=runner).ask("hello")
    assert runner.calls[0][1]["timeout"] == 12.0


# -- session isolation (guards against model self-poisoning) ---------------


def test_every_attempt_uses_a_distinct_session_state_file():
    runner = make_runner(FakeProc(0, ""), FakeProc(0, ""), FakeProc(0, "ok"))
    CliAgentClient(cfg(max_retries=2), runner=runner).ask("hello")
    files = [c[0][c[0].index("--session-state-file") + 1] for c in runner.calls]
    assert len(files) == 3
    assert len(set(files)) == 3, "session state files must never be reused"


def test_state_file_is_removed_after_each_attempt(tmp_path):
    seen = {}

    def runner(cmd, **kwargs):
        path = cmd[cmd.index("--session-state-file") + 1]
        open(path, "w").write("{}")
        seen["path"] = path
        return FakeProc(0, "done")

    CliAgentClient(cfg(), runner=runner).ask("hello")
    import os

    assert not os.path.exists(seen["path"]), "state file must not linger"


# -- failure handling ------------------------------------------------------


def test_empty_reply_retries_then_fails():
    runner = make_runner(FakeProc(0, ""), FakeProc(0, "   "))
    reply = CliAgentClient(cfg(max_retries=1), runner=runner).ask("hello")
    assert not reply.ok
    assert "empty" in reply.error
    assert reply.attempts == 2


def test_empty_reply_then_success_recovers():
    runner = make_runner(FakeProc(0, ""), FakeProc(0, "Paris."))
    reply = CliAgentClient(cfg(max_retries=1), runner=runner).ask("hello")
    assert reply.ok and reply.text == "Paris."
    assert reply.attempts == 2


def test_timeout_is_reported_not_raised():
    runner = make_runner(
        subprocess.TimeoutExpired(cmd="zeroclaw", timeout=5),
        subprocess.TimeoutExpired(cmd="zeroclaw", timeout=5),
    )
    reply = CliAgentClient(cfg(max_retries=1), runner=runner).ask("hello")
    assert not reply.ok
    assert "timed out" in reply.error


def test_missing_binary_fails_fast_without_retrying():
    runner = make_runner(FileNotFoundError("nope"))
    reply = CliAgentClient(cfg(max_retries=3), runner=runner).ask("hello")
    assert not reply.ok
    assert "not found" in reply.error
    assert reply.attempts == 1, "misconfiguration must not be retried"


def test_nonzero_exit_is_reported():
    runner = make_runner(FakeProc(1, "", "boom"), FakeProc(1, "", "boom"))
    reply = CliAgentClient(cfg(max_retries=1), runner=runner).ask("hello")
    assert not reply.ok
    assert "exited 1" in reply.error


def test_unexpected_exception_does_not_escape():
    runner = make_runner(RuntimeError("weird"), FakeProc(0, "recovered"))
    reply = CliAgentClient(cfg(max_retries=1), runner=runner).ask("hello")
    assert reply.ok and reply.text == "recovered"


def test_empty_utterance_is_rejected_without_spawning():
    runner = make_runner()
    reply = CliAgentClient(cfg(), runner=runner).ask("   ")
    assert not reply.ok
    assert runner.calls == [], "must not spawn a process for an empty utterance"


# -- safety ----------------------------------------------------------------


def test_leaked_tool_markup_is_never_returned():
    leaked = '{"name": "shell_tool", "arguments": {"command": "rm -rf /"}}'
    runner = make_runner(FakeProc(0, leaked), FakeProc(0, leaked))
    reply = CliAgentClient(cfg(max_retries=1), runner=runner).ask("hello")
    assert not reply.ok
    assert reply.leaked
    assert "rm -rf" not in reply.text


def test_leaked_markup_then_clean_reply_recovers():
    leaked = "<tool_call>\n{\"name\": \"shell_tool\"}\n</tool_call>"
    runner = make_runner(FakeProc(0, leaked), FakeProc(0, "Paris."))
    reply = CliAgentClient(cfg(max_retries=1), runner=runner).ask("hello")
    assert reply.ok and reply.text == "Paris."


# -- factory ---------------------------------------------------------------


def test_factory_defaults_to_cli_transport():
    assert isinstance(build_agent_client(cfg()), CliAgentClient)


def test_factory_honours_explicit_http_transport():
    from homeai.agent_client import AgentClient

    assert isinstance(build_agent_client(cfg(transport="http")), AgentClient)


def test_default_transport_is_cli_for_security():
    # Regression guard: the HTTP gateway does not enforce risk profiles.
    assert AgentConfig().transport == "cli"


def test_cli_and_http_clients_expose_the_same_interface():
    """Both transports are duck-typed by the daemon; drift breaks startup."""
    from homeai.agent_client import AgentClient

    required = {"ask", "health"}
    for cls in (CliAgentClient, AgentClient):
        missing = required - {m for m in dir(cls) if not m.startswith("_")}
        assert not missing, f"{cls.__name__} is missing {missing}"
