# Copyright (c) 2026 Adam Wosotowsky
# SPDX-License-Identifier: BUSL-1.1
# Licensed under the Business Source License 1.1; see LICENSE at the
# repository root. Converts to Apache-2.0 on 2030-09-15.
"""Jarvis's systemctl commands follow where homeai.service lives (SANDBOX_PLAN step 3).

Why: after the move to a system unit, a leftover ``systemctl --user restart
homeai`` fails ("Unit not found"), so mode switching and voice-ID admin would
break; and the coder/ComfyUI must stay user units.
"""
from __future__ import annotations

import argparse

import pytest

from homeai import gpu_mode as G
from homeai import service_scope as S
from homeai import speaker_admin as sa


def test_the_system_unit_file_decides(monkeypatch, tmp_path):
    monkeypatch.delenv(S.ENV)
    unit = tmp_path / "homeai.service"
    assert S.scope(unit) == "user"
    unit.write_text("[Service]\n")
    assert S.scope(unit) == "system"


def test_the_env_overrides_the_file(monkeypatch, tmp_path):
    unit = tmp_path / "homeai.service"
    unit.write_text("[Service]\n")
    monkeypatch.setenv(S.ENV, "user")
    assert S.scope(unit) == "user"
    monkeypatch.setenv(S.ENV, "system")
    assert S.scope(tmp_path / "missing") == "system"


def test_a_bad_env_value_falls_back_to_the_file(monkeypatch, tmp_path):
    monkeypatch.setenv(S.ENV, "sytem")
    assert S.scope(tmp_path / "missing") == "user"


def test_only_jarvis_moves_to_the_system_manager(monkeypatch):
    monkeypatch.setenv(S.ENV, "system")
    assert S.systemctl("restart", "homeai") == ["systemctl", "restart", "homeai"]
    assert S.systemctl("is-active", "--quiet", "homeai.service") == ["systemctl", "is-active", "--quiet",
                                                                    "homeai.service"]
    assert S.systemctl("stop", "llama-server.service") == ["systemctl", "--user", "stop", "llama-server.service"]
    assert S.systemctl("daemon-reload") == ["systemctl", "--user", "daemon-reload"]


def test_user_layout_keeps_everything_user(monkeypatch):
    monkeypatch.setenv(S.ENV, "user")
    assert S.systemctl("restart", "homeai") == ["systemctl", "--user", "restart", "homeai"]


def test_jarvis_and_a_user_unit_in_one_call_is_refused(monkeypatch):
    monkeypatch.setenv(S.ENV, "system")
    with pytest.raises(ValueError):
        S.systemctl("stop", "homeai.service", "llama-server.service")


def test_the_log_hint_matches_the_layout(monkeypatch):
    monkeypatch.setenv(S.ENV, "system")
    assert S.journal_hint() == "journalctl -u homeai -n 50"
    monkeypatch.setenv(S.ENV, "user")
    assert S.journal_hint(10) == "journalctl --user -u homeai -n 10"


def test_mode_switching_uses_the_system_manager_for_jarvis(monkeypatch):
    monkeypatch.setenv(S.ENV, "system")
    ran = []
    monkeypatch.setattr(G.subprocess, "run", lambda cmd, **kw: ran.append(cmd) or argparse.Namespace(returncode=0))
    system = G.System()
    assert system.systemctl("stop", G.VOICE_SERVICE) and system.is_active(G.VOICE_SERVICE)
    system.systemctl("stop", G.CODER_SERVICE)
    assert ran == [["systemctl", "stop", G.VOICE_SERVICE],
                   ["systemctl", "is-active", "--quiet", G.VOICE_SERVICE],
                   ["systemctl", "--user", "stop", G.CODER_SERVICE]]


def test_voice_id_admin_restarts_the_system_unit(monkeypatch):
    monkeypatch.setenv(S.ENV, "system")
    ran = []
    assert sa.restart_homeai(lambda cmd, **kw: ran.append(cmd) or argparse.Namespace(returncode=0, stderr="")) \
        == "homeai restarted"
    assert ran == [["systemctl", "restart", "homeai"]]


def test_tests_run_with_the_user_layout_pinned():
    """conftest pins it: once /etc/systemd/system/homeai.service exists, unpinned tests would flip layout."""
    assert S.scope() == "user" and S.os.environ[S.ENV] == "user"
