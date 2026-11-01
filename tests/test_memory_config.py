# Copyright (c) 2026 Adam Wosotowsky
# SPDX-License-Identifier: BUSL-1.1
"""MemoryConfig and how the daemon wires long-term memory in (or doesn't)."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from homeai.config import Config, MemoryConfig


def test_defaults(monkeypatch):
    for k in ("HOMEAI_LTM", "HOMEAI_LTM_DSN", "HOMEAI_LTM_IGNORE_SPEAKERS", "HOMEAI_LTM_ADMINS"):
        monkeypatch.delenv(k, raising=False)
    m = MemoryConfig()
    assert m.enabled and m.dsn == "dbname=homeai"
    assert m.recall_timeout_ms == 150 and m.budget_tokens == 600
    assert m.min_similarity == pytest.approx(0.60)
    assert m.ignore_speakers == ("ryan", "amy", "joe", "alba", "hfc")
    assert m.admins == ("adam",)


def test_env_overrides(monkeypatch):
    monkeypatch.setenv("HOMEAI_LTM", "0")
    monkeypatch.setenv("HOMEAI_LTM_IGNORE_SPEAKERS", " Ryan , ,Bob")
    monkeypatch.setenv("HOMEAI_LTM_ADMINS", "Adam,Sue")
    monkeypatch.setenv("HOMEAI_LTM_MIN_SIM", "0.7")
    m = MemoryConfig()
    assert not m.enabled
    assert m.ignore_speakers == ("ryan", "bob")
    assert m.admins == ("adam", "sue")
    assert m.min_similarity == pytest.approx(0.7)


def test_tests_never_point_at_the_real_database():
    m = Config().memory
    assert not m.enabled and m.dsn != "dbname=homeai"


def _assistant(cfg):
    with patch("homeai.daemon.Microphone"), patch("homeai.daemon.Transcriber"), \
         patch("homeai.daemon.Speaker"), patch("homeai.daemon.build_agent_client"), \
         patch("homeai.daemon.TranscriptLog") as tlog:
        from homeai.daemon import VoiceAssistant

        return VoiceAssistant(cfg), tlog


def test_daemon_without_ltm():
    va, tlog = _assistant(Config())
    assert va.ltm is None and va.memory._long_term is None
    tlog.return_value.add_listener.assert_not_called()


def test_daemon_wires_ltm_when_enabled(monkeypatch):
    monkeypatch.setenv("HOMEAI_LTM", "1")
    svc = MagicMock()
    with patch("homeai.memory_store.service.MemoryService") as cls:
        cls.return_value.start.return_value = svc
        va, tlog = _assistant(Config())
    assert va.ltm is svc
    assert va.memory._long_term == svc.recall
    tlog.return_value.add_listener.assert_called_once_with(svc.on_turn)
    va._threads = []
    va.stop()
    svc.close.assert_called_once()


def test_daemon_survives_ltm_failing_to_build(monkeypatch, caplog):
    monkeypatch.setenv("HOMEAI_LTM", "1")
    with patch("homeai.memory_store.service.MemoryService", side_effect=ImportError("no psycopg")):
        with caplog.at_level("WARNING"):
            va, _ = _assistant(Config())
    assert va.ltm is None and va.memory._long_term is None
    assert any("long-term memory disabled" in r.message for r in caplog.records)
