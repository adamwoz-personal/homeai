# Copyright (c) 2026 Adam Wosotowsky
# SPDX-License-Identifier: BUSL-1.1
# Licensed under the Business Source License 1.1; see LICENSE at the
# repository root. Converts to Apache-2.0 on 2030-09-15.
"""After a picture check, art/voice.py reloads Jarvis's own model, thinking off.

Why: on 2026-11-10 its lazy import still named a constant the switch to
homeai/voice_model.py removed; no test ran it, so the reload would have raised
ImportError on the first check. Ollama holds one model: reloading anything but
Jarvis's leaves the next answer slow.
"""
from __future__ import annotations

from homeai import httpjson
from homeai.art import voice as AV
from tests.conftest import TEST_VOICE_MODEL


def test_the_reload_is_jarvis_s_model_pinned_without_thinking(monkeypatch):
    seen = []
    monkeypatch.setattr(httpjson, "ollama_chat", lambda model, messages, **kw: seen.append((model, kw)) or "")
    AV._warm_voice_model()
    (model, kw), = seen
    assert model == TEST_VOICE_MODEL and kw["keep_alive"] == -1 and kw["think"] is False
