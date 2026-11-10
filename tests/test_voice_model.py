# Copyright (c) 2026 Adam Wosotowsky
# SPDX-License-Identifier: BUSL-1.1
# Licensed under the Business Source License 1.1; see LICENSE at the
# repository root. Converts to Apache-2.0 on 2030-09-15.
"""homeai/voice_model.py reads the voice model from ZeroClaw's config, the one place it is set.

Why: six hard-coded copies could disagree, and with one model held by Ollama a
mismatch evicts Jarvis's model. The reader must find the agent by its house
tools bundle (not by name), follow its provider, and never crash a caller.
"""
from __future__ import annotations

import pytest

from homeai import voice_model as V
from tests.conftest import TEST_VOICE_MODEL

LIVE_SHAPE = """
[agents.local]
mcp_bundles = ["homeai-tools"]
model_provider = "custom.voice"
[agents.builder]
model_provider = "llamacpp.coder"
[providers.models.custom.voice]
model = "voice-x"
[providers.models.llamacpp.coder]
model = "coder"
"""


def test_the_tests_never_read_the_real_config():
    assert V.current() == TEST_VOICE_MODEL and ".zeroclaw" not in str(V.config_path())


def test_the_voice_agent_is_the_one_with_the_house_tools(tmp_path):
    p = tmp_path / "c.toml"
    p.write_text(LIVE_SHAPE)
    assert V.current(p) == "voice-x"
    p.write_text(LIVE_SHAPE.replace("[agents.local]", "[agents.jarvis]"))
    assert V.current(p) == "voice-x"     # a fresh install names it jarvis


@pytest.mark.parametrize("text", [
    "",                                                                  # empty
    "not toml [",                                                        # broken
    LIVE_SHAPE.replace('mcp_bundles = ["homeai-tools"]', ""),            # no voice agent
    LIVE_SHAPE + '[agents.other]\nmcp_bundles = ["homeai-tools"]\n',     # two: ambiguous
    LIVE_SHAPE.replace('model_provider = "custom.voice"', ""),           # no provider ref
    LIVE_SHAPE.replace("[providers.models.custom.voice]", "[providers.models.custom.x]"),   # dangling ref
    LIVE_SHAPE.replace('model = "voice-x"', ""),                    # no model
])
def test_an_unreadable_config_falls_back_to_the_default(tmp_path, text):
    p = tmp_path / "c.toml"
    p.write_text(text)
    assert V.current(p) == V.DEFAULT == "qwen35-voice"


def test_a_missing_file_falls_back_too(tmp_path):
    assert V.current(tmp_path / "nope.toml") == V.DEFAULT


def test_the_memory_helper_restores_jarvis_s_model(monkeypatch):
    """After a helper job Ollama reloads this model; any other name would evict Jarvis's."""
    from homeai.config import MemoryConfig
    monkeypatch.delenv("HOMEAI_LTM_RESTORE_MODEL", raising=False)
    assert MemoryConfig().restore_model == TEST_VOICE_MODEL


def test_the_spoken_budget_is_180_words(monkeypatch):
    """Adam 2026-11-10: ~100 words is a guideline; qwen3.5 was cut off twice at 110 (LLM_UPGRADE_PLAN step 2)."""
    from homeai.config import TtsConfig
    monkeypatch.delenv("HOMEAI_SPOKEN_BUDGET_WORDS", raising=False)
    assert TtsConfig().spoken_budget_words == 180
