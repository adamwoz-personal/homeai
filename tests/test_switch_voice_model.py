# Copyright (c) 2026 Adam Wosotowsky
# SPDX-License-Identifier: BUSL-1.1
# Licensed under the Business Source License 1.1; see LICENSE at the
# repository root. Converts to Apache-2.0 on 2030-09-15.
"""tools/switch_voice_model.py changes only the voice model, and puts things back if ZeroClaw fails.

Why: it rewrites ZeroClaw's live config and restarts it. A wrong edit (another
provider's model, a dropped section) or a failed restart would leave Jarvis
deaf; the checks before it must refuse a model Ollama can't serve at 16384.
"""
from __future__ import annotations

import importlib.util
import io
import sys
import tomllib
import urllib.error
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("switch_voice_model", REPO / "tools" / "switch_voice_model.py")
S = importlib.util.module_from_spec(spec)
spec.loader.exec_module(S)

from homeai import voice_model  # noqa: E402

CONFIG = """schema_version = 3

[agents.local]
mcp_bundles = ["homeai-tools"]
model_provider = "custom.voice"

[providers.models.llamacpp.coder]
model = "coder-model"

# --- the voice model ---
[providers.models.custom.voice]
model = "llama31-voice"
uri = "http://127.0.0.1:11434/v1"
temperature = 0.45

[risk_profiles.voice]
allowed_tools = ["calculator"]
"""


def test_only_the_voice_model_changes_and_thinking_goes_off():
    new = S.edit(CONFIG, "qwen35-voice")
    cfg = tomllib.loads(new)
    assert cfg["providers"]["models"]["custom"]["voice"]["model"] == "qwen35-voice"
    assert cfg["providers"]["models"]["custom"]["voice"]["provider_extra"] == {"reasoning_effort": "none"}
    assert cfg["providers"]["models"]["llamacpp"]["coder"]["model"] == "coder-model"
    assert "# --- the voice model ---" in new and new.count("\n") == CONFIG.count("\n") + 1
    assert S.edit(new, "llama31-voice").replace('model = "llama31-voice"', "") \
        == new.replace('model = "qwen35-voice"', "")                     # and back: the extra line stays once


def test_an_edit_twice_is_a_no_op():
    once = S.edit(CONFIG, "qwen35-voice")
    assert S.edit(once, "qwen35-voice") == once


def test_a_foreign_provider_extra_is_left_to_a_human():
    text = CONFIG.replace('temperature = 0.45', 'provider_extra = { route = "x" }')
    with pytest.raises(S.SwitchError):
        S.edit(text, "qwen35-voice")


def test_a_model_without_jarvis_s_context_is_refused():
    def post(url, payload=None, timeout=30.0):
        return {"parameters": "num_ctx                        4096\ntemperature 0.45"}
    with pytest.raises(S.SwitchError, match="16384"):
        S.check_model("m", post)
    S.check_model("m", lambda url, payload=None, timeout=30.0: {"parameters": "num_ctx                        16384"})


def test_a_model_ollama_lacks_is_refused():
    def post(url, payload=None, timeout=30.0):
        raise urllib.error.HTTPError(url, 404, "not found", {}, io.BytesIO())
    with pytest.raises(S.SwitchError, match="ollama create"):
        S.check_model("nope", post)


def test_a_change_beyond_the_voice_model_is_never_written():
    with pytest.raises(S.SwitchError):
        S.verify(CONFIG, CONFIG.replace('model = "coder-model"', 'model = "other"').replace(
            'model = "llama31-voice"', 'model = "q"\nprovider_extra = { reasoning_effort = "none" }'), "q")


def test_apply_backs_up_writes_and_restarts(tmp_path):
    path = tmp_path / "config.toml"
    path.write_text(CONFIG)
    new = S.edit(CONFIG, "qwen35-voice")
    ran = []
    backup = S.apply(path, new, run=lambda argv, check: ran.append(argv), ok=lambda: True, stamp="t",
                     log=lambda line: None)
    assert path.read_text() == new and backup.read_text() == CONFIG and backup.name == "config.toml.bak-t"
    assert ran == [["systemctl", "--user", "restart", "zeroclaw.service"]]
    assert voice_model.current(path) == "qwen35-voice"


def test_a_zeroclaw_that_fails_gets_the_old_config_back(tmp_path):
    path = tmp_path / "config.toml"
    path.write_text(CONFIG)
    ran = []
    with pytest.raises(S.SwitchError, match="restored"):
        S.apply(path, S.edit(CONFIG, "qwen35-voice"), run=lambda argv, check: ran.append(argv), ok=lambda: False,
                stamp="t", log=lambda line: None)
    assert path.read_text() == CONFIG and len(ran) == 2


def test_jarvis_in_use_is_noticed():
    def journal(argv, **kw):
        assert "--since" in argv and argv[-1] == "-q"
        return type("R", (), {"stdout": "Nov 10 heard: 'what time is it'"})()
    assert S.recently_heard(journal)
    assert not S.recently_heard(lambda argv, **kw: type("R", (), {"stdout": ""})())


def test_health_is_retried_then_given_up():
    calls = []

    def post(url, payload=None, timeout=30.0):
        calls.append(url)
        raise OSError("refused")
    assert not S.healthy(post, sleep=lambda s: None, tries=3) and len(calls) == 3
    assert S.healthy(lambda url, payload=None, timeout=30.0: {}, sleep=lambda s: None)
