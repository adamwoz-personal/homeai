# Copyright (c) 2026 Adam Wosotowsky
# SPDX-License-Identifier: BUSL-1.1
# Licensed under the Business Source License 1.1; see LICENSE at the
# repository root. Converts to Apache-2.0 on 2030-09-15.
"""Which Ollama model Jarvis talks with: read from ZeroClaw's config, the one place it is set.

Why: the name was hard-coded in six places (pinning in gpu_mode, the memory
helper's restore, the art patcher, the art tools, the installer). Ollama holds
one model (OLLAMA_MAX_LOADED_MODELS=1), so any of them naming a different model
from ZeroClaw's would evict Jarvis's model and leave the next answer slow.

The voice agent is the agent carrying the ``homeai-tools`` MCP bundle (named
``local`` here, ``jarvis`` on a fresh install); its ``model_provider`` points at
``[providers.models.<ref>]``, whose ``model`` is the answer.
``tools/switch_voice_model.py`` changes it.
"""
from __future__ import annotations

import logging
import os
import tomllib
from pathlib import Path

log = logging.getLogger(__name__)

DEFAULT = "qwen35-voice"
BUNDLE = "homeai-tools"
# Set on the voice provider for a model that can think. Ollama's OpenAI route honours only reasoning_effort
# (think/enable_thinking are ignored, Ollama 0.34). Measured 2026-11-10 on 93 questions: "none" answered in
# 0.95 s but said "I don't have real-time access" instead of looking up live data (Bitcoin 3/3); "medium" got
# 92/93 in 1.57 s. Adam chose thinking on.
REASONING = {"reasoning_effort": "medium"}


def config_path() -> Path:
    return Path(os.environ.get("HOMEAI_ZC_CONFIG", str(Path.home() / ".zeroclaw" / "config.toml")))


def provider_ref(cfg: dict) -> str:
    """The voice agent's model_provider, e.g. "custom.voice". Exactly one agent may carry the bundle."""
    agents = [name for name, a in (cfg.get("agents") or {}).items()
              if isinstance(a, dict) and BUNDLE in (a.get("mcp_bundles") or [])]
    if len(agents) != 1:
        raise ValueError(f"expected one agent with the {BUNDLE} bundle, found {agents or 'none'}")
    ref = cfg["agents"][agents[0]].get("model_provider")
    if not ref:
        raise ValueError(f"agent {agents[0]} has no model_provider")
    return ref


def provider_section(cfg: dict) -> dict:
    node = cfg.get("providers", {}).get("models", {})
    for part in provider_ref(cfg).split("."):
        node = node.get(part) if isinstance(node, dict) else None
        if node is None:
            raise ValueError(f"no [providers.models.{provider_ref(cfg)}] section")
    return node


def current(path: Path | None = None) -> str:
    """The voice model's Ollama name; DEFAULT (with a warning) if the config can't be read."""
    path = path or config_path()
    try:
        model = provider_section(tomllib.loads(path.read_text(encoding="utf-8"))).get("model")
        if model:
            return model
        raise ValueError("the voice provider has no model")
    except (OSError, ValueError, tomllib.TOMLDecodeError) as exc:
        log.warning("voice model from %s unreadable (%s); using %s", path, exc, DEFAULT)
        return DEFAULT


if __name__ == "__main__":
    print(current())
