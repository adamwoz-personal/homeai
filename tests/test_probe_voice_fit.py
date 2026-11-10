# Copyright (c) 2026 Adam Wosotowsky
# SPDX-License-Identifier: BUSL-1.1
# Licensed under the Business Source License 1.1; see LICENSE at the
# repository root. Converts to Apache-2.0 on 2030-09-15.
"""tools/probe_voice_fit.py reads VRAM fit at Jarvis's context and never leaves Ollama (LLM_UPGRADE_PLAN step 1b).

Why: the voice-model switch rests on "fits wholly on the GPU beside the coder at
num_ctx 16384"; a spill or a wrong context must not read as "fits".
"""
from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "tools"))
spec = importlib.util.spec_from_file_location("probe_voice_fit", REPO / "tools" / "probe_voice_fit.py")
F = importlib.util.module_from_spec(spec)
spec.loader.exec_module(F)

GB = 1024 ** 3
CAPTURE = {"body": {"temperature": 0.45, "tools": [{"type": "function", "function": {"name": "homeai__weather"}}],
                    "messages": [{"role": "system", "content": "You are Jarvis."}]}}


class FakeOllama:
    """Holds one model; loading one evicts the other. ``spill`` models sit partly on the CPU."""

    def __init__(self, spill=(), ctx=16384):
        self.calls, self.held, self.spill, self.ctx = [], None, set(spill), ctx

    def __call__(self, url, payload=None, timeout=30.0):
        self.calls.append((url, payload))
        if url.endswith("/api/ps"):
            if not self.held:
                return {"models": []}
            vram = 6 * GB if self.held in self.spill else 8 * GB
            return {"models": [{"name": f"{self.held}:latest", "size": 8 * GB, "size_vram": vram,
                                "context_length": self.ctx}]}
        if url.endswith("/api/generate"):
            assert payload["keep_alive"] == 0
            self.held = None
            return {}
        self.held = payload["model"]
        return {"message": {"content": "A short answer."}}


def run(fake, models, reps=2, tmp_path=None):
    def card():   # 12 GB for the coder and desktop, 6.5 for whatever Ollama holds
        return (18.5 if fake.held else 12.0, 20.0)
    return F.run(CAPTURE, models, reps, tmp_path, fake, card=card,
                 sleep=lambda s: None, log=lambda line: None)


def test_a_model_wholly_on_the_gpu_at_jarvis_s_context_fits(tmp_path):
    fake = FakeOllama()
    data = run(fake, ["qwen3.5:9b"], tmp_path=tmp_path)
    d = data["models"]["qwen3.5:9b"]
    assert d["verdict"] == "fits" and len(d["rows"]) == 2
    row = d["rows"][0]
    assert (row["size_gb"], row["gpu_percent"], row["context"], row["headroom_gb"]) == (8.0, 100.0, 16384, 1.5)
    assert len(row["warm_s"]) == len(F.WARM_QUESTIONS)
    assert (tmp_path / "result.json").exists()
    assert row["card_model_gb"] == 6.5    # measured on the card, not taken from /api/ps
    assert "| qwen3.5:9b | fits | 8.0 | 100.0 | 16384 | 6.5 | 18.5 | 1.5 |" in F.table(data)


def test_a_spill_or_a_wrong_context_is_not_a_fit(tmp_path):
    assert run(FakeOllama(spill={"big"}), ["big"], tmp_path=tmp_path)["models"]["big"]["verdict"] == "spills"
    assert run(FakeOllama(ctx=4096), ["m"], tmp_path=tmp_path)["models"]["m"]["verdict"] == "wrong context"


def test_each_rep_starts_from_an_empty_card_and_ends_empty(tmp_path):
    fake = FakeOllama()
    run(fake, ["a", "b"], reps=1, tmp_path=tmp_path)
    urls = [url for url, _ in fake.calls]
    chats = [p for url, p in fake.calls if url.endswith("/api/chat")]
    assert fake.held is None                               # unloaded for jarvis_paused's reload
    assert urls.count(f"{F.OLLAMA}/api/generate") == 2     # a before b, b at the end
    assert all(p["options"]["num_ctx"] == F.CTX == 16384 and p["think"] is False for p in chats)
    assert {u.rsplit("/", 2)[-2] + "/" + u.rsplit("/", 1)[-1] for u in urls} <= {"api/ps", "api/generate", "api/chat"}
