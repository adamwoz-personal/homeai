# Copyright (c) 2026 Adam Wosotowsky
# SPDX-License-Identifier: BUSL-1.1
# Licensed under the Business Source License 1.1; see LICENSE at the
# repository root. Converts to Apache-2.0 on 2030-09-15.
"""tools/bench_tokps.py never reports a generation rate it didn't measure.

Why: on 2026-11-10 Qwen3.6 stopped after one token on the raw prompt and the
bench printed 1000000 tok/s; the coder choice (LLM_UPGRADE_PLAN step 3) rests
on these numbers. The prompt goes through the model's own chat template.
"""
from __future__ import annotations

import importlib.util
import json
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("bench_tokps", REPO / "tools" / "bench_tokps.py")
B = importlib.util.module_from_spec(spec)
spec.loader.exec_module(B)


def test_the_prompt_is_wrapped_in_the_model_s_template():
    seen = {}

    def post(url, body, timeout):
        seen.update(url=url, body=body)
        return {"prompt": "<|im_start|>user\nX<|im_end|>\n<|im_start|>assistant\n"}
    assert B.chat_prompt("http://h:8080/", "X", post).startswith("<|im_start|>user\nX")
    assert seen["url"] == "http://h:8080/apply-template"
    assert seen["body"] == {"messages": [{"role": "user", "content": "X"}]}


def test_no_template_endpoint_falls_back_to_the_raw_text():
    def post(url, body, timeout):
        raise OSError("404")
    assert B.chat_prompt("http://h", "X", post) == "X"


def _main(tmp_path, monkeypatch, n_gen):
    prompts = []
    monkeypatch.setattr(B, "chat_prompt", lambda url, text: f"TEMPLATE({text})")
    monkeypatch.setattr(B, "completion", lambda url, prompt, n, timeout: prompts.append(prompt) or {
        "timings": {"prompt_n": 100, "prompt_per_second": 500.0, "predicted_n": n_gen,
                    "predicted_per_second": 1000000.0 if n_gen < 2 else 47.0},
        "stop_type": "eos" if n_gen < 400 else "limit"})
    out = tmp_path / "r.jsonl"
    assert B.main(["--reps", "2", "--out", str(out)]) == 0
    assert prompts and all(p.startswith("TEMPLATE(Here is a Python module.") for p in prompts)
    return json.loads(out.read_text())


def test_a_full_generation_is_measured(tmp_path, monkeypatch):
    r = _main(tmp_path, monkeypatch, 400)
    assert r["tg_median"] == 47.0 and r["short_runs"] == []


def test_a_one_token_generation_is_not_a_rate(tmp_path, monkeypatch, capsys):
    r = _main(tmp_path, monkeypatch, 1)
    assert r["tg_median"] is None and len(r["short_runs"]) == 2 and "stop 'eos'" in r["short_runs"][0]
    assert "NOT MEASURED" in capsys.readouterr().out
