# Copyright (c) 2026 Adam Wosotowsky
# SPDX-License-Identifier: BUSL-1.1
# Licensed under the Business Source License 1.1; see LICENSE at the
# repository root. Converts to Apache-2.0 on 2030-09-15.
"""tools/eval_coder_presets.sh evaluates each preset in turn and always puts the original coder back.

Why: it switches the live coder (llama-server on :8080) for hours. A failed
eval or a preset that won't start must still end with the coder Adam had.
Runs against fake systemctl/curl and a fake eval; nothing real is touched.
"""
from __future__ import annotations

import os
import subprocess
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]


def _fake(d: Path, name: str, body: str) -> None:
    p = d / name
    p.write_text("#!/usr/bin/env bash\n" + body + "\n")
    p.chmod(0o755)


def _run(tmp_path: Path, *presets: str, eval_fails: bool = False):
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    preset_file = tmp_path / "llm-preset"
    preset_file.write_text("qwen\n")
    log = tmp_path / "calls.log"
    _fake(bin_dir, "systemctl", f'echo "restart $(cat {preset_file})" >> {log}')
    _fake(bin_dir, "curl", "exit 0")
    _fake(bin_dir, "fakepy", f'echo "eval $* on $(cat {preset_file})" >> {log}; exit {1 if eval_fails else 0}')
    env = {**os.environ, "PATH": f"{bin_dir}:{os.environ['PATH']}", "PY": str(bin_dir / "fakepy"),
           "PRESET_FILE": str(preset_file)}
    out = subprocess.run([str(REPO / "tools" / "eval_coder_presets.sh"), *presets], cwd=REPO, env=env,
                         capture_output=True, text=True, timeout=400)
    return out, log.read_text().splitlines(), preset_file.read_text().strip()


def test_each_preset_is_served_then_evaluated_then_the_original_returns(tmp_path):
    out, calls, final = _run(tmp_path, "qwen36", "qwen")
    evals = [c for c in calls if c.startswith("eval")]
    assert len(evals) == 2
    assert evals[0].startswith("eval tools/eval_local_coder.py --label qwen36-") and evals[0].endswith("on qwen36")
    assert evals[1].endswith("on qwen")
    assert calls.index("restart qwen36") < calls.index(evals[0])
    assert calls[-1] == "restart qwen" and final == "qwen" and out.returncode == 0


def test_a_failing_eval_still_restores_and_reports_failure(tmp_path):
    out, calls, final = _run(tmp_path, "qwen36", eval_fails=True)
    assert out.returncode == 1 and calls[-1] == "restart qwen" and final == "qwen"
