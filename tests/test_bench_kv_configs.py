# Copyright (c) 2026 Adam Wosotowsky
# SPDX-License-Identifier: BUSL-1.1
# Licensed under the Business Source License 1.1; see LICENSE at the
# repository root. Converts to Apache-2.0 on 2030-09-15.
"""tools/bench_kv_configs.sh runs the chosen preset, flags a coder that pushes Jarvis off the GPU, and restores.

Why: LLM_UPGRADE_PLAN step 3 picks a coder for voice mode from this script's
numbers. A fast config that leaves Jarvis's model partly on the CPU must read as
a failure, and the real llama-server must always come back. Runs the script
against fake systemctl/ollama/curl and a fake server; nothing real is touched.
"""
from __future__ import annotations

import os
import subprocess
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]


def _fake(bin_dir: Path, name: str, body: str) -> None:
    p = bin_dir / name
    p.write_text("#!/usr/bin/env bash\n" + body + "\n")
    p.chmod(0o755)


def _run(tmp_path: Path, where: str, *configs: str, preset: str = "qwen36") -> tuple[str, str]:
    home, bin_dir = tmp_path / "home", tmp_path / "bin"
    (home / "models").mkdir(parents=True)
    bin_dir.mkdir()
    log = tmp_path / "calls.log"
    _fake(home / "models", "llm-serve.sh", f'echo "serve $1 KV=$KV" >> {log}; exec sleep 30')
    _fake(bin_dir, "systemctl", f'echo "systemctl $*" >> {log}')
    _fake(bin_dir, "curl", "exit 0")
    _fake(bin_dir, "ollama", f'[ "$1" = ps ] && printf "%s\\n" "NAME ID SIZE PROCESSOR CONTEXT UNTIL" '
                             f'"voice-x:latest abc 6.1 GB {where} 16384 Forever"; exit 0')
    _fake(bin_dir, "fakepy", f'if [ "$1" = -m ]; then echo voice-x; else echo "$*" >> {log}; echo "pp 900 tg 60"; fi')
    env = {**os.environ, "HOME": str(home), "PATH": f"{bin_dir}:{os.environ['PATH']}", "PY": str(bin_dir / "fakepy"),
           "PRESET": preset, "OUT": str(tmp_path / "kv.jsonl")}
    out = subprocess.run([str(REPO / "tools" / "bench_kv_configs.sh"), *configs], cwd=REPO, env=env,
                         capture_output=True, text=True, timeout=60)
    return out.stdout + out.stderr, log.read_text()


def test_the_preset_is_served_and_jarvis_on_the_gpu_passes(tmp_path):
    out, calls = _run(tmp_path, "100% GPU", "KV=q8_0 NCPUMOE=10")
    assert "serve qwen36 KV=q8_0" in calls
    assert "--label qwen36 KV=q8_0 NCPUMOE=10" in calls
    assert "voice-x: 100% GPU" in out and "WARNING" not in out
    assert calls.index("systemctl --user stop llama-server.service") < calls.index("serve qwen36")
    assert calls.rstrip().endswith("systemctl --user start llama-server.service")   # always restored


def test_a_config_that_pushes_jarvis_onto_the_cpu_is_flagged(tmp_path):
    out, _ = _run(tmp_path, "12%/88% CPU/GPU", "KV=q4_0 NCPUMOE=0")
    assert "WARNING: voice-x is not wholly on the GPU with this config: fails voice mode" in out
