# Copyright (c) 2026 Adam Wosotowsky
# SPDX-License-Identifier: BUSL-1.1
"""Python and Kotlin agree on the remote app's Noise channel (tools/noise_interop.py, REMOTE_APP_PLAN Phase 0).

Why: if the two libraries differ by one byte the phone can't talk to home, and
a checker that passes anything would hide it. The committed vectors are
Python's; the slow test builds the Kotlin side and checks both ways.
"""
from __future__ import annotations

import importlib.util
import shutil
import sys
from pathlib import Path

import pytest

pytest.importorskip("noise", reason="pip install .[remote]")

REPO = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("noise_interop", REPO / "tools" / "noise_interop.py")
N = importlib.util.module_from_spec(spec)
sys.modules["noise_interop"] = N
spec.loader.exec_module(N)

COMMITTED = N.FIXTURES / "nk-python.txt"


def test_committed_vectors_are_what_python_produces_today():
    assert N.check(N.read(COMMITTED)) == []


def test_one_changed_byte_is_reported():
    vec = N.read(COMMITTED)
    vec["a2h_1_cipher"] = N._flip(vec["a2h_1_cipher"])
    assert N.check(vec) == ["a2h_1_cipher differs"]


def test_a_missing_message_is_reported():
    vec = N.read(COMMITTED)
    del vec["msg2"]
    assert "missing msg2" in N.check(vec)


def test_tampered_messages_are_refused():
    assert N.tamper_problems(N.read(COMMITTED)) == []


def test_the_largest_message_fills_one_frame_exactly():
    vec = N.vectors()
    assert max(len(v) for k, v in vec.items() if k.endswith("_cipher")) == N.MAX_FRAME
    assert N.frame_problems(vec) == []
    assert N.frame_problems({"a2h_9_cipher": b"x" * (N.MAX_FRAME + 1)}) == ["a2h_9_cipher is 65536 bytes"]


def test_the_session_is_deterministic_and_round_trips(tmp_path):
    path = tmp_path / "v.txt"
    N.write(path, N.vectors())
    assert N.read(path) == N.vectors()


def test_kotlin_agrees_both_ways(tmp_path):
    """Builds tools/noise_interop_kotlin with Gradle (~10 s warm): byte-identical output, each decrypts the other."""
    if not shutil.which("java"):
        pytest.skip("no JDK")
    out = tmp_path / "kotlin.txt"
    proc = N.kotlin(out)
    assert proc.returncode == 0, proc.stdout + proc.stderr[-2000:]
    assert "OK kotlin agrees" in proc.stdout
    assert N.check(N.read(out)) == []
    assert out.read_bytes() == COMMITTED.read_bytes()


def test_kotlin_reports_a_changed_byte(tmp_path):
    """The Kotlin checker must fail on a vector that differs, not only pass on a good one."""
    if not shutil.which("java"):
        pytest.skip("no JDK")
    vec = N.read(COMMITTED)
    vec["a2h_1_cipher"] = N._flip(vec["a2h_1_cipher"])
    bad = tmp_path / "bad.txt"
    N.write(bad, vec)
    proc = N.kotlin(tmp_path / "k.txt", bad)
    assert proc.returncode != 0 and "a2h_1_cipher differs" in proc.stdout
