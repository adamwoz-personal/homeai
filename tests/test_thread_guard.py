"""conftest guards against desktop "application stopped" pop-ups (2026-11-07): leaked threads and runaway memory.

Runs pytest in a subprocess on a test that leaks a thread, and checks the
leak is named and the real exit status survives the early exit.
"""
from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
LEAKY = """
import threading, time

def test_leaks():
    threading.Thread(target=time.sleep, args=(30,), name="left-behind", daemon=True).start()
    assert {ok}
"""


@pytest.mark.parametrize("ok, status", [(True, 0), (False, 1)])
def test_leaked_thread_is_named_and_status_kept(tmp_path, ok, status):
    (tmp_path / "test_leaky.py").write_text(LEAKY.format(ok=ok))
    run = subprocess.run([sys.executable, "-m", "pytest", "-q", "-o", "addopts=", "-p", "no:cacheprovider",
                          "-p", "tests.conftest", str(tmp_path / "test_leaky.py")],
                         cwd=REPO, capture_output=True, text=True, timeout=60)
    assert "threads still running at exit (1): left-behind" in run.stderr
    assert run.returncode == status


def test_memory_cap_turns_a_runaway_into_memory_error(memory_cap):
    import resource
    before = resource.getrlimit(resource.RLIMIT_DATA)
    with pytest.raises(MemoryError), memory_cap(64):
        hog = [bytearray(2**20) for _ in range(1024)]   # 1 GB: fits uncapped, so a broken cap fails without an OOM
        del hog
    assert resource.getrlimit(resource.RLIMIT_DATA) == before
