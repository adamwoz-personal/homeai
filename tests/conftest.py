# Copyright (c) 2026 Adam Wosotowsky
# SPDX-License-Identifier: BUSL-1.1
# Licensed under the Business Source License 1.1; see LICENSE at the
# repository root. Converts to Apache-2.0 on 2030-09-15.
"""Keep unit tests away from the real house.

This machine has real Home Assistant credentials. Without this, the MCP
tests would list (and could call) the real home tools, and timer ledgers
and quiet windows would be written to the real runtime directory.
"""

from __future__ import annotations

import os

# Art mode by voice switches real services (stops the coder, starts ComfyUI).
# A daemon test that says "draw a fox" must never do that; tests that want
# art mode build an ArtVoice with a fake switcher.
os.environ["HOMEAI_ART_VOICE"] = "0"

import pytest

from tests.memory_db import clean_db, mem_store, pg_dsn  # noqa: F401 - shared fixtures

@pytest.fixture(autouse=True)
def _isolated_home(tmp_path, monkeypatch):
    # Art-mode clips (homeai/art_clips.py): never into the real ~/.local/share (a run wrote 5 there, 2026-11-13).
    import homeai.art_clips
    monkeypatch.setattr(homeai.art_clips, "DIR", tmp_path / "art-clips")
    monkeypatch.setenv("HOMEAI_HA_ENV", str(tmp_path / "no-ha.env"))
    monkeypatch.setenv("XDG_RUNTIME_DIR", str(tmp_path / "run"))
    monkeypatch.delenv("HA_TOKEN", raising=False)
    monkeypatch.delenv("HA_URL", raising=False)
    # Long-term memory: off, and pointed nowhere, unless a test opts in with
    # its own throwaway database (tests/memory_db.py). Never the real one.
    monkeypatch.setenv("HOMEAI_LTM", "0")
    monkeypatch.setenv("HOMEAI_LTM_DSN", "dbname=homeai_tests_must_not_use_this")
    # Nor the real helper model: tests use ScriptedHelper. Port 9 refuses fast.
    monkeypatch.setenv("HOMEAI_LTM_HELPER_URL", "http://127.0.0.1:9")
    # No avatar event bus unless a test opts in (tests/test_avatar_presence.py).
    monkeypatch.setenv("HOMEAI_AVATAR", "0")
    # homeai.service may be a system unit on this box; tests pick the layout themselves.
    monkeypatch.setenv("HOMEAI_SERVICE_SCOPE", "user")
    # Never the real ZeroClaw config: the voice model comes from this one (homeai/voice_model.py).
    zc = tmp_path / "zc-config.toml"
    zc.write_text(VOICE_ZC_CONFIG)
    monkeypatch.setenv("HOMEAI_ZC_CONFIG", str(zc))
    (tmp_path / "run").mkdir(exist_ok=True)


TEST_VOICE_MODEL = "llama31-voice"
VOICE_ZC_CONFIG = f"""
[agents.local]
mcp_bundles = ["homeai-tools"]
model_provider = "custom.voice"

[agents.builder]
model_provider = "llamacpp.coder"

[providers.models.custom.voice]
model = "{TEST_VOICE_MODEL}"
"""


# -- threads left running at exit ---------------------------------------------
# A daemon thread still running while Python shuts down can segfault the
# interpreter, and Ubuntu's crash reporter then puts an "application stopped"
# pop-up on Adam's desktop (2026-11-07: a mutant left the mic watchdog
# reopening). Tests must stop what they start; this hook names any thread that
# is still alive and leaves without interpreter finalisation, so a leak shows
# up in the output instead of as a crash.
_THREADS_AT_START: set[int] = set()


def pytest_configure(config):
    import threading
    _THREADS_AT_START.update(t.ident for t in threading.enumerate())


def pytest_unconfigure(config):
    import os
    import sys
    import threading
    leaked = [t for t in threading.enumerate()
              if t.is_alive() and t.ident not in _THREADS_AT_START and t is not threading.current_thread()]
    if not leaked:
        return
    names = ", ".join(sorted({t.name for t in leaked}))
    print(f"\nthreads still running at exit ({len(leaked)}): {names}", file=sys.stderr)
    status = getattr(config, "_homeai_exitstatus", 0)
    sys.stdout.flush()
    sys.stderr.flush()
    os._exit(int(status))


def pytest_sessionfinish(session, exitstatus):
    session.config._homeai_exitstatus = exitstatus


@pytest.fixture(autouse=True)
def _close_memory_services(monkeypatch):
    """Close every MemoryService a test started, so its ltm-recorder thread ends with the test."""
    from homeai.memory_store.service import MemoryService
    started = []
    real_start = MemoryService.start

    def start(self):
        started.append(self)
        return real_start(self)

    monkeypatch.setattr(MemoryService, "start", start)
    yield
    for svc in started:
        svc.close(timeout=2.0)


@pytest.fixture
def memory_cap():
    """``with memory_cap(mb):`` makes a runaway allocation raise MemoryError inside the test.

    For tests of loop guards: with the guard broken (a mutant), the loop eats
    memory until the kernel OOM-kills it, and GNOME shows an "application
    stopped" pop-up for every OOM kill (2026-11-07). RLIMIT_DATA turns that
    into an ordinary failure. The limit is restored afterwards.
    """
    import contextlib
    import re
    import resource
    from pathlib import Path

    @contextlib.contextmanager
    def cap(extra_mb: int = 256):
        status = Path("/proc/self/status").read_text()
        used = int(re.search(r"VmData:\s+(\d+) kB", status).group(1)) * 1024
        soft, hard = resource.getrlimit(resource.RLIMIT_DATA)
        resource.setrlimit(resource.RLIMIT_DATA, (used + extra_mb * 2**20, hard))
        try:
            yield
        finally:
            resource.setrlimit(resource.RLIMIT_DATA, (soft, hard))

    return cap
