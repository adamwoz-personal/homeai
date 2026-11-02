"""Daemon -> avatar state events (plans/AVATAR_PLAN.md §3, Phase 1).

Presence must never break the voice loop, must create no socket when the avatar
is off, and the daemon must publish the right state sequence for a turn. The
daemon must also never import the renderer (pygame/moderngl).
"""
from __future__ import annotations

import subprocess
import sys
import threading
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from homeai.avatar.presence import Presence
from homeai.config import Config
from tests.test_avatar_bus import connect, read_events, wait_until

REPO = Path(__file__).resolve().parents[1]


class FakeBus:
    def __init__(self, fail_start=False, fail_publish=False):
        self.events = []
        self.fail_start = fail_start
        self.fail_publish = fail_publish
        self.started = self.stopped = False

    def start(self):
        if self.fail_start:
            raise RuntimeError("socket in use")
        self.started = True

    def stop(self):
        self.stopped = True

    def publish(self, etype, data):
        if self.fail_publish:
            raise OSError("boom")
        self.events.append((etype, data))


def states(bus):
    return [d["state"] for t, d in bus.events if t == "state"]


# ---- Presence ---------------------------------------------------------------------

def test_disabled_is_noop_and_creates_no_socket(tmp_path):
    p = Presence(False, str(tmp_path / "ev.sock"))
    p.start()
    p.state("speaking")
    p.caption("user", "hi")
    p.stop()
    assert not p.enabled
    assert not (tmp_path / "ev.sock").exists()


def test_start_publishes_idle_and_stop_offline():
    bus = FakeBus()
    p = Presence(True, bus=bus)
    p.start()
    p.stop()
    assert states(bus) == ["idle", "offline"] and bus.stopped


def test_repeated_state_is_sent_once():
    bus = FakeBus()
    p = Presence(True, bus=bus)
    for s in ("thinking", "thinking", "speaking", "speaking", "thinking"):
        p.state(s)
    assert states(bus) == ["thinking", "speaking", "thinking"]


def test_unknown_state_ignored():
    bus = FakeBus()
    p = Presence(True, bus=bus)
    p.state("dancing")
    assert bus.events == []


def test_bus_start_failure_disables_quietly():
    p = Presence(True, bus=FakeBus(fail_start=True))
    p.start()
    assert not p.enabled
    p.state("speaking")  # no raise


def test_publish_failure_never_raises():
    p = Presence(True, bus=FakeBus(fail_publish=True))
    p.state("speaking")
    p.caption("jarvis", "hello")


def test_blank_caption_skipped():
    bus = FakeBus()
    p = Presence(True, bus=bus)
    p.caption("user", "   ")
    p.caption("user", " hi ")
    assert bus.events == [("caption", {"who": "user", "text": "hi"})]


def test_real_socket_snapshot_carries_state(tmp_path):
    path = tmp_path / "ev.sock"
    p = Presence(True, str(path))
    p.start()
    try:
        p.state("thinking")
        s = connect(path)
        try:
            hello, snap = read_events(s, 2)
            assert hello["type"] == "hello"
            assert snap["data"]["events"]["state"]["data"]["state"] == "thinking"
        finally:
            s.close()
    finally:
        p.stop()
    assert wait_until(lambda: not path.exists())


# ---- daemon wiring ----------------------------------------------------------------

@pytest.fixture
def assistant():
    with patch("homeai.daemon.Microphone"), \
         patch("homeai.daemon.Transcriber"), \
         patch("homeai.daemon.Speaker"), \
         patch("homeai.daemon.build_agent_client"), \
         patch("homeai.daemon.TranscriptLog"):
        from homeai.daemon import VoiceAssistant

        va = VoiceAssistant(Config())
    va.bus = FakeBus()
    va.presence = Presence(True, bus=va.bus)
    va._detector = None
    va._start_interrupt_listener = MagicMock(return_value=None)
    va.tts.say_safe.return_value = MagicMock(first_audio_ms=5.0)
    return va


def test_daemon_off_by_env_has_no_bus():
    with patch("homeai.daemon.Microphone"), patch("homeai.daemon.Transcriber"), \
         patch("homeai.daemon.Speaker"), \
         patch("homeai.daemon.build_agent_client"), patch("homeai.daemon.TranscriptLog"):
        from homeai.daemon import VoiceAssistant

        va = VoiceAssistant(Config())
    assert not va.presence.enabled
    assert va.tts.publish != va.presence.publish


def test_daemon_on_wires_speaker_publish(monkeypatch):
    monkeypatch.setenv("HOMEAI_AVATAR", "1")
    with patch("homeai.daemon.Microphone"), patch("homeai.daemon.Transcriber"), \
         patch("homeai.daemon.Speaker"), \
         patch("homeai.daemon.build_agent_client"), patch("homeai.daemon.TranscriptLog"):
        from homeai.daemon import VoiceAssistant

        va = VoiceAssistant(Config())
    assert va.presence.enabled
    assert va.tts.publish == va.presence.publish


def test_speak_goes_speaking_with_caption_then_thinking(assistant):
    assistant._speak("Hello there.")
    assert states(assistant.bus) == ["speaking", "thinking"]
    assert ("caption", {"who": "jarvis", "text": "Hello there."}) in assistant.bus.events


def test_enqueue_means_thinking(assistant):
    assistant._enqueue(np.zeros(10, dtype=np.float32))
    assert states(assistant.bus) == ["thinking"]


def test_turn_ends_idle_followup_or_listening(assistant):
    assistant._end_turn_state()
    assistant._followup_armed.set()
    assistant._end_turn_state()
    assistant._bargein_armed.set()
    assistant._end_turn_state()
    assert states(assistant.bus) == ["idle", "followup", "listening"]


def test_worker_turn_failure_still_ends_in_idle(assistant):
    assistant._handle = MagicMock(side_effect=RuntimeError("bad turn"))
    assistant._queue.put((np.zeros(4, dtype=np.float32), "wake", None))
    t = threading.Thread(target=assistant._worker_loop, daemon=True)
    t.start()
    assert wait_until(lambda: states(assistant.bus)[-1:] == ["idle"])
    assistant._stop.set()
    t.join(timeout=2)
    assert states(assistant.bus) == ["speaking", "thinking", "idle"]


def test_daemon_never_imports_renderer():
    code = ("import sys, homeai.daemon; "
            "bad = [m for m in ('pygame', 'moderngl') if m in sys.modules]; "
            "sys.exit(1 if bad else 0)")
    r = subprocess.run([sys.executable, "-c", code], cwd=REPO, capture_output=True, timeout=60)
    assert r.returncode == 0, r.stderr.decode()[-500:]
