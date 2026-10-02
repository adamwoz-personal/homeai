# Copyright (c) 2026 Adam Wosotowsky
# SPDX-License-Identifier: BUSL-1.1
"""Tests for coding/voice GPU mode switching (homeai/gpu_mode.py)."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from homeai import gpu_mode as G


class FakeSystem:
    """Simulates systemd, Ollama and llama-server. Records every action."""

    def __init__(self, *, voice_loaded=True, dropin=False, coder_starts=True,
                 voice_loads=True, voice_unloads=True, jarvis_stays_up=True):
        self.actions: list[str] = []
        self.voice_is_loaded = voice_loaded
        self.dropin: str | None = "x" if dropin else None
        self.coder_starts = coder_starts
        self._coder_ready_at: float | None = 0.0  # None: never becomes healthy
        self.voice_loads = voice_loads
        self.voice_unloads = voice_unloads
        self.jarvis_stays_up = jarvis_stays_up
        self.jarvis = True
        self.clock = 0.0
        self.coder_started_with: str | None = None

    def systemctl(self, *args: str) -> bool:
        self.actions.append("systemctl " + " ".join(args))
        verb, unit = args[0], args[-1]
        if verb in ("restart", "start") and unit == G.CODER_SERVICE:
            # Like the real server: healthy only after the weights load.
            self._coder_ready_at = self.clock + 4.0 if self.coder_starts else None
            self.coder_started_with = self.dropin
        elif verb == "stop" and unit == G.VOICE_SERVICE:
            self.jarvis = False
        elif verb == "start" and unit == G.VOICE_SERVICE:
            self.jarvis = self.jarvis_stays_up
        return True

    @property
    def coder_up(self) -> bool:
        return self._coder_ready_at is not None and self.clock >= self._coder_ready_at

    def is_active(self, unit: str) -> bool:
        return self.jarvis if unit == G.VOICE_SERVICE else self.coder_up

    def http(self, url, payload=None, timeout=10.0):
        if url.endswith("/api/ps"):
            models = [{"name": "llama31-voice:latest"}] if self.voice_is_loaded else []
            return 200, json.dumps({"models": models})
        if url.endswith("/api/generate"):
            self.actions.append(f"ollama keep_alive={payload['keep_alive']}")
            if payload["keep_alive"] == 0 and self.voice_unloads:
                self.voice_is_loaded = False
            if payload["keep_alive"] == -1 and self.voice_loads:
                assert self.coder_up, "voice loaded while llama-server was not ready"
                self.voice_is_loaded = True
            return 200, "{}"
        if url == G.CODER_HEALTH:
            return (200 if self.coder_up else 503), ""
        raise AssertionError(url)

    def dropin_exists(self) -> bool:
        return self.dropin is not None

    def write_dropin(self, text: str) -> None:
        self.actions.append("write dropin")
        self.dropin = text

    def remove_dropin(self) -> None:
        self.actions.append("remove dropin")
        self.dropin = None

    def coder_cmdline(self):
        return ["llama-server", "--cache-type-k", "q4_0", "--n-cpu-moe", "0"]

    def vram_gb(self):
        return 20.0, 18.5

    def sleep(self, s):
        self.clock += s

    def now(self):
        return self.clock


def switcher(fake):
    return G.ModeSwitcher(fake, log=lambda *_: None)


def index(fake, action):
    return next(i for i, a in enumerate(fake.actions) if a.startswith(action))


class TestCoding:
    def test_order_jarvis_off_then_unload_then_coder_restart(self):
        fake = FakeSystem()
        switcher(fake).coding()
        assert index(fake, "systemctl stop homeai") < index(fake, "ollama keep_alive=0") \
            < index(fake, "write dropin") < index(fake, "systemctl restart llama-server")
        assert not fake.voice_is_loaded and not fake.jarvis

    def test_coder_restarted_with_override(self):
        fake = FakeSystem()
        switcher(fake).coding(kv="q8_0", cpu_moe=6, ctx=98304)
        assert "Environment=KV=q8_0" in fake.coder_started_with
        assert "Environment=NCPUMOE=6" in fake.coder_started_with
        assert "Environment=CTX=98304" in fake.coder_started_with

    def test_default_is_measured_fastest(self):
        text = G.render_dropin(G.DEFAULT_KV, G.DEFAULT_CPU_MOE, None)
        assert "KV=q4_0" in text and "NCPUMOE=0" in text and "CTX" not in text

    def test_coder_fails_rolls_back_to_voice(self):
        fake = FakeSystem(coder_starts=False)
        sw = switcher(fake)
        fake_restart = fake.systemctl

        def systemctl(*args):
            ok = fake_restart(*args)
            if args[0] == "restart" and fake.dropin is None:
                fake._coder_ready_at = fake.clock + 4.0  # default config starts fine
            return ok
        fake.systemctl = systemctl
        with pytest.raises(G.ModeError, match="rolled back"):
            sw.coding()
        assert fake.dropin is None, "override must be removed"
        assert fake.voice_is_loaded and fake.jarvis, "Jarvis must be back"

    def test_voice_will_not_unload_rolls_back(self):
        fake = FakeSystem(voice_unloads=False)
        with pytest.raises(G.ModeError, match="still loaded"):
            switcher(fake).coding()
        assert fake.dropin is None and fake.jarvis

    def test_rollback_failure_is_reported(self):
        fake = FakeSystem(coder_starts=False, voice_loads=False)
        with pytest.raises(G.ModeError, match="rollback also failed"):
            switcher(fake).coding()

    def test_already_unloaded_is_fine(self):
        fake = FakeSystem(voice_loaded=False)
        switcher(fake).coding()
        assert "ollama keep_alive=0" not in fake.actions


class TestVoice:
    def test_from_coding_restores_default_coder_before_loading_voice(self):
        fake = FakeSystem(voice_loaded=False, dropin=True)
        fake.jarvis = False
        switcher(fake).voice()
        assert index(fake, "remove dropin") < index(fake, "systemctl restart llama-server") \
            < index(fake, "ollama keep_alive=-1") < index(fake, "systemctl start homeai")
        assert fake.coder_started_with is None
        assert fake.voice_is_loaded and fake.jarvis

    def test_already_voice_does_not_restart_coder(self):
        fake = FakeSystem()
        switcher(fake).voice()
        assert not any("restart llama-server" in a for a in fake.actions)
        assert "ollama keep_alive=-1" not in fake.actions, "already loaded"

    def test_voice_load_retries_then_fails(self):
        fake = FakeSystem(voice_loaded=False, voice_loads=False)
        with pytest.raises(G.ModeError, match="did not load"):
            switcher(fake).voice()
        assert fake.actions.count("ollama keep_alive=-1") == G.LOAD_ATTEMPTS

    def test_jarvis_crashing_is_reported(self):
        fake = FakeSystem(jarvis_stays_up=False)
        with pytest.raises(G.ModeError, match="did not stay up"):
            switcher(fake).voice()


class TestStatus:
    def test_reports_mode_and_config(self):
        st = switcher(FakeSystem(dropin=True, voice_loaded=False)).status()
        assert (st.mode, st.coder_kv, st.coder_cpu_moe, st.loaded) == ("coding", "q4_0", "0", [])
        assert "mode:          coding" in st.lines()

    def test_ollama_down_means_nothing_loaded(self):
        fake = FakeSystem()
        fake.http = lambda url, payload=None, timeout=10: (0, "")
        assert switcher(fake).loaded_models() == []


class TestRealSystemFiles:
    def test_dropin_round_trip(self, tmp_path: Path):
        sys_ = G.System(dropin_dir=tmp_path / "llama-server.service.d")
        assert not sys_.dropin_exists()
        sys_.write_dropin("x")
        assert sys_.dropin_exists()
        sys_.remove_dropin()
        sys_.remove_dropin()  # idempotent
        assert not sys_.dropin_exists()

    def test_dropin_is_runtime_only(self):
        assert str(G.runtime_dropin_dir(1000)).startswith("/run/user/1000/systemd/user/")
