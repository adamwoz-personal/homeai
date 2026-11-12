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
                 voice_loads=True, voice_unloads=True, jarvis_stays_up=True,
                 comfy=False, comfy_starts=True, comfy_healthy=True, coder_busy=False):
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
        self.comfy = comfy
        self.comfy_starts = comfy_starts
        self.comfy_healthy = comfy_healthy
        self.comfy_stops = True
        self.coder_busy = coder_busy

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
        elif verb == "stop" and unit == G.CODER_SERVICE:
            self._coder_ready_at = None
        elif verb == "stop" and unit == G.COMFY_UNIT:
            self.comfy = not self.comfy_stops
        return True

    def start_comfy(self) -> bool:
        self.actions.append("start comfy")
        assert not self.coder_up, "ComfyUI started while the coder still held its VRAM"
        self.comfy = self.comfy_starts
        return self.comfy_starts

    @property
    def coder_up(self) -> bool:
        return self._coder_ready_at is not None and self.clock >= self._coder_ready_at

    def is_active(self, unit: str) -> bool:
        if unit == G.COMFY_UNIT:
            return self.comfy
        return self.jarvis if unit == G.VOICE_SERVICE else self.coder_up

    def http(self, url, payload=None, timeout=10.0):
        if url.endswith("/api/ps"):
            cpu = getattr(self, "voice_on_cpu", 0)
            models = ([{"name": "llama31-voice:latest", "size": 10, "size_vram": 10 - cpu}]
                      if self.voice_is_loaded else [])
            if cpu and payload is None and getattr(self, "cpu_heals", False) and "ollama keep_alive=-1" in self.actions:
                self.voice_on_cpu = 0
            return 200, json.dumps({"models": models})
        if url.endswith("/api/generate"):
            self.actions.append(f"ollama keep_alive={payload['keep_alive']}")
            if payload["keep_alive"] == 0 and self.voice_unloads:
                self.voice_is_loaded = False
            if payload["keep_alive"] == -1 and self.voice_loads:
                assert self.coder_up, "voice loaded while llama-server was not ready"
                self.voice_is_loaded = True
            return 200, "{}"
        if url == G.COMFY_HEALTH:
            return (200 if self.comfy and self.comfy_healthy else 0), ""
        if url == G.CODER_SLOTS:
            return 200, json.dumps([{"id": 0, "is_processing": self.coder_busy}])
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

    @pytest.mark.parametrize("text, want", [("qwen\n", ("q4_0", 0)), ("qwen36\n", ("q8_0", 6)),
                                            ("", ("q4_0", 0)), ("newcoder", ("q8_0", 6))])
    def test_defaults_follow_the_coder_preset(self, tmp_path, text, want):
        """qwen36 (22.4 GB) cannot load with every expert on the GPU; an unknown coder gets the safe offload."""
        (tmp_path / "p").write_text(text)
        assert G.coding_defaults(tmp_path / "p") == want

    def test_no_preset_file_means_llm_serve_s_fallback(self, tmp_path):
        assert G.coding_defaults(tmp_path / "missing") == ("q4_0", 0)

    @pytest.mark.parametrize("argv, want", [([], ("q8_0", 6, None)),
                                            (["--kv", "q4_0", "--cpu-moe", "0"], ("q4_0", 0, None))])
    def test_cli_uses_the_preset_s_defaults_unless_told(self, tmp_path, monkeypatch, argv, want):
        (tmp_path / "p").write_text("qwen36")
        monkeypatch.setattr(G, "PRESET_FILE", tmp_path / "p")
        got = []
        monkeypatch.setattr(G, "System", lambda: None)
        monkeypatch.setattr(G, "ModeSwitcher", lambda sys, **kw: type("S", (), {
            "coding": lambda self, *a: got.append(a),
            "status": lambda self: type("St", (), {"lines": lambda self: []})()})())
        G.main(["coding", *argv])
        assert got == [want]

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


class TestCharacterisation:
    """Behaviour pinned before art mode was added (AVATAR_PLAN 6a step 1)."""

    def test_status_in_voice_mode(self):
        st = switcher(FakeSystem()).status()
        assert (st.mode, st.jarvis, st.coder_up) == ("voice", True, True)
        assert st.loaded == ["llama31-voice:latest"]
        assert "jarvis:        running (homeai.service)" in st.lines()

    def test_voice_starts_a_stopped_coder_before_loading_voice(self):
        fake = FakeSystem(voice_loaded=False)
        fake._coder_ready_at = None
        switcher(fake).voice()
        assert index(fake, "systemctl start llama-server") < index(fake, "ollama keep_alive=-1")
        assert fake.voice_is_loaded and fake.jarvis

    def test_voice_carries_on_when_the_coder_will_not_come_up(self):
        fake = FakeSystem(coder_starts=False)
        fake._coder_ready_at = None
        logs = []
        G.ModeSwitcher(fake, log=logs.append).voice()
        assert any("not healthy; continuing with Jarvis" in line for line in logs)
        assert fake.jarvis

    def test_coding_from_coding_rewrites_the_override(self):
        fake = FakeSystem(dropin=True, voice_loaded=False)
        fake.jarvis = False
        switcher(fake).coding(kv="q8_0")
        assert "Environment=KV=q8_0" in fake.coder_started_with


class TestArt:
    def test_coder_stops_before_comfy_starts_and_jarvis_stays(self):
        fake = FakeSystem()
        switcher(fake).art()
        assert index(fake, "systemctl stop llama-server") < index(fake, "start comfy")
        assert fake.comfy and fake.jarvis and fake.voice_is_loaded
        assert f"systemctl stop {G.VOICE_SERVICE}" not in fake.actions

    def test_status_says_art(self):
        fake = FakeSystem()
        sw = switcher(fake)
        sw.art()
        st = sw.status()
        assert (st.mode, st.comfy_up) == ("art", True)
        assert "comfyui:       running (homeai-comfyui.service)" in st.lines()

    def test_busy_coder_is_left_alone(self):
        fake = FakeSystem(coder_busy=True)
        with pytest.raises(G.ModeError, match="busy"):
            switcher(fake).art()
        assert fake.actions == [] and fake.coder_up

    def test_refused_from_coding_mode(self):
        fake = FakeSystem(dropin=True)
        with pytest.raises(G.ModeError, match="voice first"):
            switcher(fake).art()
        assert "start comfy" not in fake.actions

    def test_already_in_art_does_nothing(self):
        fake = FakeSystem(comfy=True)
        switcher(fake).art()
        assert fake.actions == []

    def test_comfy_that_never_answers_rolls_back_to_voice(self):
        fake = FakeSystem(comfy_healthy=False)
        with pytest.raises(G.ModeError, match="rolled back"):
            switcher(fake).art()
        assert not fake.comfy, "ComfyUI must be stopped"
        assert fake.coder_up and fake.jarvis and fake.voice_is_loaded

    def test_comfy_that_will_not_start_rolls_back(self):
        fake = FakeSystem(comfy_starts=False)
        with pytest.raises(G.ModeError, match="could not start.*rolled back"):
            switcher(fake).art()
        assert fake.coder_up

    def test_rollback_failure_is_reported(self):
        fake = FakeSystem(comfy_healthy=False, coder_starts=False, voice_loaded=False, voice_loads=False)
        with pytest.raises(G.ModeError, match="rollback also failed"):
            switcher(fake).art()

    def test_odd_slots_answer_counts_as_idle(self):
        fake = FakeSystem()
        real = fake.http
        fake.http = lambda url, payload=None, timeout=10: (200, "not json") if url == G.CODER_SLOTS \
            else real(url, payload, timeout)
        assert switcher(fake).coder_busy() is False


class TestLeavingArt:
    def test_voice_stops_comfy_before_starting_the_coder(self):
        fake = FakeSystem()
        sw = switcher(fake)
        sw.art()
        fake.actions.clear()
        sw.voice()
        assert index(fake, "systemctl stop homeai-comfyui") < index(fake, "systemctl start llama-server")
        assert not fake.comfy and fake.coder_up and sw.mode() == "voice"

    def test_coding_stops_comfy_first(self):
        fake = FakeSystem()
        sw = switcher(fake)
        sw.art()
        fake.actions.clear()
        sw.coding()
        assert fake.actions[0] == f"systemctl stop {G.COMFY_UNIT}"
        assert fake.actions.index(f"systemctl stop {G.COMFY_UNIT}") < fake.actions.index(f"systemctl stop {G.VOICE_SERVICE}")
        assert not fake.comfy and sw.mode() == "coding"

    def test_comfy_that_will_not_stop_is_an_error(self):
        fake = FakeSystem(comfy=True)
        fake.comfy_stops = False
        with pytest.raises(G.ModeError, match="would not stop"):
            switcher(fake).voice()


class TestComfyCommand:
    def test_capped_and_transient(self):
        cmd = G.comfy_command(Path("/c"))
        assert cmd[:2] == ["systemd-run", "--user"] and "--scope" not in cmd
        assert "MemoryMax=24G" in cmd and "MemorySwapMax=0" in cmd
        assert "--unit=homeai-comfyui.service" in cmd and "WorkingDirectory=/c/app" in cmd
        assert cmd[cmd.index("--listen") + 1] == "127.0.0.1"


class TestRestartComfy:
    def test_stops_then_starts_and_waits(self):
        fake = FakeSystem()
        sw = switcher(fake)
        sw.art()
        fake.actions.clear()
        sw.restart_comfy()
        assert fake.actions[0] == f"systemctl stop {G.COMFY_UNIT}" and "start comfy" in fake.actions
        assert fake.comfy and sw.mode() == "art"

    def test_only_in_art_mode(self):
        fake = FakeSystem()
        with pytest.raises(G.ModeError, match="not in art mode"):
            switcher(fake).restart_comfy()
        assert "start comfy" not in fake.actions

    def test_comfy_that_stays_dead_is_reported(self):
        fake = FakeSystem()
        sw = switcher(fake)
        sw.art()
        fake.comfy_healthy = False
        with pytest.raises(G.ModeError, match="after a restart"):
            sw.restart_comfy()


# -- benchmarks: the coder paused for a with block (code review item 3) ---------------------------------

def paused(fs, **kw):
    return G.ModeSwitcher(fs, log=lambda line: None).coder_paused(**kw)


def test_coder_paused_stops_the_coder_and_brings_everything_back():
    fs = FakeSystem()
    with paused(fs):
        assert fs.actions == [f"systemctl stop {G.CODER_SERVICE}"] and fs.jarvis
    assert fs.actions[1:] == [f"systemctl start {G.CODER_SERVICE}", "ollama keep_alive=0", "ollama keep_alive=-1",
                              f"systemctl start {G.VOICE_SERVICE}"]
    assert fs.coder_up and fs.voice_is_loaded and fs.jarvis


def test_voice_off_stops_jarvis_and_unloads_the_voice_model_first():
    fs = FakeSystem()
    with paused(fs, voice_off=True):
        assert fs.actions == [f"systemctl stop {G.CODER_SERVICE}", f"systemctl stop {G.VOICE_SERVICE}",
                              "ollama keep_alive=0"]
        assert not fs.jarvis and not fs.voice_is_loaded
    assert fs.jarvis and fs.voice_is_loaded and fs.coder_up


def test_a_failing_benchmark_still_gets_everything_back():
    fs = FakeSystem()
    with pytest.raises(RuntimeError, match="boom"):
        with paused(fs, voice_off=True):
            raise RuntimeError("boom")
    assert fs.jarvis and fs.voice_is_loaded and fs.coder_up


@pytest.mark.parametrize("kw, why", [({"coder_busy": True}, "busy"), ({"comfy": True}, "art mode"),
                                     ({"dropin": True}, "coding mode")])
def test_coder_paused_refuses_unless_idle_in_voice_mode(kw, why):
    fs = FakeSystem(**kw)
    with pytest.raises(G.ModeError, match=why):
        with paused(fs):
            pass
    assert fs.actions == []


def test_a_voice_model_left_on_the_cpu_is_reloaded_until_it_is_on_the_gpu():
    fs = FakeSystem()
    fs.voice_on_cpu, fs.cpu_heals = 1, False
    with paused(fs):
        pass
    assert fs.actions.count("ollama keep_alive=-1") == G.PAUSE_RELOADS and fs.jarvis
    fs = FakeSystem()
    fs.voice_on_cpu, fs.cpu_heals = 1, True
    with paused(fs):
        pass
    assert fs.actions.count("ollama keep_alive=-1") == 1


def test_a_coder_that_never_comes_back_still_gets_jarvis_back():
    fs = FakeSystem(voice_loaded=False, coder_starts=False)
    with paused(fs):
        pass
    assert fs.actions[-1] == f"systemctl start {G.VOICE_SERVICE}" and "ollama keep_alive=-1" not in fs.actions


# -- voice-model evals: Jarvis paused, the coder left up (LLM_UPGRADE_PLAN) --------------------------------

def jarvis_paused(fs):
    return G.ModeSwitcher(fs, log=lambda line: None).jarvis_paused()


def test_jarvis_paused_stops_only_jarvis_and_brings_it_back_on_the_gpu():
    fs = FakeSystem()
    with jarvis_paused(fs):
        assert fs.actions == [f"systemctl stop {G.VOICE_SERVICE}"] and not fs.jarvis and fs.coder_up
        fs.voice_is_loaded = False   # a candidate took its place on the card
    assert fs.actions[1:] == ["ollama keep_alive=-1", f"systemctl start {G.VOICE_SERVICE}"]
    assert fs.jarvis and fs.voice_is_loaded and fs.coder_up


def test_a_failing_eval_still_gets_jarvis_back():
    fs = FakeSystem()
    with pytest.raises(RuntimeError, match="boom"):
        with jarvis_paused(fs):
            raise RuntimeError("boom")
    assert fs.jarvis and fs.voice_is_loaded


def test_jarvis_comes_back_even_when_the_voice_model_will_not_load():
    fs = FakeSystem()
    fs.voice_on_cpu, fs.cpu_heals = 1, False
    with jarvis_paused(fs):
        pass
    assert fs.actions.count("ollama keep_alive=-1") == G.PAUSE_RELOADS and fs.actions[-1].endswith(G.VOICE_SERVICE)


@pytest.mark.parametrize("kw, why", [({"comfy": True}, "art mode"), ({"dropin": True}, "coding mode")])
def test_jarvis_paused_refuses_outside_voice_mode(kw, why):
    fs = FakeSystem(**kw)
    with pytest.raises(G.ModeError, match=why):
        with jarvis_paused(fs):
            pass
    assert fs.actions == []


class TestOneSwitchAtATime:
    """The CLI and the daemon's art voice share one lock (AVATAR_PLAN 6a known gap, fixed 2026-11-11)."""

    def test_a_second_switcher_is_refused_while_one_is_switching(self, tmp_path):
        lock = tmp_path / "homeai" / "mode.lock"
        fake = FakeSystem()
        other = G.ModeSwitcher(FakeSystem(), log=lambda *_: None, lock_path=lock)
        seen = []
        start = fake.start_comfy

        def start_comfy():
            with pytest.raises(G.ModeError, match="another mode switch"):
                other.voice()
            seen.append("refused")
            return start()
        fake.start_comfy = start_comfy
        G.ModeSwitcher(fake, log=lambda *_: None, lock_path=lock).art()
        assert seen == ["refused"] and fake.comfy
        other.voice()                      # released once the first switch ended

    def test_a_failed_switch_rolls_back_under_its_own_lock_and_releases_it(self, tmp_path):
        lock = tmp_path / "mode.lock"
        sw = G.ModeSwitcher(FakeSystem(comfy_starts=False), log=lambda *_: None, lock_path=lock)
        with pytest.raises(G.ModeError, match="rolled back"):
            sw.art()
        G.ModeSwitcher(FakeSystem(), log=lambda *_: None, lock_path=lock).voice()

    def test_the_lock_is_where_the_sandboxed_daemon_may_write(self):
        assert G.mode_lock_path(1000) == Path("/run/user/1000/homeai/mode.lock")

    def test_the_cli_and_the_daemon_both_lock(self, monkeypatch):
        made = []
        monkeypatch.setattr(G, "ModeSwitcher", lambda sys, **kw: made.append(kw) or type("S", (), {
            "voice": lambda self: None, "status": lambda self: type("St", (), {"lines": lambda self: []})()})())
        monkeypatch.setattr(G, "System", lambda: None)
        G.main(["voice"])
        assert made == [{"lock_path": G.mode_lock_path()}]
