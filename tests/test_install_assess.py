# Copyright (c) 2026 Adam Wosotowsky
# SPDX-License-Identifier: BUSL-1.1
"""Tests for the installer's machine assessment (homeai/install/assess.py)."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

from homeai.install import assess as A
from homeai.install.assess import Facts, Gpu

ALL_COMMANDS = {name: True for name in A.REQUIRED_COMMANDS} | {"compiler": True}


def facts(**overrides) -> Facts:
    """The reference machine (RX 7900 XT), overridable per test."""
    base = dict(
        os="Linux", arch="x86_64", python=(3, 14), cpu_cores=24,
        ram_total_gb=30.0, ram_available_gb=20.0, disk_free_gb=500.0,
        gpus=[Gpu("amd", "card1", 20.0, 0.5)], capture_devices=3,
        playback_devices=9, commands=dict(ALL_COMMANDS), systemd_user=True,
    )
    base.update(overrides)
    return Facts(**base)


class TestParsers:
    def test_meminfo(self) -> None:
        text = "MemTotal:       31457280 kB\nMemFree: 100 kB\nMemAvailable:   20971520 kB\n"
        assert A.parse_meminfo(text) == (30.0, 20.0)

    def test_meminfo_falls_back_to_memfree(self) -> None:
        assert A.parse_meminfo("MemTotal: 1048576 kB\nMemFree: 524288 kB\n") == (1.0, 0.5)

    def test_meminfo_garbage_is_zero_not_crash(self) -> None:
        assert A.parse_meminfo("nonsense\n:::\n") == (0.0, 0.0)

    def test_asound_pcm_counts_both_directions(self) -> None:
        text = (
            "00-00: ALC897 Analog : ALC897 Analog : playback 1 : capture 1\n"
            "00-03: HDMI 0 : HDMI 0 : playback 1\n"
            "03-00: USB Audio : USB Audio : capture 1\n"
        )
        assert A.parse_asound_pcm(text) == (2, 2)

    def test_asound_pcm_empty(self) -> None:
        assert A.parse_asound_pcm("") == (0, 0)

    def test_nvidia_smi(self) -> None:
        gpus = A.parse_nvidia_smi("NVIDIA GeForce RTX 3060, 12288, 1024\nbad line\nX, n/a, 1\n")
        assert gpus == [Gpu("nvidia", "NVIDIA GeForce RTX 3060", 12.0, 1.0)]

    def test_amd_sysfs(self, tmp_path: Path) -> None:
        for card, total, used in (("card1", 21458059264, 1073741824), ("card2", 536870912, 0)):
            dev = tmp_path / card / "device"
            dev.mkdir(parents=True)
            (dev / "mem_info_vram_total").write_text(f"{total}\n")
            (dev / "mem_info_vram_used").write_text(f"{used}\n")
        gpus = A.read_amd_sysfs(tmp_path)
        assert [g.name for g in gpus] == ["card1", "card2"]
        assert gpus[0].vram_total_gb == pytest.approx(20.0, abs=0.1)
        assert gpus[0].vram_used_gb == pytest.approx(1.0)

    def test_amd_sysfs_unreadable_card_skipped(self, tmp_path: Path) -> None:
        dev = tmp_path / "card0" / "device"
        dev.mkdir(parents=True)
        (dev / "mem_info_vram_total").write_text("not a number")
        assert A.read_amd_sysfs(tmp_path) == []


class TestTierChoice:
    def test_reference_machine_gets_16k_on_gpu(self) -> None:
        tier = A.choose_tier(facts())
        assert tier is not None
        assert (tier.name, tier.num_ctx, tier.uses_gpu, tier.whisper_model) == (
            "gpu-16k", 16384, True, "small.en")

    def test_best_gpu_is_chosen_not_first(self) -> None:
        f = facts(gpus=[Gpu("amd", "igpu", 0.5), Gpu("nvidia", "big", 12.0)])
        assert A.choose_tier(f).name == "gpu-16k"

    @pytest.mark.parametrize("vram,expected", [
        (A.VRAM_FOR_16K_GB, "gpu-16k"),
        (A.VRAM_FOR_16K_GB - 0.1, "gpu-8k"),
        (A.VRAM_FOR_8K_GB, "gpu-8k"),
        (A.VRAM_FOR_8K_GB - 0.1, "cpu-8k"),  # 30 GB RAM, 24 threads
    ])
    def test_vram_boundaries(self, vram: float, expected: str) -> None:
        assert A.choose_tier(facts(gpus=[Gpu("nvidia", "g", vram)])).name == expected

    def test_cpu_tier_needs_ram_and_cores(self) -> None:
        assert A.choose_tier(facts(gpus=[], ram_total_gb=A.MIN_RAM_CPU_GB - 0.1)) is None
        assert A.choose_tier(facts(gpus=[], cpu_cores=A.MIN_CORES_CPU - 1)) is None
        assert A.choose_tier(facts(gpus=[])).name == "cpu-8k"

    def test_big_gpu_but_tiny_ram_falls_through(self) -> None:
        assert A.choose_tier(facts(ram_total_gb=4.0)) is None

    def test_few_cores_get_base_whisper(self) -> None:
        assert A.choose_tier(facts(cpu_cores=6)).whisper_model == "base.en"


class TestAssess:
    def test_reference_machine_ok(self) -> None:
        result = A.assess(facts())
        assert result.ok and result.blockers == []

    def test_underpowered_aborts_with_explanation(self) -> None:
        result = A.assess(facts(gpus=[], ram_total_gb=7.6, cpu_cores=4))
        assert not result.ok
        text = " ".join(result.blockers)
        assert "Not enough resources" in text
        assert "7.6 GB RAM" in text and "4 threads" in text, "must say what was found"

    def test_all_blockers_reported_at_once(self) -> None:
        result = A.assess(facts(
            os="Darwin", python=(3, 10), disk_free_gb=2.0, capture_devices=0,
            commands={**ALL_COMMANDS, "ollama": False, "compiler": False}))
        joined = "\n".join(result.blockers)
        for needle in ("Linux is required", "Python 3.12", "free disk", "`ollama`",
                       "C++ compiler", "microphone"):
            assert needle in joined, needle

    def test_missing_command_includes_fix(self) -> None:
        result = A.assess(facts(commands={**ALL_COMMANDS, "cmake": False}))
        assert any("cmake" in b and "apt install cmake" in b for b in result.blockers)

    def test_no_audio_is_warning_when_not_required(self) -> None:
        result = A.assess(facts(capture_devices=0, playback_devices=0), require_audio=False)
        assert result.ok
        assert any("microphone" in w for w in result.warnings)

    def test_busy_gpu_warns_but_installs(self) -> None:
        result = A.assess(facts(gpus=[Gpu("amd", "card1", 20.0, 19.9)]))
        assert result.ok
        assert any("in use right now" in w for w in result.warnings)

    def test_cpu_tier_warns_it_is_slow(self) -> None:
        result = A.assess(facts(gpus=[]))
        assert result.ok and result.tier.name == "cpu-8k"
        assert any("CPU" in w and "unmeasured" in w for w in result.warnings)

    def test_too_few_cores_for_speech_blocks_even_with_gpu(self) -> None:
        result = A.assess(facts(cpu_cores=2))
        assert not result.ok
        assert any("CPU threads" in b for b in result.blockers)

    def test_no_systemd_is_only_a_warning(self) -> None:
        result = A.assess(facts(systemd_user=False))
        assert result.ok and any("systemd" in w for w in result.warnings)


class TestOutputs:
    def test_facts_round_trip_through_json(self) -> None:
        original = facts()
        data = json.loads(json.dumps(A.assess(original).to_dict()))["facts"]
        assert Facts.from_dict(data) == original

    def test_from_dict_ignores_unknown_keys(self) -> None:
        assert Facts.from_dict({"os": "Linux", "future_field": 1}).os == "Linux"

    def test_env_file_is_shell_safe(self, tmp_path: Path) -> None:
        env = tmp_path / "env"
        A.write_env_file(A.assess(facts()), env)
        out = subprocess.run(["bash", "-c", f'source "{env}"; echo "$TIER_NAME $NUM_CTX $USES_GPU"'],
                             capture_output=True, text=True, check=True)
        assert out.stdout.strip() == "gpu-16k 16384 1"

    def test_env_file_for_refusal(self, tmp_path: Path) -> None:
        env = tmp_path / "env"
        A.write_env_file(A.assess(facts(gpus=[], ram_total_gb=4)), env)
        assert "ASSESS_OK=0" in env.read_text()

    def test_report_mentions_result(self) -> None:
        assert "CANNOT INSTALL" in A.format_report(A.assess(facts(gpus=[], ram_total_gb=4)))
        assert "OK to install" in A.format_report(A.assess(facts()))

    def test_cli_exit_codes_with_facts_file(self, tmp_path: Path) -> None:
        good, bad = tmp_path / "good.json", tmp_path / "bad.json"
        good.write_text(json.dumps(A.assess(facts()).to_dict()["facts"]))
        bad.write_text(json.dumps(A.assess(facts(gpus=[], ram_total_gb=4)).to_dict()["facts"]))
        run = lambda p: subprocess.run(  # noqa: E731
            [sys.executable, "-m", "homeai.install.assess", "--facts", str(p)],
            capture_output=True, text=True)
        assert run(good).returncode == 0
        refused = run(bad)
        assert refused.returncode == 2 and "Not enough resources" in refused.stdout

    def test_probe_runs_on_this_machine(self, tmp_path: Path) -> None:
        f = A.probe(tmp_path)
        assert f.os and f.cpu_cores > 0 and f.disk_free_gb > 0
