# Copyright (c) 2026 Adam Wosotowsky
# SPDX-License-Identifier: BUSL-1.1
"""Tests for adding the voice agent to a ZeroClaw config (homeai/install/zc_config.py)."""

from __future__ import annotations

import stat
import tomllib
from pathlib import Path

import pytest

from homeai.install import zc_config as Z

REPO = Path("/opt/homeai")
HOME = Path("/home/someone")


def merge(text: str, **kw) -> Z.MergeResult:
    return Z.merge(text, repo=REPO, home=HOME, num_ctx=kw.pop("num_ctx", 16384), **kw)


EXISTING = """\
schema_version = 3

[agents.local]
enabled = true
model_provider = "custom.mine"
secret = "enc2:abcdef"   # must survive byte-for-byte

[mcp]
enabled = true

[[mcp.servers]]
name = "hermes"
command = "/usr/bin/hermes"
"""


class TestFreshConfig:
    def test_creates_complete_config(self) -> None:
        result = merge("")
        data = tomllib.loads(result.text)
        assert data["schema_version"] == Z.SCHEMA_VERSION
        agent = data["agents"]["jarvis"]
        assert agent["risk_profile"] == Z.RISK_PROFILE
        assert agent["model_provider"] == f"custom.{Z.PROVIDER}"
        assert data["mcp"]["enabled"] is True
        assert data["mcp"]["servers"][0]["command"] == str(REPO / "bin/homeai-mcp")
        assert data["providers"]["models"]["custom"][Z.PROVIDER]["context_window"] == 16384

    def test_num_ctx_follows_tier(self) -> None:
        data = tomllib.loads(merge("", num_ctx=8192).text)
        assert data["providers"]["models"]["custom"][Z.PROVIDER]["context_window"] == 8192

    def test_voice_profile_is_locked_down(self) -> None:
        profile = tomllib.loads(merge("").text)["risk_profiles"][Z.RISK_PROFILE]
        assert profile["allowed_tools"] == ["calculator"]
        assert profile["level"] == "readonly"
        assert profile["workspace_only"] is True
        assert profile["allowed_commands"] == ["echo"]
        # The regression that made the model parrot old conversations.
        assert "memory_recall" not in profile["auto_approve"]
        assert str(REPO) in profile["forbidden_paths"], "voice must not read the code"
        assert str(HOME / ".zeroclaw/voice-scratch") in profile["allowed_roots"]

    def test_custom_agent_name(self) -> None:
        assert "friday" in tomllib.loads(merge("", agent="friday").text)["agents"]


class TestExistingConfig:
    def test_appends_and_preserves_original_bytes(self) -> None:
        result = merge(EXISTING)
        assert result.text.startswith(EXISTING.rstrip("\n"))
        data = tomllib.loads(result.text)
        assert data["agents"]["local"]["secret"] == "enc2:abcdef"
        assert [s["name"] for s in data["mcp"]["servers"]] == ["hermes", "homeai"]
        assert "mcp" not in result.added, "existing [mcp] must not be redefined"

    def test_rerun_is_a_no_op(self) -> None:
        once = merge(EXISTING)
        twice = merge(once.text)
        assert not twice.changed
        assert twice.text == once.text

    def test_existing_agent_owns_its_sections(self) -> None:
        result = merge(EXISTING, agent="local")
        data = tomllib.loads(result.text)
        assert Z.RISK_PROFILE not in data.get("risk_profiles", {})
        assert Z.PROVIDER not in data.get("providers", {}).get("models", {}).get("custom", {})
        assert set(result.added) == {"mcp_bundles.homeai-tools", "mcp.servers.homeai"}

    def test_mcp_disabled_is_refused(self) -> None:
        with pytest.raises(Z.ConfigError, match="enabled = false"):
            merge(EXISTING.replace("[mcp]\nenabled = true", "[mcp]\nenabled = false"))

    def test_mcp_table_without_enabled_is_refused(self) -> None:
        with pytest.raises(Z.ConfigError, match="without `enabled = true`"):
            merge(EXISTING.replace("[mcp]\nenabled = true\n", ""))

    def test_conflicting_homeai_server_is_refused(self) -> None:
        text = EXISTING + '\n[[mcp.servers]]\nname = "homeai"\ncommand = "/elsewhere/homeai-mcp"\n'
        with pytest.raises(Z.ConfigError, match="already exists"):
            merge(text)

    def test_broken_config_is_refused_not_rewritten(self) -> None:
        with pytest.raises(Z.ConfigError, match="does not parse"):
            merge("[agents\nbroken")

    def test_existing_profile_with_our_name_is_left_alone(self) -> None:
        text = EXISTING + f'\n[risk_profiles.{Z.RISK_PROFILE}]\nlevel = "full"\n'
        result = merge(text)
        assert tomllib.loads(result.text)["risk_profiles"][Z.RISK_PROFILE]["level"] == "full"
        assert any(s.startswith(f"risk_profiles.{Z.RISK_PROFILE}") for s in result.skipped)


class TestWrite:
    def test_writes_with_backup_and_private_mode(self, tmp_path: Path) -> None:
        cfg = tmp_path / "config.toml"
        cfg.write_text(EXISTING)
        backup = Z.write_config(cfg, merge(EXISTING))
        assert backup is not None and backup.read_text() == EXISTING
        assert stat.S_IMODE(cfg.stat().st_mode) == 0o600
        assert "jarvis" in tomllib.loads(cfg.read_text())["agents"]
        assert not cfg.with_name("config.toml.tmp").exists()

    def test_fresh_write_creates_directory(self, tmp_path: Path) -> None:
        cfg = tmp_path / "new" / "config.toml"
        assert Z.write_config(cfg, merge("")) is None
        assert cfg.exists()

    def test_cli_dry_run_writes_nothing(self, tmp_path: Path, capsys) -> None:
        cfg = tmp_path / "config.toml"
        cfg.write_text(EXISTING)
        assert Z.main(["--config", str(cfg), "--repo", str(REPO), "--dry-run"]) == 0
        assert cfg.read_text() == EXISTING
        assert "would add" in capsys.readouterr().out

    def test_cli_conflict_exit_code(self, tmp_path: Path) -> None:
        cfg = tmp_path / "config.toml"
        cfg.write_text("[agents\nbroken")
        assert Z.main(["--config", str(cfg), "--repo", str(REPO)]) == 2
        assert cfg.read_text() == "[agents\nbroken"


SPLIT = """\
[mcp]
enabled = true

[[mcp.servers]]
name = "homeai"
command = "/opt/homeai/bin/homeai-mcp"

[[mcp.servers]]
name = "memory"
command = "/opt/homeai/bin/homeai-memory-mcp"

[mcp_bundles.homeai-tools]
servers = ["homeai"]

[mcp_bundles.memory-admin]
servers = ["memory"]

[mcp_bundles.ha-coding]
servers = ["ha"]

[agents.local]
mcp_bundles = ["homeai-tools"]

[agents.builder]
mcp_bundles = ["memory-admin", "ha-coding"]
"""


class TestVoiceIsolation:
    """The voice agent answers anyone within earshot: it must never reach the
    memory tools or Home Assistant's full API (plan §9)."""

    def test_coding_agents_may_have_them(self) -> None:
        assert Z.voice_leaks(tomllib.loads(SPLIT)) == []

    @pytest.mark.parametrize("bundle,server", [("memory-admin", "memory"), ("ha-coding", "ha")])
    def test_voice_bundle_agent_may_not(self, bundle: str, server: str) -> None:
        text = SPLIT.replace('[agents.local]\nmcp_bundles = ["homeai-tools"]',
                             f'[agents.local]\nmcp_bundles = ["homeai-tools", "{bundle}"]')
        (leak,) = Z.voice_leaks(tomllib.loads(text))
        assert f"voice agent 'local'" in leak and server in leak
        with pytest.raises(Z.ConfigError, match="coding-only"):
            merge(text)

    def test_voice_risk_profile_agent_may_not(self) -> None:
        text = SPLIT + f'\n[agents.jarvis]\nrisk_profile = "{Z.RISK_PROFILE}"\n' \
                       'mcp_bundles = ["memory-admin"]\n'
        (leak,) = Z.voice_leaks(tomllib.loads(text))
        assert "'jarvis'" in leak and "memory" in leak

    def test_bundle_that_mixes_servers_is_caught(self) -> None:
        text = SPLIT.replace('[mcp_bundles.homeai-tools]\nservers = ["homeai"]',
                             '[mcp_bundles.homeai-tools]\nservers = ["homeai", "memory"]')
        assert len(Z.voice_leaks(tomllib.loads(text))) == 1

    def test_cli_check(self, tmp_path: Path, capsys) -> None:
        cfg = tmp_path / "config.toml"
        cfg.write_text(SPLIT)
        assert Z.main(["--config", str(cfg), "--check"]) == 0
        cfg.write_text(SPLIT.replace('["homeai-tools"]', '["homeai-tools", "memory-admin"]'))
        assert Z.main(["--config", str(cfg), "--check"]) == 2
        assert "memory" in capsys.readouterr().err
        cfg.write_text("[broken")
        assert Z.main(["--config", str(cfg), "--check"]) == 2

    def test_cli_needs_repo_unless_checking(self, tmp_path: Path) -> None:
        with pytest.raises(SystemExit):
            Z.main(["--config", str(tmp_path / "c.toml")])


def test_a_fresh_install_names_one_voice_model_everywhere(tmp_path):
    """assess (what install.sh creates), zc_config (what ZeroClaw asks for), voice_model's default and the
    Modelfile must agree: with one model held by Ollama, a mismatch evicts Jarvis's model (2026-11-10)."""
    from homeai import voice_model
    from homeai.install import assess
    repo = Path(__file__).resolve().parents[1]
    config = tmp_path / "config.toml"
    config.write_text(merge("").text)
    section = voice_model.provider_section(tomllib.loads(config.read_text()))
    assert voice_model.current(config) == Z.OLLAMA_MODEL == assess.VOICE_MODEL == voice_model.DEFAULT
    assert section["provider_extra"] == voice_model.NO_REASONING     # thinking off
    modelfile = (repo / "deploy" / "ollama" / f"{assess.VOICE_MODEL}.Modelfile").read_text()
    assert f"FROM {assess.VOICE_BASE_MODEL}\n" in modelfile and "PARAMETER num_ctx 16384\n" in modelfile
