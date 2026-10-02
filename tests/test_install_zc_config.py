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
