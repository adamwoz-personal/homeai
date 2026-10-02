# Copyright (c) 2026 Adam Wosotowsky
# SPDX-License-Identifier: BUSL-1.1
# Licensed under the Business Source License 1.1; see LICENSE at the
# repository root. Converts to Apache-2.0 on 2030-09-15.
"""Add the voice agent to a ZeroClaw ``config.toml`` without disturbing it.

A ZeroClaw config is the user's own file, often hand-edited and holding
encrypted secrets, so this never rewrites it. It only *appends* sections that
are absent, then re-parses the result and refuses to write anything that
fails to parse or lacks the voice agent. The previous file is kept as
``config.toml.bak.<epoch>``.

TOML forbids defining a table twice, so every section is checked against the
parsed existing config first. A section that already exists is left alone and
reported, which makes re-running the installer safe.

    python3 -m homeai.install.zc_config --config ~/.zeroclaw/config.toml \\
        --repo ~/src/homeai --num-ctx 16384 [--dry-run]

Standard library only (tomllib is 3.11+).
"""

from __future__ import annotations

import argparse
import shutil
import sys
import time
import tomllib
from dataclasses import dataclass, field
from pathlib import Path

DEFAULT_AGENT = "jarvis"
RISK_PROFILE = "homeai_voice"
PROVIDER = "homeai_voice"
MCP_SERVER = "homeai"
MCP_BUNDLE = "homeai-tools"
OLLAMA_MODEL = "llama31-voice"
OLLAMA_URI = "http://127.0.0.1:11434/v1"
SCHEMA_VERSION = 3


class ConfigError(Exception):
    """The existing config conflicts with the voice agent in a way we must not
    paper over (the user has to decide)."""


@dataclass
class MergeResult:
    text: str
    added: list[str] = field(default_factory=list)
    skipped: list[str] = field(default_factory=list)

    @property
    def changed(self) -> bool:
        return bool(self.added)


def _toml_list(items: list[str]) -> str:
    return "[" + ", ".join(f'"{i}"' for i in items) + "]"


def _sections(agent: str, repo: Path, home: Path, num_ctx: int) -> list[tuple[tuple[str, ...], str]]:
    """(key path that proves the section exists, TOML text) in write order."""
    mcp_command = repo / "bin" / "homeai-mcp"
    forbidden = ["/root", "/boot", "/proc", "/sys", "/etc", "/var", "/dev",
                 "~/.ssh", "~/.gnupg", "~/.aws", "~/.zeroclaw/config.toml", str(repo)]
    return [
        (("agents", agent), f"""
# ---------------------------------------------------------------------------
# Home AI voice agent (added by homeai's install.sh).
# Reachable by anyone within earshot of the microphone: a voice carries no
# authentication, so this agent is deliberately locked down. Do not widen it.
# ---------------------------------------------------------------------------
[agents.{agent}]
enabled = true
model_provider = "custom.{PROVIDER}"
risk_profile = "{RISK_PROFILE}"
mcp_bundles = ["{MCP_BUNDLE}"]
acp_enable_mcp = true
delegate_same_risk_profile = true

[agents.{agent}.identity]
format = "openclaw"

[agents.{agent}.memory]
backend = "sqlite"
"""),
        (("providers", "models", "custom", PROVIDER), f"""
# llama3.1:8b via a local Modelfile variant with a larger context; stock
# Ollama serves every model at 4096 tokens, which truncates the persona.
[providers.models.custom.{PROVIDER}]
model = "{OLLAMA_MODEL}"
uri = "{OLLAMA_URI}"
context_window = {num_ctx}
native_tools = true
temperature = 0.45
"""),
        (("risk_profiles", RISK_PROFILE), f"""
[risk_profiles.{RISK_PROFILE}]
# Also limits what the model is SHOWN: with every tool advertised, an 8B
# model hedged and searched instead of answering. MCP tools (homeai__weather,
# homeai__research) are admitted automatically.
# memory_recall is deliberately absent: ZeroClaw auto-saves every request and
# the model parroted old answers back when it could recall them.
allowed_tools = ["calculator"]
level = "readonly"
workspace_only = true
block_high_risk_commands = true
require_approval_for_medium_risk = true
allowed_commands = ["echo"]
allowed_roots = {_toml_list([str(home / ".zeroclaw" / "voice-scratch"), "/tmp"])}
forbidden_paths = {_toml_list(forbidden)}
auto_approve = ["calculator", "weather", "mcp", "mcp_tool", "homeai", "research"]
"""),
        (("mcp_bundles", MCP_BUNDLE), f"""
[mcp_bundles.{MCP_BUNDLE}]
servers = ["{MCP_SERVER}"]
"""),
        (("mcp", "servers", MCP_SERVER), f"""
[[mcp.servers]]
# Local weather and research tools; see homeai/mcp_server.py.
name = "{MCP_SERVER}"
transport = "stdio"
command = "{mcp_command}"
args = []
tool_timeout_secs = 45
"""),
    ]


def _has_path(data: dict, path: tuple[str, ...]) -> bool:
    node: object = data
    for key in path:
        if not isinstance(node, dict) or key not in node:
            return False
        node = node[key]
    return True


def _mcp_server(data: dict, name: str) -> dict | None:
    servers = data.get("mcp", {}).get("servers", [])
    return next((s for s in servers if isinstance(s, dict) and s.get("name") == name), None)


def merge(existing: str, *, repo: Path, home: Path, num_ctx: int,
          agent: str = DEFAULT_AGENT) -> MergeResult:
    """Return the config text with any missing voice-agent sections appended."""
    try:
        data = tomllib.loads(existing) if existing.strip() else {}
    except tomllib.TOMLDecodeError as exc:
        raise ConfigError(f"existing config does not parse, refusing to touch it: {exc}") from exc

    result = MergeResult(existing)
    parts: list[str] = []
    if not existing.strip():
        parts.append(f"schema_version = {SCHEMA_VERSION}\n")

    mcp = data.get("mcp", {})
    if isinstance(mcp, dict) and mcp.get("enabled") is False:
        raise ConfigError("[mcp] enabled = false in the existing config; the voice "
                          "agent's weather and research tools need MCP. Enable it "
                          "(or remove the line) and re-run.")
    if not (isinstance(mcp, dict) and "enabled" in mcp):
        if "mcp" in data:
            # Cannot add `enabled` to a table that is already defined without
            # rewriting the user's file; ZeroClaw's default would then apply.
            raise ConfigError("an [mcp] table exists without `enabled = true`; add that "
                              "line to it and re-run.")
        parts.append("\n[mcp]\nenabled = true\n")
        result.added.append("mcp")

    expected_cmd = str(repo / "bin" / "homeai-mcp")
    # An agent that already exists is the user's: its provider and risk
    # profile may be named anything, so adding ours would only leave unused
    # sections behind. Only the shared MCP server and bundle are ensured.
    agent_exists = _has_path(data, ("agents", agent))
    agent_owned = {("agents", agent), ("providers", "models", "custom", PROVIDER),
                   ("risk_profiles", RISK_PROFILE)}
    for path, text in _sections(agent, repo, home, num_ctx):
        label = ".".join(path)
        if agent_exists and path in agent_owned:
            result.skipped.append(f"{label} (agent '{agent}' already configured)")
            continue
        if path[:2] == ("mcp", "servers"):
            server = _mcp_server(data, MCP_SERVER)
            if server is not None:
                if server.get("command") != expected_cmd:
                    raise ConfigError(
                        f"an MCP server named '{MCP_SERVER}' already exists with command "
                        f"{server.get('command')!r}, not {expected_cmd!r}. Point it at this "
                        "checkout (or remove it) and re-run.")
                result.skipped.append(f"{label} (already present)")
                continue
        elif _has_path(data, path):
            result.skipped.append(f"{label} (already present)")
            continue
        parts.append(text)
        result.added.append(label)

    if not result.added:
        return result
    joined = existing.rstrip("\n") + ("\n" if existing.strip() else "") + "".join(parts)
    try:
        merged = tomllib.loads(joined)
    except tomllib.TOMLDecodeError as exc:  # a bug here, never the user's fault
        raise ConfigError(f"internal error: merged config would not parse: {exc}") from exc
    if not _has_path(merged, ("agents", agent)) or _mcp_server(merged, MCP_SERVER) is None:
        raise ConfigError("internal error: merged config lacks the voice agent")
    result.text = joined
    return result


def write_config(path: Path, result: MergeResult) -> Path | None:
    """Write atomically, keeping a timestamped backup. Returns the backup path."""
    backup = None
    if path.exists():
        backup = path.with_name(f"{path.name}.bak.{int(time.time())}")
        shutil.copy2(path, backup)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(result.text, encoding="utf-8")
    tmp.chmod(0o600)
    tmp.replace(path)
    return backup


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Add the homeai voice agent to ZeroClaw")
    parser.add_argument("--config", type=Path, default=Path.home() / ".zeroclaw" / "config.toml")
    parser.add_argument("--repo", type=Path, required=True)
    parser.add_argument("--num-ctx", type=int, default=16384)
    parser.add_argument("--agent", default=DEFAULT_AGENT)
    parser.add_argument("--dry-run", action="store_true", help="print, do not write")
    args = parser.parse_args(argv)

    existing = args.config.read_text(encoding="utf-8") if args.config.exists() else ""
    try:
        result = merge(existing, repo=args.repo.resolve(), home=Path.home(),
                       num_ctx=args.num_ctx, agent=args.agent)
    except ConfigError as exc:
        print(f"zeroclaw config: {exc}", file=sys.stderr)
        return 2
    for label in result.skipped:
        print(f"  left alone: {label}")
    if not result.changed:
        print("  zeroclaw config already has the voice agent; nothing to do")
        return 0
    if args.dry_run:
        print(f"  would add: {', '.join(result.added)} to {args.config}")
        return 0
    backup = write_config(args.config, result)
    print(f"  added: {', '.join(result.added)}" + (f" (backup: {backup})" if backup else ""))
    return 0


if __name__ == "__main__":
    sys.exit(main())
