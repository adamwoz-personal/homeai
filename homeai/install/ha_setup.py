# Copyright (c) 2026 Adam Wosotowsky
# SPDX-License-Identifier: BUSL-1.1
# Licensed under the Business Source License 1.1; see LICENSE at the
# repository root. Converts to Apache-2.0 on 2030-09-15.
"""Opt-in Home Assistant setup for install.sh (`./install.sh --home-assistant`).

    python -m homeai.install.ha_setup [--dry-run] [--sudo]

Steps, each skipped when already done: check Docker (print, or with --sudo
run, the install/usermod commands), start the Home Assistant container
(tools/ha/ha_container.sh), walk through the browser onboarding, take a
long-lived token for a non-admin `jarvis` user, validate it against HA and
write ~/.config/homeai/ha.env (mode 600), then run the connection check and
list the steps only a browser can do.

Never fatal for the installer: exit 0 = working, 1 = needs something from the
user (every reason is printed). The token is never echoed or logged. Set
HOMEAI_HA_TOKEN to supply it non-interactively.
"""

from __future__ import annotations

import argparse
import getpass
import os
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Callable

from .. import ha_client
from ..ha_client import HAAuthError, HAClient, HAError, HASettings

ROOT = Path(__file__).resolve().parents[2]
CONTAINER_SCRIPT = ROOT / "tools" / "ha" / "ha_container.sh"
DEFAULT_URL = "http://127.0.0.1:8123"
TOKEN_TRIES = 3

DOCKER_INSTALL = "sudo apt-get install -y docker.io"
DOCKER_GROUP = "sudo usermod -aG docker $USER"

ONBOARDING = (
    "Open {lan} in a browser and create the owner account (onboarding).",
    "Settings > People > Add person: create a user named `jarvis`, leave 'Administrator' OFF.",
    "Log in as jarvis, open your profile > Security > Long-lived access tokens > Create "
    "token, and copy it (HA shows it once).",
)

BROWSER_STEPS = (
    "Philips Hue: Settings > Devices & services > Add > Philips Hue; press the bridge button.",
    "Alexa: Add > Alexa Devices; sign in to Amazon and enter the authenticator code. Without "
    "it Jarvis can't set timers, announce or play music.",
    "Ring (optional): Add > Ring; sign in with 2FA. Jarvis announces rings.",
    "Phone alerts (optional): install the Home Assistant app on your phone and log in; "
    "Jarvis then also pushes doorbell rings to it.",
    "Coding tools (optional): Add > Model Context Protocol Server, then "
    "docs/RUNBOOK.md \"Home Assistant MCP for coding\".",
    "Quieter log (recommended once the above are added): "
    "python3 tools/ha/ha_quiet_logs.py (HA's Bluetooth errors are noise here).",
)


class Setup:
    """Side effects (commands, prompts, output) are injected so every branch is testable."""

    def __init__(self, *, dry_run: bool = False, use_sudo: bool = False,
                 env_path: Path | None = None, url: str = DEFAULT_URL,
                 run: Callable[..., subprocess.CompletedProcess] = subprocess.run,
                 which: Callable[[str], str | None] = shutil.which,
                 ask_token: Callable[[], str | None] | None = None,
                 validate: Callable[[str, str], None] | None = None,
                 say: Callable[[str], None] = print,
                 lan_url: Callable[[], str] | None = None) -> None:
        self.dry_run = dry_run
        self.use_sudo = use_sudo
        self.env_path = env_path or ha_client.DEFAULT_ENV
        self.url = url
        self._run = run
        self._which = which
        self._ask_token = ask_token or _prompt_token
        self._validate = validate or _validate_token
        self.say = say
        self._lan_url = lan_url or _lan_url

    # -- docker ------------------------------------------------------------

    def _ok(self, *cmd: str) -> bool:
        try:
            return self._run(list(cmd), capture_output=True, timeout=30).returncode == 0
        except (OSError, subprocess.SubprocessError):
            return False

    def docker_state(self) -> str:
        """ready | no-group | missing."""
        if not self._which("docker"):
            return "missing"
        if self._ok("docker", "info") or self._ok("sg", "docker", "-c", "docker info"):
            return "ready"  # ha_container.sh falls back to `sg docker` for a stale session
        return "no-group"

    def ensure_docker(self) -> bool:
        state = self.docker_state()
        if state == "ready":
            self.say("   Docker is ready.")
            return True
        commands = ([DOCKER_INSTALL] if state == "missing" else []) + [DOCKER_GROUP]
        if not self.use_sudo:
            self.say("   Docker "
                     + ("isn't installed." if state == "missing"
                        else "isn't usable by this user (not in the docker group).")
                     + " Run these, log out and in, then re-run with --home-assistant")
            for command in commands:
                self.say(f"       {command}")
            self.say("   (or pass --sudo and the installer runs them for you)")
            return False
        for command in commands:
            self.say(f"   $ {command}" if not self.dry_run else f"   would run: {command}")
            if self.dry_run:
                continue
            if self._run(["bash", "-c", command.replace("$USER", os.environ.get("USER", ""))]).returncode:
                self.say(f"   That failed: {command}")
                return False
        if self.dry_run:
            return True
        if self.docker_state() != "ready":
            self.say("   Docker is installed, but this session can't use it yet: log out and in, "
                     "then re-run with --home-assistant.")
            return False
        return True

    # -- container ---------------------------------------------------------

    def start_container(self) -> bool:
        if self.dry_run:
            self.say(f"   would run: {CONTAINER_SCRIPT} start")
            return True
        self.say(f"   $ {CONTAINER_SCRIPT} start   (first run pulls ~2 GB)")
        try:
            return self._run([str(CONTAINER_SCRIPT), "start"]).returncode == 0
        except OSError as exc:
            self.say(f"   Couldn't start Home Assistant: {exc}")
            return False

    # -- token -------------------------------------------------------------

    def configured(self) -> bool:
        return ha_client.is_configured(self.env_path)

    def write_env(self, token: str) -> None:
        write_env_file(self.env_path, self.url, token)
        self.say(f"   Saved the token to {self.env_path} (mode 600).")

    def collect_token(self) -> bool:
        for _ in range(TOKEN_TRIES):
            token = (self._ask_token() or "").strip()
            if not token:
                self.say("   No token entered. Create one (steps above) and re-run "
                         "./install.sh --home-assistant, or put HA_URL= / HA_TOKEN= "
                         f"in {self.env_path} yourself.")
                return False
            try:
                self._validate(self.url, token)
            except HAAuthError:
                self.say("   Home Assistant rejected that token. Try again.")
                continue
            except HAError as exc:
                self.say(f"   Couldn't check the token ({exc}). Is Home Assistant up? "
                         "tools/ha/ha_container.sh status")
                return False
            self.write_env(token)
            return True
        self.say("   Three bad tokens; stopping. Re-run when you have a fresh one.")
        return False

    # -- whole flow --------------------------------------------------------

    def run(self) -> int:
        self.say(f"   Home Assistant setup{' (dry run)' if self.dry_run else ''}")
        if self.configured() and not self.dry_run:
            self.say(f"   {self.env_path} already exists; checking it instead of redoing setup.")
        else:
            if not self.ensure_docker():
                return 1
            if not self.start_container():
                self.say("   Home Assistant didn't start. See: tools/ha/ha_container.sh logs")
                return 1
            if self.dry_run:
                self.say("   would walk through onboarding and ask for the token")
                self._print_browser_steps()
                return 0
            if not self.configured():
                self.say("   Onboarding (browser):")
                for n, step in enumerate(ONBOARDING, 1):
                    self.say(f"     {n}. " + step.format(lan=self._lan_url()))
                if not self.collect_token():
                    return 1
        code, message = ha_client.check(self.env_path)
        self.say(f"   {message}")
        self._print_browser_steps()
        return 0 if code == 0 else 1

    def _print_browser_steps(self) -> None:
        self.say("   Still to do in the browser (only you can):")
        for step in BROWSER_STEPS:
            self.say(f"     - {step}")


def write_env_file(path: Path, url: str, token: str) -> None:
    """Write HA_URL/HA_TOKEN atomically, mode 600 from the first byte."""
    if not token or any(c.isspace() for c in token):
        raise ValueError("the token must be one non-empty word")
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    fd = os.open(tmp, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
    with os.fdopen(fd, "w") as handle:
        handle.write(f"HA_URL={url}\nHA_TOKEN={token}\n")
    os.replace(tmp, path)
    os.chmod(path, 0o600)


def _prompt_token() -> str | None:
    token = os.environ.get("HOMEAI_HA_TOKEN")
    if token:
        return token
    if not sys.stdin.isatty():
        return None
    return getpass.getpass("   Paste the long-lived token (hidden): ")


def _validate_token(url: str, token: str) -> None:
    HAClient(HASettings(url, token), timeout=10).states()


def _lan_url() -> str:
    try:
        out = subprocess.run(["hostname", "-I"], capture_output=True, text=True, timeout=5).stdout
        ip = out.split()[0]
    except (OSError, subprocess.SubprocessError, IndexError):
        ip = "127.0.0.1"
    return f"http://{ip}:8123"


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--dry-run", action="store_true", help="print what would happen")
    ap.add_argument("--sudo", action="store_true", help="run the apt/usermod commands")
    args = ap.parse_args(argv)
    try:
        return Setup(dry_run=args.dry_run, use_sudo=args.sudo).run()
    except KeyboardInterrupt:
        print("\n   Cancelled; nothing half-written (re-run any time).")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
