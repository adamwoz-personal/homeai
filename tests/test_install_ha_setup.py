# Copyright (c) 2026 Adam Wosotowsky
# SPDX-License-Identifier: BUSL-1.1
# Licensed under the Business Source License 1.1; see LICENSE at the
# repository root. Converts to Apache-2.0 on 2030-09-15.
"""./install.sh --home-assistant: Docker checks, container start, token
capture. Every side effect is injected; the token must never be printed."""

from __future__ import annotations

import stat
import subprocess

import pytest

from homeai import ha_client
from homeai.ha_client import HAAuthError, HAUnavailable
from homeai.install import ha_setup as hs
from tests.fake_ha import FakeHA

TOKEN = "s3cret-token-value"


class Runner:
    """Fake subprocess.run: `ok` maps a command prefix to a return code."""

    def __init__(self, ok=None):
        self.ok = ok or {}
        self.cmds: list[list[str]] = []

    def __call__(self, cmd, **kw):
        self.cmds.append(list(cmd))
        for prefix, rc in self.ok.items():
            if " ".join(cmd).startswith(prefix):
                return subprocess.CompletedProcess(cmd, rc)
        return subprocess.CompletedProcess(cmd, 0)


def make(tmp_path, runner=None, **kw):
    out: list[str] = []
    kw.setdefault("which", lambda name: "/usr/bin/docker")
    setup = hs.Setup(env_path=tmp_path / "ha.env", run=runner or Runner(), say=out.append,
                     lan_url=lambda: "http://10.0.0.5:8123", **kw)
    return setup, out


# -- docker -------------------------------------------------------------

def test_docker_ready(tmp_path):
    setup, _ = make(tmp_path)
    assert setup.docker_state() == "ready"


def test_docker_missing(tmp_path):
    setup, _ = make(tmp_path, which=lambda n: None)
    assert setup.docker_state() == "missing"


def test_docker_not_in_group(tmp_path):
    setup, _ = make(tmp_path, Runner({"docker info": 1, "sg docker": 1}))
    assert setup.docker_state() == "no-group"


def test_docker_usable_through_sg_counts_as_ready(tmp_path):
    setup, _ = make(tmp_path, Runner({"docker info": 1}))
    assert setup.docker_state() == "ready"


def test_missing_docker_prints_commands_without_sudo(tmp_path):
    runner = Runner()
    setup, out = make(tmp_path, runner, which=lambda n: None)
    assert setup.ensure_docker() is False
    text = "\n".join(out)
    assert hs.DOCKER_INSTALL in text and hs.DOCKER_GROUP in text and "--sudo" in text
    assert not any(c[0] == "bash" for c in runner.cmds)     # nothing ran


def test_missing_docker_with_sudo_runs_them(tmp_path):
    state = iter(["missing", "ready"])
    setup, out = make(tmp_path, use_sudo=True)
    setup.docker_state = lambda: next(state)
    assert setup.ensure_docker() is True
    ran = [c[2] for c in setup._run.cmds if c[0] == "bash"]
    assert len(ran) == 2 and ran[0].startswith("sudo apt-get install")


def test_sudo_but_session_still_cannot_use_docker(tmp_path):
    setup, out = make(tmp_path, use_sudo=True)
    setup.docker_state = lambda: "no-group"
    assert setup.ensure_docker() is False
    assert "log out and in" in "\n".join(out)


def test_sudo_command_failure_stops(tmp_path):
    setup, out = make(tmp_path, Runner({"bash": 1}), use_sudo=True)
    setup.docker_state = lambda: "missing"
    assert setup.ensure_docker() is False
    assert "That failed" in "\n".join(out)


def test_dry_run_with_sudo_runs_nothing(tmp_path):
    runner = Runner()
    setup, out = make(tmp_path, runner, dry_run=True, use_sudo=True, which=lambda n: None)
    assert setup.ensure_docker() is True
    assert "would run" in "\n".join(out)
    assert not any(c[0] == "bash" for c in runner.cmds)


# -- container ----------------------------------------------------------

def test_start_container(tmp_path):
    setup, _ = make(tmp_path)
    assert setup.start_container() is True
    assert setup._run.cmds[-1] == [str(hs.CONTAINER_SCRIPT), "start"]


def test_start_container_failure_and_missing_script(tmp_path):
    setup, _ = make(tmp_path, Runner({str(hs.CONTAINER_SCRIPT): 1}))
    assert setup.start_container() is False

    def boom(*a, **k):
        raise FileNotFoundError("nope")

    setup, out = make(tmp_path, boom)
    assert setup.start_container() is False and "Couldn't start" in "\n".join(out)


# -- token --------------------------------------------------------------

def test_write_env_is_600_and_complete(tmp_path):
    path = tmp_path / "sub" / "ha.env"
    hs.write_env_file(path, "http://x:8123", TOKEN)
    assert path.read_text() == f"HA_URL=http://x:8123\nHA_TOKEN={TOKEN}\n"
    assert stat.S_IMODE(path.stat().st_mode) == 0o600
    assert not list(path.parent.glob("*.tmp"))


@pytest.mark.parametrize("bad", ["", "two words", "tab\there"])
def test_write_env_rejects_bad_tokens(tmp_path, bad):
    with pytest.raises(ValueError):
        hs.write_env_file(tmp_path / "ha.env", "http://x", bad)
    assert not (tmp_path / "ha.env").exists()


def test_collect_token_saves_valid_token_without_echoing_it(tmp_path):
    setup, out = make(tmp_path, ask_token=lambda: f"  {TOKEN}\n", validate=lambda u, t: None)
    assert setup.collect_token() is True
    assert ha_client.load_settings(tmp_path / "ha.env").token == TOKEN
    assert TOKEN not in "\n".join(out)


def test_collect_token_retries_then_gives_up(tmp_path):
    def reject(url, token):
        raise HAAuthError("no")

    setup, out = make(tmp_path, ask_token=lambda: "bad", validate=reject)
    assert setup.collect_token() is False
    assert "Three bad tokens" in "\n".join(out)
    assert not (tmp_path / "ha.env").exists()


def test_collect_token_second_try_works(tmp_path):
    answers = iter(["bad", TOKEN])

    def validate(url, token):
        if token == "bad":
            raise HAAuthError("no")

    setup, _ = make(tmp_path, ask_token=lambda: next(answers), validate=validate)
    assert setup.collect_token() is True


def test_collect_token_ha_down_is_explained(tmp_path):
    def down(url, token):
        raise HAUnavailable("can't reach")

    setup, out = make(tmp_path, ask_token=lambda: TOKEN, validate=down)
    assert setup.collect_token() is False
    assert "ha_container.sh status" in "\n".join(out)
    assert not (tmp_path / "ha.env").exists()


def test_no_token_entered_non_interactive(tmp_path):
    setup, out = make(tmp_path, ask_token=lambda: None)
    assert setup.collect_token() is False
    assert "re-run" in "\n".join(out)


def test_prompt_token_prefers_environment(monkeypatch):
    monkeypatch.setenv("HOMEAI_HA_TOKEN", TOKEN)
    assert hs._prompt_token() == TOKEN
    monkeypatch.delenv("HOMEAI_HA_TOKEN")
    monkeypatch.setattr("sys.stdin.isatty", lambda: False, raising=False)
    assert hs._prompt_token() is None


# -- whole flow ---------------------------------------------------------

@pytest.fixture
def healthy(monkeypatch):
    monkeypatch.setattr(ha_client, "HAClient", lambda settings: FakeHA())


def test_full_flow(tmp_path, healthy):
    setup, out = make(tmp_path, ask_token=lambda: TOKEN, validate=lambda u, t: None)
    assert setup.run() == 0
    text = "\n".join(out)
    assert TOKEN not in text
    assert "Onboarding" in text and "http://10.0.0.5:8123" in text
    assert "Hue" in text and "Alexa" in text and "Ring" in text
    assert (tmp_path / "ha.env").exists()


def test_already_configured_skips_docker_and_prompts(tmp_path, healthy):
    hs.write_env_file(tmp_path / "ha.env", "http://x", TOKEN)
    runner = Runner()
    setup, out = make(tmp_path, runner, ask_token=lambda: pytest.fail("asked for a token"))
    assert setup.run() == 0
    assert runner.cmds == []
    assert "already exists" in "\n".join(out)


def test_flow_stops_when_docker_is_missing(tmp_path):
    setup, out = make(tmp_path, which=lambda n: None)
    assert setup.run() == 1
    assert not (tmp_path / "ha.env").exists()


def test_flow_stops_when_container_will_not_start(tmp_path):
    setup, out = make(tmp_path, Runner({str(hs.CONTAINER_SCRIPT): 1}))
    assert setup.run() == 1
    assert "ha_container.sh logs" in "\n".join(out)


def test_flow_without_token_returns_1_but_lists_next_steps(tmp_path):
    setup, out = make(tmp_path, ask_token=lambda: None)
    assert setup.run() == 1
    assert "Onboarding" in "\n".join(out)


def test_partial_setup_returns_1(tmp_path, monkeypatch):
    monkeypatch.setattr(ha_client, "HAClient", lambda s: FakeHA(devices=[]))
    setup, out = make(tmp_path, ask_token=lambda: TOKEN, validate=lambda u, t: None)
    assert setup.run() == 1
    assert "Alexa Devices" in "\n".join(out)


def test_dry_run_changes_nothing(tmp_path):
    runner = Runner()
    setup, out = make(tmp_path, runner, dry_run=True, ask_token=lambda: pytest.fail("asked"))
    assert setup.run() == 0
    assert not (tmp_path / "ha.env").exists()
    assert not any(c and c[0] == str(hs.CONTAINER_SCRIPT) for c in runner.cmds)
    assert "would run" in "\n".join(out)


def test_main_ctrl_c_is_clean(monkeypatch, capsys):
    monkeypatch.setattr(hs.Setup, "run", lambda self: (_ for _ in ()).throw(KeyboardInterrupt()))
    assert hs.main(["--dry-run"]) == 1
    assert "Cancelled" in capsys.readouterr().out
