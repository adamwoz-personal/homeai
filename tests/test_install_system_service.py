# Copyright (c) 2026 Adam Wosotowsky
# SPDX-License-Identifier: BUSL-1.1
# Licensed under the Business Source License 1.1; see LICENSE at the
# repository root. Converts to Apache-2.0 on 2030-09-15.
"""tools/install_system_service.sh moves Jarvis to a system unit safely (SANDBOX_PLAN step 4).

Why: the switch touches the live service. Two Jarvises on one microphone, a
user unit left enabled beside the system one, or a failed start with no way
back would each break voice. This runs the real script with fake systemctl,
runuser and loginctl (they log their arguments) and a temporary home and unit dir.
"""
from __future__ import annotations

import getpass
import os
import stat
import subprocess
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
SCRIPT = REPO / "tools" / "install_system_service.sh"

FAKE_SYSTEMCTL = """#!/bin/bash
echo "systemctl $*" >> "$FAKE_LOG"
if [[ "$1" == is-active ]]; then [[ "${FAKE_ACTIVE:-1}" == 1 ]]; exit; fi
exit 0
"""
FAKE_RUNUSER = """#!/bin/bash
echo "runuser $2" >> "$FAKE_LOG"
while [[ "$1" != -- ]]; do shift; done; shift
exec "$@"
"""
FAKE_LOGINCTL = """#!/bin/bash
echo "loginctl $*" >> "$FAKE_LOG"
[[ "$1" == show-user ]] && echo "${FAKE_LINGER:-yes}"
exit 0
"""


@pytest.fixture
def box(tmp_path):
    bin_ = tmp_path / "bin"
    bin_.mkdir()
    for name, body in (("systemctl", FAKE_SYSTEMCTL), ("runuser", FAKE_RUNUSER), ("loginctl", FAKE_LOGINCTL)):
        (bin_ / name).write_text(body)
        (bin_ / name).chmod(stat.S_IRWXU)
    home = tmp_path / "home"
    user_unit = home / ".config/systemd/user/homeai.service"
    user_unit.parent.mkdir(parents=True)
    user_unit.write_text("[Service]\n# the user unit\n")
    rule = tmp_path / "50-homeai.rules"
    rule.write_text("// rule\n")
    env = {**os.environ, "PATH": f"{bin_}:{os.environ['PATH']}", "FAKE_LOG": str(tmp_path / "log"),
           "SUDO_USER": getpass.getuser(), "HOMEAI_SKIP_ROOT_CHECK": "1", "HOMEAI_SETTLE_S": "0",
           "HOMEAI_UNIT_DIR": str(tmp_path / "etc"), "HOMEAI_POLKIT_RULE": str(rule),
           "HOMEAI_OWNER_HOME": str(home)}

    class Box:
        unit = tmp_path / "etc" / "homeai.service"

        def __init__(self):
            self.env, self.user_unit, self.rule = env, user_unit, rule

        def run(self, *args, **extra):
            return subprocess.run(["bash", str(SCRIPT), *args], env={**self.env, **extra},
                                  capture_output=True, text=True, timeout=60)

        def log(self) -> list[str]:
            path = Path(env["FAKE_LOG"])
            return path.read_text().splitlines() if path.exists() else []

    return Box()


def test_switch_stops_the_user_unit_before_starting_the_system_one(box):
    proc = box.run()
    assert proc.returncode == 0, proc.stderr
    log = box.log()
    user_off = log.index("systemctl --user disable --now homeai.service")
    system_on = log.index("systemctl enable --now homeai.service")
    assert user_off < system_on
    assert not box.user_unit.exists() and box.user_unit.with_name("homeai.service.moved-to-system").exists()


def test_the_rendered_unit_runs_as_the_owner_with_the_session(box):
    assert box.run().returncode == 0
    text = box.unit.read_text()
    uid = os.getuid()
    assert "@" not in text.replace(f"user@{uid}.service", "")
    assert f"User={getpass.getuser()}" in text and "ProtectHome=read-only" in text
    assert f"Environment=XDG_RUNTIME_DIR=/run/user/{uid}" in text
    assert f"-/run/user/{uid}/homeai" in text and "WantedBy=multi-user.target" in text
    assert f"Wants=user@{uid}.service" in text


def test_a_system_unit_that_does_not_stay_up_is_rolled_back(box):
    proc = box.run(FAKE_ACTIVE="0")
    assert proc.returncode == 1 and "going back to the user unit" in proc.stderr
    assert not box.unit.exists() and box.user_unit.exists()
    log = box.log()
    assert log.index("systemctl disable --now homeai.service") < log.index(
        "systemctl --user enable --now homeai.service")


def test_undo_brings_the_user_unit_back(box):
    assert box.run().returncode == 0
    proc = box.run("--undo")
    assert proc.returncode == 0, proc.stderr
    assert not box.unit.exists() and box.user_unit.read_text() == "[Service]\n# the user unit\n"
    assert box.log()[-1] == "systemctl --user enable --now homeai.service"


def test_no_polkit_rule_means_nothing_is_touched(box):
    box.rule.unlink()
    proc = box.run()
    assert proc.returncode == 1 and "polkit rule" in proc.stderr
    assert box.log() == [] and box.user_unit.exists() and not box.unit.exists()


def test_dry_run_changes_nothing(box):
    proc = box.run("--dry-run")
    assert proc.returncode == 0 and "dry run" in proc.stdout
    assert box.user_unit.exists() and not box.unit.exists()
    assert not [line for line in box.log() if not line.startswith("loginctl show-user")]


def test_linger_is_turned_on_when_off(box):
    assert box.run(FAKE_LINGER="no").returncode == 0
    assert f"loginctl enable-linger {getpass.getuser()}" in box.log()


def test_install_sh_leaves_the_user_unit_alone_once_jarvis_is_a_system_unit():
    text = (REPO / "install.sh").read_text()
    guard = text.index('if [[ -f "$SYSTEM_UNIT" ]]; then')
    assert guard < text.index("run systemctl --user enable homeai.service")
    assert guard < text.index('run install -D -m 644 "$RENDERED" "$UNIT"')


@pytest.mark.skipif(not Path("/usr/bin/systemd-analyze").exists(), reason="no systemd-analyze")
def test_systemd_accepts_the_rendered_unit(box):
    assert box.run().returncode == 0
    proc = subprocess.run(["systemd-analyze", "verify", str(box.unit)], capture_output=True, text=True, timeout=60)
    mine = [line for line in proc.stderr.splitlines() if "homeai.service" in line]
    assert proc.returncode == 0 and not mine, proc.stderr   # unknown keys warn but exit 0
