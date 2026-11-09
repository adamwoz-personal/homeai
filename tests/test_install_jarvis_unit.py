# Copyright (c) 2026 Adam Wosotowsky
# SPDX-License-Identifier: BUSL-1.1
# Licensed under the Business Source License 1.1; see LICENSE at the
# repository root. Converts to Apache-2.0 on 2030-09-15.
"""install.sh picks the right service for Jarvis (deploy/jarvis_unit.sh, SANDBOX_PLAN).

Why: a fresh install used to get a user unit whose sandbox silently does
nothing on Ubuntu, without a word; and a re-run must never put the user unit
back beside a system one (two Jarvises on one microphone). This sources the
real function with run/info/note that only log.
"""
from __future__ import annotations

import subprocess
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]

HARNESS = r'''
set -euo pipefail
run()  { echo "RUN $*"; }
info() { echo "INFO $*"; }
note() { echo "NOTE $*"; }
. "$REPO/deploy/jarvis_unit.sh"
install_jarvis_unit
'''


@pytest.fixture
def case(tmp_path):
    def go(sudo: bool = False, system: bool = False, blocked: bool = True, user_unit: str | None = None):
        sysunit = tmp_path / "etc-homeai.service"
        if system:
            sysunit.write_text("[Service]\n")
        unit = tmp_path / "user-homeai.service"
        if user_unit is not None:
            unit.write_text(user_unit)
        flag = tmp_path / "userns"
        flag.write_text("1\n" if blocked else "0\n")
        env = {"PATH": "/usr/bin:/bin", "REPO": str(REPO), "ZC_DIR": "/zc", "UNIT": str(unit),
               "SYSTEM_UNIT": str(sysunit), "USE_SUDO": "1" if sudo else "0",
               "RENDERED": str(tmp_path / "rendered"), "USERNS_SYSCTL": str(flag)}
        proc = subprocess.run(["bash", "-c", HARNESS], env=env, capture_output=True, text=True, timeout=30)
        assert proc.returncode == 0, proc.stderr
        return proc.stdout.splitlines()
    return go


def test_sudo_installs_the_system_unit_and_no_user_unit(case):
    out = case(sudo=True)
    assert out == [f"RUN sudo {REPO}/tools/install_system_service.sh"]


def test_sudo_rerun_updates_the_system_unit(case):
    assert case(sudo=True, system=True) == [f"RUN sudo {REPO}/tools/install_system_service.sh"]


def test_without_sudo_a_system_unit_is_left_alone(case):
    out = case(system=True)
    assert len(out) == 1 and out[0].startswith("INFO Jarvis runs as a system unit")


def test_without_sudo_the_user_unit_comes_with_a_warning(case):
    out = case()
    assert any(line.startswith("RUN install -D -m 644") for line in out)
    assert "RUN systemctl --user enable homeai.service" in out
    assert out[-1].startswith("NOTE Jarvis's sandbox (read-only home) is NOT in force")


def test_no_warning_where_user_sandboxes_work(case):
    assert not [line for line in case(blocked=False) if line.startswith("NOTE")]


def test_an_up_to_date_user_unit_is_not_reinstalled(case, tmp_path):
    case()
    rendered = (tmp_path / "rendered").read_text()
    out = case(user_unit=rendered)
    assert "INFO service unit up to date" in out and not [line for line in out if "install -D" in line]
