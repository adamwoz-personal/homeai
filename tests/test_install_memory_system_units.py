# Copyright (c) 2026 Adam Wosotowsky
# SPDX-License-Identifier: BUSL-1.1
# Licensed under the Business Source License 1.1; see LICENSE at the
# repository root. Converts to Apache-2.0 on 2030-09-15.
"""tools/install_memory_system_units.sh moves the memory timers to system units safely.

Why: a backup running twice (user and system timer both on), a system unit
whose sandbox breaks the backup with no way back, or install.sh putting the
user units back afterwards would each cost backups silently. Runs the real
script with fake systemctl and runuser (they log their arguments) and a
temporary home and unit dir.
"""
from __future__ import annotations

import getpass
import os
import stat
import subprocess
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
SCRIPT = REPO / "tools" / "install_memory_system_units.sh"
NAMES = ("homeai-memory-backup", "homeai-memory-maintain")

FAKE_SYSTEMCTL = """#!/bin/bash
echo "systemctl $*" >> "$FAKE_LOG"
if [[ "$1" == start ]]; then exit "${FAKE_START_RC:-0}"; fi
exit 0
"""
FAKE_RUNUSER = """#!/bin/bash
echo "runuser $2" >> "$FAKE_LOG"
while [[ "$1" != -- ]]; do shift; done; shift
exec "$@"
"""


@pytest.fixture
def box(tmp_path):
    bin_ = tmp_path / "bin"
    bin_.mkdir()
    for name, body in (("systemctl", FAKE_SYSTEMCTL), ("runuser", FAKE_RUNUSER)):
        (bin_ / name).write_text(body)
        (bin_ / name).chmod(stat.S_IRWXU)
    home = tmp_path / "home"
    user_dir = home / ".config/systemd/user"
    user_dir.mkdir(parents=True)
    for n in NAMES:
        for kind in ("service", "timer"):
            (user_dir / f"{n}.{kind}").write_text(f"# user {n}.{kind}\n")
    env = {**os.environ, "PATH": f"{bin_}:{os.environ['PATH']}", "FAKE_LOG": str(tmp_path / "log"),
           "SUDO_USER": getpass.getuser(), "HOMEAI_SKIP_ROOT_CHECK": "1",
           "HOMEAI_UNIT_DIR": str(tmp_path / "etc"), "HOMEAI_OWNER_HOME": str(home)}

    class Box:
        etc = tmp_path / "etc"

        def __init__(self):
            self.env, self.user_dir, self.home = env, user_dir, home

        def run(self, *args, **extra):
            return subprocess.run(["bash", str(SCRIPT), *args], env={**self.env, **extra},
                                  capture_output=True, text=True, timeout=60)

        def log(self) -> list[str]:
            path = Path(env["FAKE_LOG"])
            return path.read_text().splitlines() if path.exists() else []

    return Box()


TIMERS = "homeai-memory-backup.timer homeai-memory-maintain.timer"


def test_user_timers_stop_before_the_system_ones_start_and_a_backup_proves_it(box):
    proc = box.run()
    assert proc.returncode == 0, proc.stderr
    log = box.log()
    assert log.index(f"systemctl --user disable --now {TIMERS}") < log.index(f"systemctl enable --now {TIMERS}")
    assert log[-1] == "systemctl start homeai-memory-backup.service"
    for n in NAMES:
        assert not (box.user_dir / f"{n}.timer").exists()
        assert (box.user_dir / f"{n}.timer.moved-to-system").exists()
        assert (box.etc / f"{n}.timer").exists()


def test_the_units_run_as_the_owner_with_the_sandbox(box):
    assert box.run().returncode == 0
    for n in NAMES:
        text = (box.etc / f"{n}.service").read_text()
        assert "@" not in text and "%h" not in text
        assert f"User={getpass.getuser()}" in text and f"Environment=HOME={box.home}" in text
        assert "ProtectSystem=strict" in text and "ProtectHome=read-only" in text
    assert f"ReadWritePaths={box.home}/.local/share/homeai" in (box.etc / "homeai-memory-backup.service").read_text()


def test_a_failed_proof_backup_goes_back_to_the_user_units(box):
    proc = box.run(FAKE_START_RC="1")
    assert proc.returncode == 1 and "going back to the user units" in proc.stderr
    for n in NAMES:
        assert not (box.etc / f"{n}.service").exists()
        assert (box.user_dir / f"{n}.service").read_text() == f"# user {n}.service\n"
    assert box.log()[-1] == f"systemctl --user enable --now {TIMERS}"


def test_undo_brings_the_user_units_back(box):
    assert box.run().returncode == 0
    proc = box.run("--undo")
    assert proc.returncode == 0, proc.stderr
    assert not list(box.etc.glob("homeai-memory-*"))
    assert (box.user_dir / "homeai-memory-maintain.timer").read_text() == "# user homeai-memory-maintain.timer\n"


def test_dry_run_changes_nothing(box):
    proc = box.run("--dry-run")
    assert proc.returncode == 0 and "dry run" in proc.stdout
    assert not box.etc.exists() and (box.user_dir / "homeai-memory-backup.timer").exists()
    assert box.log() == []


def test_install_sh_leaves_system_memory_units_alone():
    text = (REPO / "install.sh").read_text()
    guard = text.index('if [[ -f "$MEMORY_SYSTEM_TIMER" ]]; then')
    assert guard < text.index("for f in homeai-memory-backup.service homeai-memory-backup.timer")


@pytest.mark.skipif(not Path("/usr/bin/systemd-analyze").exists(), reason="no systemd-analyze")
def test_systemd_accepts_the_rendered_units(box):
    assert box.run().returncode == 0
    units = [str(box.etc / f"{n}.service") for n in NAMES]
    proc = subprocess.run(["systemd-analyze", "verify", *units], capture_output=True, text=True, timeout=60)
    mine = [line for line in proc.stderr.splitlines() if "homeai-memory" in line]
    assert proc.returncode == 0 and not mine, proc.stderr
