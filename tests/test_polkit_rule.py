# Copyright (c) 2026 Adam Wosotowsky
# SPDX-License-Identifier: BUSL-1.1
# Licensed under the Business Source License 1.1; see LICENSE at the
# repository root. Converts to Apache-2.0 on 2030-09-15.
"""The polkit rule for the system homeai.service grants exactly what it says (SANDBOX_PLAN step 2).

Why: the rule gives a password-free systemctl to whoever matches it, so a typo
widens it silently. This runs the real rule file in node against a fake polkit
object and checks every grant and refusal.
"""
from __future__ import annotations

import json
import shutil
import subprocess
from pathlib import Path

import pytest

RULE = Path(__file__).resolve().parents[1] / "deploy" / "polkit" / "50-homeai.rules.in"
NODE = shutil.which("node")

HARNESS = """
const fs = require("fs");
let rule = null;
const polkit = {Result: {YES: "yes", NO: "no", NOT_HANDLED: null}, addRule: f => { rule = f; }};
eval(fs.readFileSync(process.argv[1], "utf8").replace(/@USER@/g, "adam"));
const out = JSON.parse(process.argv[2]).map(([id, user, unit, verb]) => {
  const details = {unit: unit, verb: verb};
  return rule({id: id, lookup: k => details[k]}, {user: user});
});
console.log(JSON.stringify(out));
"""

MANAGE = "org.freedesktop.systemd1.manage-units"


def decide(*cases: tuple[str, str, str, str]) -> list:
    proc = subprocess.run([NODE, "-e", HARNESS, str(RULE), json.dumps(cases)],
                          capture_output=True, text=True, timeout=30, check=True)
    return json.loads(proc.stdout)


pytestmark = pytest.mark.skipif(NODE is None, reason="node not installed")


def test_adam_may_start_stop_and_restart_homeai():
    assert decide(*[(MANAGE, "adam", "homeai.service", v) for v in ("start", "stop", "restart")]) == ["yes"] * 3


def test_nothing_else_is_granted():
    assert decide(
        (MANAGE, "ani", "homeai.service", "restart"),           # another user
        (MANAGE, "adam", "ssh.service", "restart"),             # another unit
        (MANAGE, "adam", "homeai.service", "reload"),           # another verb
        (MANAGE, "adam", "homeai.service", ""),
        ("org.freedesktop.systemd1.manage-unit-files", "adam", "homeai.service", "restart"),  # enable/disable
    ) == [None] * 5


def test_the_template_names_its_user_placeholder():
    assert "@USER@" in RULE.read_text()
