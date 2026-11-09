# Copyright (c) 2026 Adam Wosotowsky
# SPDX-License-Identifier: BUSL-1.1
# Licensed under the Business Source License 1.1; see LICENSE at the
# repository root. Converts to Apache-2.0 on 2030-09-15.
"""Is Jarvis a system service or a user service? (SANDBOX_PLAN step 3)

Why: a user unit's ProtectHome is silently skipped on this Ubuntu (AppArmor
blocks the user namespace systemd needs), so homeai.service moves to a system
unit with User=adam. Only that one unit moves: the coder, ComfyUI and the
picture window stay user units. Everything that starts, stops or restarts
Jarvis builds its systemctl command here, so both layouts work.

The system unit file decides: /etc/systemd/system/homeai.service present means
system. HOMEAI_SERVICE_SCOPE=system|user overrides it (tests pin "user").
"""
from __future__ import annotations

import os
from pathlib import Path

ENV = "HOMEAI_SERVICE_SCOPE"
SYSTEM_UNIT_FILE = Path("/etc/systemd/system/homeai.service")
VOICE_UNITS = frozenset({"homeai", "homeai.service"})


def scope(unit_file: Path = SYSTEM_UNIT_FILE) -> str:
    """'system' or 'user' for homeai.service."""
    forced = os.environ.get(ENV, "")
    if forced in ("system", "user"):
        return forced
    return "system" if unit_file.exists() else "user"


def _units(args: tuple[str, ...]) -> list[str]:
    return [a for a in args if not a.startswith("-") and (a.endswith(".service") or a in VOICE_UNITS)]


def systemctl(*args: str) -> list[str]:
    """The systemctl argv for these args: system scope only when they name Jarvis and Jarvis is a system unit.

    One call may not mix Jarvis with user units: they live in different managers.
    """
    units = _units(args)
    voice = [u for u in units if u in VOICE_UNITS]
    if voice and len(voice) != len(units):
        raise ValueError(f"homeai and user units in one systemctl call: {units}")
    if voice and scope() == "system":
        return ["systemctl", *args]
    return ["systemctl", "--user", *args]


def journal_hint(lines: int = 50) -> str:
    """The command that shows Jarvis's log, for error messages."""
    user = "" if scope() == "system" else " --user"
    return f"journalctl{user} -u homeai -n {lines}"
