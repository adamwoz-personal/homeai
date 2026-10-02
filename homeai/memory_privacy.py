# Copyright (c) 2026 Adam Wosotowsky
# SPDX-License-Identifier: BUSL-1.1
# Licensed under the Business Source License 1.1; see LICENSE at the
# repository root. Converts to Apache-2.0 on 2030-09-15.
"""Control what ZeroClaw remembers of what is said to it (``homeai-mode memory``).

ZeroClaw's ``[memory] auto_save`` is global: with it on, every request to
every agent, including speech Jarvis overheard, is stored in
``data/memory/brain.db`` as a ``conversation`` row and deleted after
``conversation_retention_days``. The voice agent cannot read it back
(``allowed_tools = ["calculator"]``), but anyone with the disk can.

    homeai-mode memory                  # status
    homeai-mode memory off | on         # stop / resume saving
    homeai-mode memory retention DAYS   # 0 = keep forever
    homeai-mode memory purge [--yes]    # delete saved conversations now

Only the ``conversation`` category is touched: notes an agent stored on
purpose (``core``, ``daily``) are left alone. All side effects go through
``MemoryBackend`` so the logic is unit-tested without touching the machine.
"""

from __future__ import annotations

import re
import sqlite3
import subprocess
from dataclasses import dataclass, field
from pathlib import Path

CATEGORY = "conversation"
DAEMON_SERVICE = "zeroclaw.service"
DEFAULT_CONFIG_DIR = Path.home() / ".zeroclaw"


class MemoryControlError(Exception):
    pass


class MemoryBackend:
    """The real ZeroClaw install. Tests substitute a fake with the same methods."""

    def __init__(self, config_dir: Path | None = None,
                 restart_daemon: bool | None = None) -> None:
        self.config_dir = Path(config_dir or DEFAULT_CONFIG_DIR)
        # The systemd daemon serves ~/.zeroclaw only; never restart it for a copy.
        if restart_daemon is None:
            restart_daemon = self.config_dir.resolve() == DEFAULT_CONFIG_DIR.resolve()
        self.restart_daemon = restart_daemon

    @property
    def memory_dir(self) -> Path:
        return self.config_dir / "data" / "memory"

    def zeroclaw(self, *args: str) -> tuple[int, str]:
        cmd = ["zeroclaw", *args[:2], "--config-dir", str(self.config_dir), *args[2:]]
        try:
            proc = subprocess.run(cmd, capture_output=True, text=True, check=False, timeout=60)
        except FileNotFoundError:
            raise MemoryControlError("zeroclaw is not on PATH") from None
        except subprocess.TimeoutExpired:
            raise MemoryControlError(f"timed out: {' '.join(cmd)}") from None
        return proc.returncode, (proc.stdout + proc.stderr).strip()

    def disk_bytes(self) -> int:
        if not self.memory_dir.is_dir():
            return 0
        return sum(p.stat().st_size for p in self.memory_dir.rglob("*") if p.is_file())

    def compact(self) -> None:
        """Make deleted rows unrecoverable: fold the WAL in and rewrite the file."""
        db = self.memory_dir / "brain.db"
        if not db.exists():
            return
        try:
            con = sqlite3.connect(db, timeout=30)
            try:
                con.execute("PRAGMA wal_checkpoint(TRUNCATE)")
                con.execute("VACUUM")
                con.execute("PRAGMA wal_checkpoint(TRUNCATE)")
            finally:
                con.close()
        except sqlite3.Error as exc:
            raise MemoryControlError(f"rows deleted, but compacting brain.db failed: {exc}") from exc

    def reload_daemon(self) -> str:
        """The daemon runs the retention sweep, so it must see the new config."""
        if not self.restart_daemon:
            return "skipped"
        active = subprocess.run(["systemctl", "--user", "is-active", "--quiet", DAEMON_SERVICE],
                                check=False).returncode == 0
        if not active:
            return "not running"
        ok = subprocess.run(["systemctl", "--user", "restart", DAEMON_SERVICE],
                            check=False, capture_output=True).returncode == 0
        return "restarted" if ok else "RESTART FAILED"


@dataclass
class MemoryStatus:
    auto_save: bool
    retention_days: int
    counts: dict[str, int] = field(default_factory=dict)
    disk_bytes: int = 0

    def lines(self) -> list[str]:
        keep = "forever" if self.retention_days == 0 else f"{self.retention_days} days"
        counts = ", ".join(f"{k} {v}" for k, v in sorted(self.counts.items())) or "none"
        return [
            f"saving:    {'ON' if self.auto_save else 'off'} (everything said to any agent)",
            f"retention: {keep} for conversations",
            f"stored:    {counts}",
            f"disk:      {self.disk_bytes / 1024**2:.1f} MB",
        ]


_COUNT_LINE = re.compile(r"^\s{2,}([a-z_]+)\s+(\d+)\s*$")


def parse_stats(text: str) -> dict[str, int]:
    """Parse the 'By category' block of ``zeroclaw memory stats``."""
    counts: dict[str, int] = {}
    in_block = False
    for line in text.splitlines():
        if "By category" in line:
            in_block = True
            continue
        if in_block:
            m = _COUNT_LINE.match(line)
            if m:
                counts[m.group(1)] = int(m.group(2))
            elif line.strip():
                break
    return counts


def _parse_bool(text: str) -> bool:
    value = text.strip().splitlines()[-1].strip().lower() if text.strip() else ""
    if value in ("true", "false"):
        return value == "true"
    raise MemoryControlError(f"unexpected memory.auto_save value: {text!r}")


def _parse_int(text: str) -> int:
    value = text.strip().splitlines()[-1].strip() if text.strip() else ""
    if not value.isdigit():
        raise MemoryControlError(f"unexpected memory.conversation_retention_days value: {text!r}")
    return int(value)


class MemoryControl:
    def __init__(self, backend: MemoryBackend, log=print) -> None:
        self.be = backend
        self.log = log

    def _run(self, *args: str) -> str:
        code, out = self.be.zeroclaw(*args)
        if code != 0:
            raise MemoryControlError(f"zeroclaw {' '.join(args)} failed: {out[-300:]}")
        return out

    def counts(self) -> dict[str, int]:
        return parse_stats(self._run("memory", "stats"))

    def status(self) -> MemoryStatus:
        return MemoryStatus(
            auto_save=_parse_bool(self._run("config", "get", "memory.auto_save")),
            retention_days=_parse_int(self._run("config", "get",
                                                "memory.conversation_retention_days")),
            counts=self.counts(),
            disk_bytes=self.be.disk_bytes(),
        )

    def _set(self, path: str, value: str, comment: str) -> None:
        self._run("config", "set", path, value, "--no-interactive", "--comment", comment)
        daemon = self.be.reload_daemon()
        self.log(f"{path} = {value} (daemon {daemon})")

    def set_saving(self, on: bool) -> None:
        self._set("memory.auto_save", "true" if on else "false",
                  "set by homeai-mode memory " + ("on" if on else "off"))
        if _parse_bool(self._run("config", "get", "memory.auto_save")) != on:
            raise MemoryControlError("memory.auto_save did not change")

    def set_retention(self, days: int) -> None:
        if days < 0:
            raise MemoryControlError("retention must be 0 (forever) or a positive number of days")
        self._set("memory.conversation_retention_days", str(days),
                  "set by homeai-mode memory retention; 0 = keep forever")
        if _parse_int(self._run("config", "get", "memory.conversation_retention_days")) != days:
            raise MemoryControlError("memory.conversation_retention_days did not change")

    def purge(self, confirm=None) -> int:
        """Delete saved conversations. ``confirm(n)`` returns False to abort."""
        n = self.counts().get(CATEGORY, 0)
        if n == 0:
            self.log("no saved conversations")
            self.be.compact()
            return 0
        if confirm is not None and not confirm(n):
            self.log("cancelled; nothing deleted")
            return 0
        self._run("memory", "clear", "--category", CATEGORY, "--yes")
        left = self.counts().get(CATEGORY, 0)
        if left:
            raise MemoryControlError(f"{left} conversation rows remain after clearing")
        self.be.compact()
        self.log(f"deleted {n} saved conversations")
        return n
