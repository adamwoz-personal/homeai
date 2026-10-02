# Copyright (c) 2026 Adam Wosotowsky
# SPDX-License-Identifier: BUSL-1.1
"""Tests for homeai-mode memory (homeai/memory_privacy.py)."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest

from homeai import gpu_mode
from homeai import memory_privacy as M

STATS = """Memory Statistics:

  Backend:  sqlite
  Health:   healthy
  Total:    {total}

  By category:
{rows}"""


class FakeBackend:
    def __init__(self, *, conversation=5, core=2, auto_save=True, retention=30,
                 clear_works=True, set_works=True):
        self.config = {"memory.auto_save": "true" if auto_save else "false",
                       "memory.conversation_retention_days": str(retention)}
        self.rows = {"conversation": conversation, "core": core}
        self.clear_works = clear_works
        self.set_works = set_works
        self.calls: list[tuple[str, ...]] = []
        self.compacted = 0
        self.reloads = 0

    def zeroclaw(self, *args):
        self.calls.append(args)
        if args[:2] == ("memory", "stats"):
            rows = "\n".join(f"    {k:<20} {v}" for k, v in self.rows.items() if v)
            return 0, STATS.format(total=sum(self.rows.values()), rows=rows)
        if args[:2] == ("config", "get"):
            return 0, self.config[args[2]]
        if args[:2] == ("config", "set"):
            if self.set_works:
                self.config[args[2]] = args[3]
            return 0, "ok"
        if args[:2] == ("memory", "clear"):
            cat = args[args.index("--category") + 1]
            if self.clear_works:
                self.rows[cat] = 0
            return 0, "cleared"
        return 2, "unknown"

    def disk_bytes(self):
        return 3 * 1024**2

    def compact(self):
        self.compacted += 1

    def reload_daemon(self):
        self.reloads += 1
        return "restarted"


def control(be):
    return M.MemoryControl(be, log=lambda *_: None)


def test_parse_stats_reads_category_block_only():
    text = STATS.format(total=7, rows="    conversation         5\n    core                 2")
    assert M.parse_stats(text + "\n\nOther:\n  junk 9") == {"conversation": 5, "core": 2}


def test_parse_stats_empty_db():
    assert M.parse_stats(STATS.format(total=0, rows="")) == {}


def test_status_reports_config_counts_and_disk():
    st = control(FakeBackend(auto_save=False, retention=0)).status()
    assert (st.auto_save, st.retention_days, st.counts) == (False, 0, {"conversation": 5, "core": 2})
    text = "\n".join(st.lines())
    assert "off" in text and "forever" in text and "3.0 MB" in text


def test_status_rejects_garbage_config_value():
    be = FakeBackend()
    be.config["memory.auto_save"] = "maybe"
    with pytest.raises(M.MemoryControlError):
        control(be).status()


@pytest.mark.parametrize("on", [True, False])
def test_set_saving_writes_config_and_reloads_daemon(on):
    be = FakeBackend(auto_save=not on)
    control(be).set_saving(on)
    assert be.config["memory.auto_save"] == ("true" if on else "false")
    assert be.reloads == 1
    assert any(c[:3] == ("config", "set", "memory.auto_save") and "--no-interactive" in c
               for c in be.calls)


def test_set_saving_verifies_the_change():
    with pytest.raises(M.MemoryControlError, match="did not change"):
        control(FakeBackend(auto_save=True, set_works=False)).set_saving(False)


def test_set_retention():
    be = FakeBackend()
    control(be).set_retention(7)
    assert be.config["memory.conversation_retention_days"] == "7"


def test_set_retention_rejects_negative_without_touching_config():
    be = FakeBackend()
    with pytest.raises(M.MemoryControlError):
        control(be).set_retention(-1)
    assert not any(c[:2] == ("config", "set") for c in be.calls)


def test_purge_clears_only_conversations_and_compacts():
    be = FakeBackend()
    assert control(be).purge() == 5
    assert be.rows == {"conversation": 0, "core": 2}
    assert be.compacted == 1


def test_purge_cancelled_deletes_nothing():
    be = FakeBackend()
    asked = []
    assert control(be).purge(confirm=lambda n: asked.append(n) or False) == 0
    assert asked == [5] and be.rows["conversation"] == 5 and be.compacted == 0


def test_purge_detects_rows_left_behind():
    be = FakeBackend(clear_works=False)
    with pytest.raises(M.MemoryControlError, match="remain"):
        control(be).purge()
    assert be.compacted == 0


def test_purge_with_nothing_saved_still_compacts_leftovers():
    be = FakeBackend(conversation=0)
    assert control(be).purge(confirm=lambda n: pytest.fail("should not ask")) == 0
    assert be.compacted == 1


def test_failing_zeroclaw_becomes_error():
    class Broken(FakeBackend):
        def zeroclaw(self, *args):
            return 1, "boom"
    with pytest.raises(M.MemoryControlError, match="boom"):
        control(Broken()).status()


def test_daemon_only_restarted_for_default_config_dir(tmp_path):
    assert M.MemoryBackend(tmp_path).restart_daemon is False
    assert M.MemoryBackend().restart_daemon is True
    assert M.MemoryBackend(tmp_path).reload_daemon() == "skipped"


def test_compact_shrinks_db_after_delete(tmp_path):
    mem = tmp_path / "data" / "memory"
    mem.mkdir(parents=True)
    import sqlite3
    con = sqlite3.connect(mem / "brain.db")
    con.execute("PRAGMA journal_mode=WAL")
    con.execute("create table t(x)")
    con.executemany("insert into t values (?)", [("secret " * 200,)] * 500)
    con.commit()
    con.execute("delete from t")
    con.commit()
    con.close()
    be = M.MemoryBackend(tmp_path)
    before = be.disk_bytes()
    be.compact()
    assert be.disk_bytes() < before / 5
    assert b"secret" not in (mem / "brain.db").read_bytes()


def _args(cmd, **kw):
    return SimpleNamespace(memory_cmd=cmd, config_dir=None, **kw)


def test_cli_purge_asks_and_respects_no(capsys):
    be = FakeBackend()
    rc = gpu_mode._memory_main(_args("purge", yes=False), backend=be, ask=lambda _: "n")
    assert rc == 0 and be.rows["conversation"] == 5


def test_cli_purge_yes_skips_prompt():
    be = FakeBackend()
    rc = gpu_mode._memory_main(_args("purge", yes=True), backend=be,
                               ask=lambda _: pytest.fail("prompted"))
    assert rc == 0 and be.rows["conversation"] == 0


def test_cli_purge_eof_means_no():
    be = FakeBackend()
    def eof(_):
        raise EOFError
    assert gpu_mode._memory_main(_args("purge", yes=False), backend=be, ask=eof) == 0
    assert be.rows["conversation"] == 5


def test_cli_error_returns_1(capsys):
    rc = gpu_mode._memory_main(_args("retention", days=-3), backend=FakeBackend())
    assert rc == 1 and "retention" in capsys.readouterr().err


def test_cli_default_is_status(capsys):
    assert gpu_mode._memory_main(_args(None), backend=FakeBackend()) == 0
    assert "saving:" in capsys.readouterr().out
