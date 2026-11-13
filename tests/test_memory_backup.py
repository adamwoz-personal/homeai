# Copyright (c) 2026 Adam Wosotowsky
# SPDX-License-Identifier: BUSL-1.1
"""Backups that are proven to restore (memory_store/backup.py)."""

from __future__ import annotations

import json
import os
import shutil
import stat
import subprocess
import types
from datetime import datetime, timedelta

import pytest

from homeai.memory_store import backup
from homeai.memory_store.store import Episode
from tests.test_memory_cli import make_cli

pytestmark = pytest.mark.skipif(shutil.which("pg_dump") is None, reason="no pg_dump")

T0 = datetime(2026, 11, 1, 3, 0, 0)


def fill(store):
    store.add_episode(Episode("adam", "I have a corn snake called Noodle", "Lovely."))
    store.add_fact("adam", "Adam has a corn snake called Noodle", topics=("pets",))
    store.add_fact("ryan", "Ryan likes kites")


def databases() -> set[str]:
    out = subprocess.run(["psql", "-Atc", "SELECT datname FROM pg_database", "postgres"],
                         capture_output=True, text=True, check=True).stdout
    return set(out.split())


@pytest.fixture(autouse=True)
def drop_leaked_scratch_dbs():
    """A test (or a mutant) that leaks a restore scratch DB fails on its own
    assertion; this keeps the leak from outliving it."""
    if shutil.which("psql") is None:
        yield
        return
    try:
        before = databases()
    except (OSError, subprocess.CalledProcessError):
        yield
        return
    yield
    for name in databases() - before:
        if name.startswith("homeai_restore_"):
            subprocess.run(["dropdb", "--if-exists", "--force", name], capture_output=True)


def test_dump_holds_everything_and_restores_with_the_same_counts(mem_store, pg_dsn, tmp_path):
    fill(mem_store)
    path = backup.dump(pg_dsn, tmp_path / "b", now=T0)
    meta = json.loads(path.with_suffix(".json").read_text())
    # Counts are taken with admin scope: RLS would otherwise hide ryan's rows.
    assert meta["counts"]["facts"] == 2 and meta["counts"]["episodes"] == 1
    assert {"auth.accounts", "auth.devices", "auth.audit"} <= set(meta["counts"])   # logins are backed up too
    assert meta["schema_version"] >= 2
    before = databases()
    r = backup.restore_check(path)
    assert r["counts"] == meta["counts"]
    assert databases() == before


def test_dump_files_are_private(mem_store, pg_dsn, tmp_path):
    dest = tmp_path / "b"
    path = backup.dump(pg_dsn, dest, now=T0)
    assert stat.S_IMODE(dest.stat().st_mode) == 0o700
    assert stat.S_IMODE(path.stat().st_mode) == 0o600
    assert stat.S_IMODE(path.with_suffix(".json").stat().st_mode) == 0o600


def test_restore_check_catches_missing_logins(mem_store, pg_dsn, tmp_path):
    path = backup.dump(pg_dsn, tmp_path, now=T0)
    meta = json.loads(path.with_suffix(".json").read_text())
    meta["counts"]["auth.accounts"] += 1
    path.with_suffix(".json").write_text(json.dumps(meta))
    with pytest.raises(backup.BackupError, match="auth.accounts: dump had"):
        backup.restore_check(path)


def test_counts_files_from_before_logins_still_check(mem_store, pg_dsn, tmp_path):
    path = backup.dump(pg_dsn, tmp_path, now=T0)
    meta = json.loads(path.with_suffix(".json").read_text())
    meta["counts"] = {t: n for t, n in meta["counts"].items() if not t.startswith("auth.")}
    path.with_suffix(".json").write_text(json.dumps(meta))
    assert backup.restore_check(path)["file"] == path.name


def test_counts_skip_tables_a_database_does_not_have_yet():
    class Conn:
        def execute(self, sql, args=()):
            name = args[0] if args else ""
            value = None if name.startswith("auth.") else (name or 7)
            return type("R", (), {"fetchone": lambda self: (value if "to_regclass" in sql else 7,)})()
    got = backup.counts(Conn())
    assert "auth.audit" not in got and got["facts"] == 7


def test_restore_check_catches_missing_rows(mem_store, pg_dsn, tmp_path):
    fill(mem_store)
    path = backup.dump(pg_dsn, tmp_path, now=T0)
    meta = json.loads(path.with_suffix(".json").read_text())
    meta["counts"]["facts"] += 1
    path.with_suffix(".json").write_text(json.dumps(meta))
    with pytest.raises(backup.BackupError, match="facts: dump had 3, restored 2"):
        backup.restore_check(path)


def test_restore_check_catches_a_wrong_schema_version(mem_store, pg_dsn, tmp_path):
    path = backup.dump(pg_dsn, tmp_path, now=T0)
    meta = json.loads(path.with_suffix(".json").read_text())
    meta["schema_version"] += 1
    path.with_suffix(".json").write_text(json.dumps(meta))
    with pytest.raises(backup.BackupError, match="schema version"):
        backup.restore_check(path)


def test_restore_check_catches_a_corrupt_dump_and_cleans_up(mem_store, pg_dsn, tmp_path):
    fill(mem_store)
    path = backup.dump(pg_dsn, tmp_path, now=T0)
    data = path.read_bytes()
    path.write_bytes(data[: len(data) // 2])
    before = databases()
    with pytest.raises(backup.BackupError, match="pg_restore failed"):
        backup.restore_check(path)
    assert databases() == before


def test_restore_check_needs_the_counts_file(tmp_path):
    (tmp_path / "memory-x.dump").write_bytes(b"x")
    with pytest.raises(backup.BackupError, match="no readable counts"):
        backup.restore_check(tmp_path / "memory-x.dump")


def test_restore_check_insists_row_security_is_forced(pg_dsn, tmp_path):
    name = f"homeai_test_norls_{os.getpid()}"
    subprocess.run(["createdb", "--template", pg_dsn.split("=", 1)[1], name], check=True)
    try:
        subprocess.run(["psql", "-qc", "ALTER TABLE memory.facts NO FORCE ROW LEVEL SECURITY",
                        name], check=True, capture_output=True)
        path = backup.dump(f"dbname={name}", tmp_path, now=T0)
    finally:
        subprocess.run(["dropdb", "--force", name], check=True)
    with pytest.raises(backup.BackupError, match="row security not forced on facts"):
        backup.restore_check(path)


def test_failed_dump_leaves_no_partial_file(mem_store, pg_dsn, tmp_path, monkeypatch):
    def fail(cmd, run=None):
        open(next(a.split("=", 1)[1] for a in cmd if a.startswith("--file=")), "w").close()
        raise backup.BackupError("pg_dump failed: disk full")

    monkeypatch.setattr(backup, "_run", fail)
    with pytest.raises(backup.BackupError):
        backup.dump(pg_dsn, tmp_path / "b", now=T0)
    assert list((tmp_path / "b").iterdir()) == []


def test_cli_backup_does_nothing_when_memory_is_off(pg_dsn, tmp_path):
    code, out = make_cli(pg_dsn, tmp_path, enabled=False)("backup", "--dir", str(tmp_path / "b"))
    assert code == 0 and "off" in out and not (tmp_path / "b").exists()


def test_prune_keeps_the_newest_complete_dumps(tmp_path):
    for i in range(5):
        stem = tmp_path / f"memory-2026110{i}-030000"
        stem.with_suffix(".dump").write_bytes(b"x")
        stem.with_suffix(".json").write_text("{}")
    (tmp_path / "memory-20261109-030000.dump").write_bytes(b"no counts: incomplete")
    gone = backup.prune(tmp_path, 2)
    assert [p.name for p in gone] == [f"memory-2026110{i}-030000.dump" for i in range(3)]
    assert [p.name for p in backup.dumps(tmp_path)] == ["memory-20261103-030000.dump",
                                                        "memory-20261104-030000.dump"]
    assert not (tmp_path / "memory-20261100-030000.json").exists()


def test_dump_prunes_old_ones(mem_store, pg_dsn, tmp_path):
    for i in range(3):
        backup.dump(pg_dsn, tmp_path, keep=2, now=T0 + timedelta(days=i))
    assert len(backup.dumps(tmp_path)) == 2


def test_cli_backup_then_restore_check(mem_store, pg_dsn, tmp_path):
    fill(mem_store)
    go = make_cli(pg_dsn, tmp_path, enabled=True)
    code, out = go("backup", "--dir", str(tmp_path / "b"), "--keep", "3")
    assert code == 0 and "restore check ok" in out and "facts=2" in out
    code, out = go("restore-check", "--dir", str(tmp_path / "b"))
    assert code == 0 and "restore check ok" in out


def test_cli_restore_check_with_no_dumps_fails(pg_dsn, tmp_path):
    code, _ = make_cli(pg_dsn, tmp_path, enabled=True)("restore-check", "--dir", str(tmp_path / "none"))
    assert code == 2


def test_cli_backup_reports_failure(mem_store, pg_dsn, tmp_path, monkeypatch):
    def bad(*a, **k):
        raise backup.BackupError("pg_dump failed: nope")

    monkeypatch.setattr(backup, "dump", bad)
    code, _ = make_cli(pg_dsn, tmp_path, enabled=True)("backup", "--dir", str(tmp_path))
    assert code == 1


class _Run:
    """subprocess.run stand-in: fails with ``errors`` in turn, then succeeds."""

    def __init__(self, errors):
        self.errors, self.calls = list(errors), 0

    def __call__(self, cmd, **kw):
        self.calls += 1
        err = self.errors.pop(0) if self.errors else ""
        return types.SimpleNamespace(returncode=1 if err else 0, stderr=err)


BUSY = "dropdb: error: database removal failed: ERROR:  permission denied to terminate process"


def test_dropdb_waits_out_a_server_worker_still_connected():
    """Right after a restore a server worker can hold the scratch database (2026-11-13, 3 of 4 runs failed)."""
    run, slept = _Run([BUSY, BUSY]), []
    backup._dropdb("scratch", run, slept.append)
    assert run.calls == 3 and slept == [backup.DROP_WAIT_S] * 2


def test_dropdb_gives_up_in_the_end_and_never_retries_other_errors():
    run = _Run([BUSY] * backup.DROP_TRIES)
    with pytest.raises(backup.BackupError, match="terminate"):
        backup._dropdb("scratch", run, lambda s: None)
    assert run.calls == backup.DROP_TRIES
    run = _Run(["dropdb: error: could not connect"])
    with pytest.raises(backup.BackupError, match="connect"):
        backup._dropdb("scratch", run, lambda s: None)
    assert run.calls == 1
