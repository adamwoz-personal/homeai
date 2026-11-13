# Copyright (c) 2026 Adam Wosotowsky
# SPDX-License-Identifier: BUSL-1.1
# Licensed under the Business Source License 1.1; see LICENSE at the
# repository root. Converts to Apache-2.0 on 2030-09-15.
"""Nightly dumps of the memory database, and proof they restore.

``dump`` exports a snapshot, counts every table inside it, and runs pg_dump
on that same snapshot, so the counts saved beside the dump describe exactly
what is in it. ``restore_check`` restores a dump into a scratch database and
compares: same counts, same schema version, row security still forced. A
backup that was never restored is a hope, not a backup.

Row security is forced even for the owner, so pg_dump needs
``--enable-row-security`` and ``app.scope=admin`` (the policy's
see-everything case) or it refuses to dump.

Dumps hold everything people said; files are 0600 in a 0700 directory.
"""

from __future__ import annotations

import json
import os
import subprocess
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Callable

DEFAULT_DIR = Path("~/.local/share/homeai/backups").expanduser()
KEEP = 14
PREFIX = "memory-"
# Every table with data worth keeping; health samples and finished jobs are not.
TABLES = ("people", "identities", "episodes", "facts", "fact_sources", "fact_links",
          "topics", "events", "schema_version",
          # Remote app logins (0004): schema-qualified, since they aren't in ``memory``.
          "auth.accounts", "auth.devices", "auth.audit")
FORCED_RLS = ("episodes", "facts")
ADMIN_ENV = {"PGOPTIONS": "-c app.scope=admin"}


class BackupError(RuntimeError):
    pass


def _env() -> dict[str, str]:
    return {**os.environ, **ADMIN_ENV}


def _run(cmd: list[str], run: Callable = subprocess.run) -> None:
    r = run(cmd, capture_output=True, text=True, env=_env())
    if r.returncode:
        tail = (r.stderr or "").strip().splitlines()[-3:]
        raise BackupError(f"{cmd[0]} failed: {' / '.join(tail) or f'exit {r.returncode}'}")


DROP_TRIES = 10
DROP_WAIT_S = 0.5


def _dropdb(name: str, run: Callable = subprocess.run, sleep: Callable[[float], None] = time.sleep) -> None:
    """dropdb --force, retried: right after a restore a server background worker (not ours to terminate) can still
    be connected, and dropdb fails with "permission denied to terminate process" (3 of 4 test runs, 2026-11-13)."""
    for attempt in range(DROP_TRIES):
        try:
            _run(["dropdb", "--if-exists", "--force", name], run)
            return
        except BackupError as exc:
            if "permission denied to terminate" not in str(exc) or attempt == DROP_TRIES - 1:
                raise
            sleep(DROP_WAIT_S)


def counts(conn) -> dict[str, int]:
    """Rows per table, seen with admin scope (inside the caller's transaction)."""
    conn.execute("SELECT set_config('app.scope', 'admin', true)")
    out = {}
    for t in TABLES:
        name = t if "." in t else f"memory.{t}"
        # A dump from before a migration has no such table: absent here and in its counts file alike.
        if conn.execute("SELECT to_regclass(%s)", (name,)).fetchone()[0] is not None:
            out[t] = conn.execute(f"SELECT count(*) FROM {name}").fetchone()[0]
    return out


def dump(dsn: str, dest: Path = DEFAULT_DIR, *, keep: int = KEEP,
         now: datetime | None = None) -> Path:
    """Write ``memory-<stamp>.dump`` (+ ``.json`` counts) and prune to ``keep``."""
    import psycopg

    dest.mkdir(mode=0o700, parents=True, exist_ok=True)
    os.chmod(dest, 0o700)
    stamp = (now or datetime.now()).strftime("%Y%m%d-%H%M%S")
    final = dest / f"{PREFIX}{stamp}.dump"
    partial = final.with_suffix(".partial")
    old_umask = os.umask(0o077)
    try:
        with psycopg.connect(dsn) as conn:
            conn.execute("BEGIN ISOLATION LEVEL REPEATABLE READ, READ ONLY")
            snapshot = conn.execute("SELECT pg_export_snapshot()").fetchone()[0]
            seen = counts(conn)
            version = conn.execute("SELECT max(version) FROM memory.schema_version").fetchone()[0]
            # The exporting transaction must stay open while pg_dump uses its snapshot.
            _run(["pg_dump", "--format=custom", "--enable-row-security",
                  f"--snapshot={snapshot}", f"--file={partial}", dsn])
            conn.rollback()
        partial.replace(final)
        final.with_suffix(".json").write_text(json.dumps(
            {"counts": seen, "schema_version": version, "at": stamp}, indent=1))
    except Exception:
        partial.unlink(missing_ok=True)
        raise
    finally:
        os.umask(old_umask)
    prune(dest, keep)
    return final


def dumps(dest: Path = DEFAULT_DIR) -> list[Path]:
    """Complete dumps, oldest first (the stamp sorts by time)."""
    return sorted(p for p in dest.glob(f"{PREFIX}*.dump") if p.with_suffix(".json").exists())


def prune(dest: Path, keep: int) -> list[Path]:
    gone = dumps(dest)[:-keep] if keep > 0 else []
    for p in gone:
        p.unlink(missing_ok=True)
        p.with_suffix(".json").unlink(missing_ok=True)
    return gone


def restore_check(path: Path, *, scratch: str | None = None) -> dict[str, Any]:
    """Restore ``path`` into a throwaway database and compare with its counts.

    Raises BackupError on any difference; returns the counts when it matches.
    """
    import psycopg

    meta_file = path.with_suffix(".json")
    try:
        meta = json.loads(meta_file.read_text())
    except (OSError, ValueError) as exc:
        raise BackupError(f"no readable counts beside {path.name}: {exc}") from exc
    name = scratch or f"homeai_restore_{os.getpid()}"
    _dropdb(name)
    _run(["createdb", name])
    try:
        _run(["pg_restore", "--exit-on-error", "--single-transaction", "--no-comments", f"--dbname={name}",
              str(path)])
        with psycopg.connect(f"dbname={name}") as conn:
            got = counts(conn)
            version = conn.execute("SELECT max(version) FROM memory.schema_version").fetchone()[0]
            forced = {r[0] for r in conn.execute(
                "SELECT relname FROM pg_class c JOIN pg_namespace n ON n.oid = c.relnamespace"
                " WHERE n.nspname = 'memory' AND c.relforcerowsecurity")}
    finally:
        _dropdb(name)
    problems = [f"{t}: dump had {meta['counts'].get(t)}, restored {got.get(t)}"
                for t in TABLES if got.get(t) != meta["counts"].get(t)
                and (t in meta["counts"] or "." not in t)]   # counts files from before 0004 have no auth.*
    if version != meta["schema_version"]:
        problems.append(f"schema version {version} != {meta['schema_version']}")
    problems += [f"row security not forced on {t}" for t in FORCED_RLS if t not in forced]
    if problems:
        raise BackupError("restore check failed: " + "; ".join(problems))
    return {"file": path.name, "counts": got, "schema_version": version}
