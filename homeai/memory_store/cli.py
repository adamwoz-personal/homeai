# Copyright (c) 2026 Adam Wosotowsky
# SPDX-License-Identifier: BUSL-1.1
# Licensed under the Business Source License 1.1; see LICENSE at the
# repository root. Converts to Apache-2.0 on 2030-09-15.
"""``homeai-memory``: look after long-term memory by hand.

    homeai-memory migrate                      apply schema migrations
    homeai-memory people                       who has a memory
    homeai-memory add-person ryan "Ryan" [--child]
    homeai-memory list [--as adam] [--owner ryan] [-n 20]
    homeai-memory search --as adam "what did I say about the garage"
    homeai-memory note --as adam [--profile] [--pin] "Adam is an engineer"
    homeai-memory forget --as adam <id>
    homeai-memory export --as adam > adam.json
    homeai-memory stats
    homeai-memory backfill [--dry-run] [--include-unknown]
    homeai-memory backup [--keep 14] [--no-check]   pg_dump + restore check (nightly timer)
    homeai-memory restore-check [FILE]

``--as`` is who is asking (default: adam). Admins see everyone's memories;
others see their own plus the household's.
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path
from typing import Iterable, Iterator

from ..config import Config
from . import policy, recall
from .store import Episode, StoreError


def backfill_episodes(lines: Iterable[str], *, enrolled: set[str], ignore: tuple[str, ...],
                      include_unknown: bool = False) -> Iterator[tuple[Episode, str]]:
    """Episodes from transcript.jsonl lines, with the live recording rules.

    Yields (episode, display name). Only speakers enrolled now count: a name
    from a deleted enrolment, or a test voice, is not a person.
    """
    for line in lines:
        try:
            t = json.loads(line)
            at = datetime.fromisoformat(t["ts"])
        except (ValueError, KeyError, TypeError):
            continue
        speaker = t.get("speaker")
        heard, reply = t.get("heard", ""), t.get("reply", "")
        if policy.exchange_problem(heard=heard, reply=reply, verdict=t.get("verdict", ""),
                                   ok=t.get("ok", True)):
            continue
        user = policy.username_for(speaker)
        if user in ignore:
            continue
        common = dict(user=policy.clean_heard(heard), assistant=reply, occurred_at=at,
                      source_ref=f"transcript:{t['ts']}",
                      extra={"verdict": t.get("verdict"), "backfill": True})
        if user and user in enrolled:
            yield Episode(username=user, confidence=float(t.get("speaker_score") or 1.0),
                          **common), speaker
        elif include_unknown:
            yield Episode(username="household", confidence=0.0, trust="household",
                          **common), "Household"


def enrolled_speakers(path: str) -> set[str]:
    try:
        data = json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return set()
    speakers = data.get("speakers", {}) if isinstance(data, dict) else {}
    return {u for u in (policy.username_for(str(n)) for n in speakers) if u}


class _LazyEmbedder:
    """Loads the ONNX model (~1 s, ~900 MB peak) only when something embeds:
    most commands, and the hourly ``maintain``, never do."""

    def __init__(self, make) -> None:
        self._make, self._e = make, None

    def __getattr__(self, name):
        if self._e is None:
            self._e = self._make()
        return getattr(self._e, name)


def _store(cfg: Config, embedder=None):
    from .db import Db
    from .embed import DEFAULT_MODEL, OnnxEmbedder
    from .store import PostgresStore

    db = Db(cfg.memory.dsn, role="homeai_app").open()
    return PostgresStore(db, embedder or _LazyEmbedder(
        lambda: OnnxEmbedder(cfg.memory.embed_model or DEFAULT_MODEL)))


def _scope(store, who: str) -> str:
    return "admin" if store.is_admin(who) else "person"


def _date(s: str) -> datetime:
    try:
        return datetime.strptime(s, "%Y-%m-%d").astimezone()
    except ValueError as exc:
        raise argparse.ArgumentTypeError(f"not a YYYY-MM-DD date: {s!r}") from exc


def _health(a, cfg: Config, store, out) -> int:
    from . import maintain

    if a.cmd == "health":
        with store.db.scoped(None, "admin") as conn:
            metrics = maintain.sample(conn)
        report = {"metrics": metrics, "actions": [
            {"kind": x.kind, "target": x.target, "reason": x.reason, "runs": x.runs}
            for x in maintain.decide(metrics)]}
    else:
        report = maintain.maintain(store, cfg.memory.dsn, dry_run=a.dry_run,
                                   time_limit_s=a.time_limit)
    if a.json:
        print(json.dumps(report, default=_default), file=out)
        return 1 if report.get("errors") else 0
    m = report["metrics"]
    r = m["recall"]
    p95 = f"{r['p95']:.1f} ms" if r["p95"] is not None else "n/a"
    print(f"db {m['db_bytes'] / 1e6:.1f} MB, recall p95 {p95} over {r['n']} recalls,"
          f" jobs {m['jobs'] or 'none'}, purgeable {m['purgeable']}", file=out)
    for user, p in sorted(m["people"].items()):
        red = f"{p['redundancy']:.0%} of {p['redundancy_n']}" \
            if p["redundancy"] is not None else "n/a"
        print(f"  {user:10} facts {p['facts']:5} (stale {p['stale']}, near-dup {red})"
              f"  episodes {p['episodes']:6} (>90d {p['old_episodes']})", file=out)
    if not report["actions"]:
        print("healthy, no work", file=out)
    for x in report["actions"]:
        verb = "run" if x["runs"] else "needs"
        print(f"{verb} {x['kind']}{' ' + x['target'] if x['target'] else ''}: {x['reason']}",
              file=out)
    for d in report.get("done", []):
        print(f"done {d['kind']}: {d['result']}", file=out)
    for d in report.get("deferred", []):
        print(f"deferred {d}: time limit", file=out)
    for e in report.get("errors", []):
        print(f"error {e['kind']}: {e['error']}", file=sys.stderr)
    return 1 if report.get("errors") else 0


def _rederive(a, cfg: Config, store, who: str, scope: str, out, helper=None) -> int:
    from .rederive import rederive

    person = (a.person or who).lower()
    if person != who and scope != "admin":
        raise StoreError("only an admin can rederive someone else's memories")
    if helper is None:
        from .helper import OllamaHelper

        helper = OllamaHelper(cfg.memory.helper_url, a.model or cfg.memory.helper_model,
                              threads=cfg.memory.helper_threads)
    r = rederive(store, helper, person, since=a.since, apply=a.apply,
                 batch=cfg.memory.extract_batch)
    if a.json:
        print(json.dumps(r, default=_default), file=out)
        return 0
    if "skipped" in r:
        print(f"skipped: {r['skipped']}", file=out)
        return 0
    print(f"{r['person']}: {r['episodes']} episodes -> {r['notes']} notes with {r['model']}"
          f"{', ' + str(r['invalid_batches']) + ' invalid batches' if r['invalid_batches'] else ''}",
          file=out)
    if r["applied"]:
        print(f"applied: {r['actions'] or 'nothing'}", file=out)
        return 0
    for f in r["findings"]:
        tail = f"  ~ {f['closest']} ({f['sim']:.2f})" if f["closest"] else ""
        lock = " [protected]" if f["protected"] else ""
        print(f"{f['status']:5} {f['note']}{tail}{lock}", file=out)
    for f in r["gone"]:
        print(f"gone  {f['summary']}  ({f['id']})", file=out)
    print(f"{r['counts']}; nothing changed (use --apply to store)", file=out)
    return 0


def _default(o):
    return o.isoformat() if isinstance(o, datetime) else str(o)


def main(argv: list[str] | None = None, *, cfg: Config | None = None, embedder=None,
         out=None, helper=None) -> int:
    out = out or sys.stdout
    cfg = cfg or Config()
    ap = argparse.ArgumentParser(prog="homeai-memory", description="Long-term memory tools")
    ap.add_argument("--as", dest="who", default="adam", help="who is asking (default adam)")
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("migrate")
    sub.add_parser("people")
    p = sub.add_parser("add-person")
    p.add_argument("username")
    p.add_argument("display_name")
    p.add_argument("--child", action="store_true")
    p = sub.add_parser("list")
    p.add_argument("--owner")
    p.add_argument("-n", type=int, default=20)
    p = sub.add_parser("search")
    p.add_argument("query")
    p = sub.add_parser("note")
    p.add_argument("text")
    p.add_argument("--profile", action="store_true", help="always recalled for this person")
    p.add_argument("--pin", action="store_true")
    p.add_argument("--for", dest="owner", help="whose memory (admins only; default: --as)")
    p = sub.add_parser("forget")
    p.add_argument("id")
    sub.add_parser("export")
    sub.add_parser("stats")
    p = sub.add_parser("facts", help="your active facts (incl. profile lines)")
    p.add_argument("--kind", action="append", help="only these kinds (repeatable)")
    sub.add_parser("profile", help="your reflection profile")
    sub.add_parser("jobs", help="queued, running and failed background jobs")
    p = sub.add_parser("extract", help="extract facts from your unread exchanges now")
    p.add_argument("--all", action="store_true", help="everyone (admins only)")
    sub.add_parser("reflect", help="rebuild your profile now")
    p = sub.add_parser("backfill")
    p.add_argument("--dry-run", action="store_true")
    p.add_argument("--include-unknown", action="store_true",
                   help="unrecognised turns go to 'household' (may include test voices)")
    p.add_argument("--transcript", default=cfg.transcript.path)
    p = sub.add_parser("backup", help="pg_dump now, then prove the dump restores")
    p.add_argument("--dir", default=cfg.memory.backup_dir)
    p.add_argument("--keep", type=int, default=cfg.memory.backup_keep)
    p.add_argument("--no-check", action="store_true", help="skip the restore check")
    p = sub.add_parser("restore-check", help="restore a dump into a scratch database and compare")
    p.add_argument("file", nargs="?", help="default: the newest dump")
    p.add_argument("--dir", default=cfg.memory.backup_dir)
    p = sub.add_parser("health", help="measure and say what maintenance is due (changes nothing)")
    p.add_argument("--json", action="store_true")
    p = sub.add_parser("maintain", help="record a health sample and run the due maintenance")
    p.add_argument("--dry-run", action="store_true")
    p.add_argument("--json", action="store_true")
    p.add_argument("--time-limit", type=float, default=600.0, help="seconds (default 600)")
    p = sub.add_parser("rederive", help="re-extract facts from kept episodes (default: diff only)")
    p.add_argument("--person", help="whose episodes (admins only; default: --as)")
    p.add_argument("--since", type=_date, help="YYYY-MM-DD; default: all")
    p.add_argument("--model", help=f"helper model (default {cfg.memory.helper_model})")
    mode = p.add_mutually_exclusive_group()
    mode.add_argument("--dry-run", action="store_true", help="show the diff (the default)")
    mode.add_argument("--apply", action="store_true", help="store the notes")
    p.add_argument("--json", action="store_true")
    a = ap.parse_args(argv)

    if a.cmd in ("backup", "restore-check"):
        return _backup(a, cfg, out)

    if a.cmd == "migrate":
        from .db import migrate

        applied = migrate(cfg.memory.dsn)
        print(f"applied {applied}" if applied else "schema up to date", file=out)
        return 0

    if a.cmd == "backfill" and a.dry_run:
        eps = list(backfill_episodes(_read(a.transcript), enrolled=enrolled_speakers(
            cfg.speaker.registry_path), ignore=cfg.memory.ignore_speakers,
            include_unknown=a.include_unknown))
        counts: dict[str, int] = {}
        for ep, _ in eps:
            counts[ep.username] = counts.get(ep.username, 0) + 1
        print(f"would import {len(eps)} episodes: {counts}", file=out)
        return 0

    store = _store(cfg, embedder)
    try:
        return _run(a, cfg, store, out, helper)
    except StoreError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    finally:
        store.db.close()


def _backup(a, cfg: Config, out) -> int:
    """Owner-level work (pg_dump/pg_restore as the login user), no store needed."""
    from . import backup

    if not cfg.memory.enabled:
        print("long-term memory is off (HOMEAI_LTM=0); nothing to back up", file=out)
        return 0
    dest = Path(a.dir).expanduser()
    try:
        if a.cmd == "backup":
            path = backup.dump(cfg.memory.dsn, dest, keep=a.keep)
            print(f"wrote {path} ({path.stat().st_size} bytes)", file=out)
            if a.no_check:
                return 0
        else:
            path = Path(a.file).expanduser() if a.file else (backup.dumps(dest) or [None])[-1]
            if path is None:
                print(f"error: no dumps in {dest}", file=sys.stderr)
                return 2
        r = backup.restore_check(path)
        print(f"restore check ok: {r['file']} schema {r['schema_version']} "
              + " ".join(f"{k}={v}" for k, v in r["counts"].items()), file=out)
        return 0
    except backup.BackupError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1


def _read(path: str) -> list[str]:
    try:
        return Path(path).expanduser().read_text(encoding="utf-8").splitlines()
    except OSError as exc:
        raise SystemExit(f"cannot read transcript: {exc}") from exc


def _run(a, cfg: Config, store, out, helper=None) -> int:
    who = a.who.lower()
    if a.cmd == "people":
        for p in store.people():
            flags = [k for k in ("is_admin", "is_child") if p[k]]
            if not p["memory_enabled"]:
                flags.append("memory off")
            print(f"{p['username']:12} {p['display_name']:16} {' '.join(flags)}", file=out)
        return 0
    if a.cmd == "add-person":
        made = store.ensure_person(a.username, a.display_name,
                                   is_admin=a.username in cfg.memory.admins)
        if made and a.child:
            with store.db.scoped(None, "admin") as conn:
                conn.execute("UPDATE memory.people SET is_child = true WHERE username = %s",
                             (a.username,))
        print("added" if made else "already exists", file=out)
        return 0
    if store.person(who) is None and who in cfg.memory.admins:
        store.ensure_person(who, who.title(), is_admin=True)   # bootstrap a fresh database
    if store.person(who) is None:
        raise StoreError(f"no such person {who!r}")
    scope = _scope(store, who)
    if a.cmd in ("health", "maintain"):
        if scope != "admin":
            raise StoreError(f"only an admin can run {a.cmd}")
        return _health(a, cfg, store, out)
    if a.cmd == "rederive":
        return _rederive(a, cfg, store, who, scope, out, helper)
    if a.cmd == "list":
        for e in store.list_episodes(who, scope, owner=a.owner, limit=a.n):
            print(f"{e['id']}  {e['at']:%Y-%m-%d %H:%M}  {e['username']:8} {e['user'][:70]}",
                  file=out)
        return 0
    if a.cmd == "search":
        now = datetime.now().astimezone()
        vec = store.embedder.embed([a.query], kind="query")[0]
        with store.db.scoped(who, "person") as conn:
            rows = recall.fetch(conn, who, vec, a.query, recall.time_window(a.query, now))
        for c in recall.keep(recall.fuse(rows, now), cfg.memory.min_similarity):
            text = c.summary or f"{c.text_user} -> {c.text_reply}"
            print(f"{c.id}  {c.kind:7} sim {c.sim:.2f}  {text[:90]}", file=out)
        return 0
    if a.cmd == "note":
        owner = (a.owner or who).lower()
        if owner != who and scope != "admin":
            raise StoreError("only an admin can add notes for someone else")
        fid = store.add_fact(owner, a.text, kind="profile" if a.profile else "fact",
                             pinned=a.pin, importance=0.8 if a.profile else 0.5,
                             actor=f"cli:{who}")
        print(fid, file=out)
        return 0
    if a.cmd == "forget":
        what = store.forget(a.id, who, scope)
        print(f"forgot {what}" if what else "nothing visible with that id", file=out)
        return 0 if what else 1
    if a.cmd == "export":
        data = {"episodes": store.list_episodes(who, "person", owner=who, limit=1_000_000)}
        with store.db.scoped(who, "person") as conn:
            rows = conn.execute("SELECT id, kind, summary, topics, detail, created_at"
                                " FROM memory.facts WHERE username = %s", (who,)).fetchall()
        data["facts"] = [dict(zip(("id", "kind", "summary", "topics", "detail", "created_at"), r))
                         for r in rows]
        json.dump(data, out, default=_default, indent=1)
        print(file=out)
        return 0
    if a.cmd == "stats":
        print(json.dumps(store.stats(), default=_default), file=out)
        return 0
    if a.cmd in ("facts", "profile"):
        kinds = ("profile",) if a.cmd == "profile" else tuple(a.kind or ()) or None
        for f in store.facts(who, kinds=kinds):
            flags = "pinned " if f["pinned"] else ""
            print(f"{f['id']}  {f['kind']:8} {f['source']:10} {flags}{f['summary'][:90]}",
                  file=out)
        return 0
    if a.cmd == "jobs":
        if scope != "admin":
            raise StoreError("only an admin can list jobs")
        from . import jobs

        with store.db.scoped(None, "admin") as conn:
            for j in jobs.pending(conn):
                print(f"{j['kind']:8} {j['key']:24} {j['state']:8} tries {j['attempts']}"
                      f"  due {j['not_before']:%Y-%m-%d %H:%M}  {j['last_error'] or ''}"[:160],
                      file=out)
        return 0
    if a.cmd in ("extract", "reflect"):
        from . import jobs
        from .worker import Worker, extract_key, reflect_key

        if a.cmd == "extract" and a.all and scope != "admin":
            raise StoreError("only an admin can extract for everyone")
        users = [p["username"] for p in store.people()
                 if store.count_unextracted(p["username"])] \
            if a.cmd == "extract" and a.all else [who]
        if helper is None:
            from .helper import OllamaHelper

            helper = OllamaHelper(cfg.memory.helper_url, cfg.memory.helper_model,
                                  threads=cfg.memory.helper_threads)
        worker = Worker(store, helper, batch=cfg.memory.extract_batch, reflect_delay_s=0)
        with store.db.scoped(None, "admin") as conn:
            for u in users:
                key = (extract_key if a.cmd == "extract" else reflect_key)(u)
                jobs.enqueue(conn, a.cmd, {"username": u}, key=key)
                jobs.run_now(conn, key)
        n = worker.run_all()
        print(f"ran {n} jobs", file=out)
        return 0
    if a.cmd == "backfill":
        n = 0
        for ep, display in backfill_episodes(
                _read(a.transcript), enrolled=enrolled_speakers(cfg.speaker.registry_path),
                ignore=cfg.memory.ignore_speakers, include_unknown=a.include_unknown):
            if ep.username != "household":
                store.ensure_person(ep.username, display, voice_handle=display,
                                    is_admin=ep.username in cfg.memory.admins)
            if store.add_episode(ep) is not None:
                n += 1
        print(f"imported {n} new episodes", file=out)
        return 0
    return 2


if __name__ == "__main__":
    sys.exit(main())
