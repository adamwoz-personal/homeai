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


def _store(cfg: Config, embedder=None):
    from .db import Db
    from .embed import DEFAULT_MODEL, OnnxEmbedder
    from .store import PostgresStore

    db = Db(cfg.memory.dsn, role="homeai_app").open()
    return PostgresStore(db, embedder or OnnxEmbedder(cfg.memory.embed_model or DEFAULT_MODEL))


def _scope(store, who: str) -> str:
    return "admin" if store.is_admin(who) else "person"


def _default(o):
    return o.isoformat() if isinstance(o, datetime) else str(o)


def main(argv: list[str] | None = None, *, cfg: Config | None = None, embedder=None,
         out=None) -> int:
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
    p = sub.add_parser("backfill")
    p.add_argument("--dry-run", action="store_true")
    p.add_argument("--include-unknown", action="store_true",
                   help="unrecognised turns go to 'household' (may include test voices)")
    p.add_argument("--transcript", default=cfg.transcript.path)
    a = ap.parse_args(argv)

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
        return _run(a, cfg, store, out)
    except StoreError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    finally:
        store.db.close()


def _read(path: str) -> list[str]:
    try:
        return Path(path).expanduser().read_text(encoding="utf-8").splitlines()
    except OSError as exc:
        raise SystemExit(f"cannot read transcript: {exc}") from exc


def _run(a, cfg: Config, store, out) -> int:
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
