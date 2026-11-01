"""Long-term memory as MCP tools, for coding/assistant agents (Phase 3).

A separate stdio server from ``homeai.mcp_server`` (the voice agent's home
tools) so the voice agent never sees it. Each client is registered with a
fixed scope and user on its command line::

    homeai-memory-mcp --scope admin --user adam --client builder

Scope is enforced twice: here (arguments can't name someone else's memory in
person scope) and by row-level security in Postgres, as role ``homeai_mcp``.
The scope settings are per-transaction GUCs any local code could set, so this
guards against agent mistakes and injected arguments, not local programs.

Every write is audited with the client name; deletes record a reason word,
never text.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
import uuid
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Any, Callable

from ..config import Config
from . import policy

log = logging.getLogger(__name__)

PROTOCOL_VERSION = "2024-11-05"
SERVER_INFO = {"name": "homeai-memory", "version": "1.0.0"}
PARSE_ERROR, INVALID_REQUEST, METHOD_NOT_FOUND, INTERNAL_ERROR = -32700, -32600, -32601, -32603

SCOPES = ("admin", "person")
MAX_LIMIT = 20
MAX_TOPICS = 8
MAX_DETAIL_BYTES = 4096
MAX_TEXT = 400
MAX_RESULT_CHARS = 16000
# Kinds an agent may not write: instructions steer every future reply, and
# profile/summary lines are rebuilt from facts by the reflection job.
BLOCKED_KINDS = ("instruction", "profile", "summary")
DELETE_REASONS = ("wrong", "outdated", "duplicate", "requested", "dead_end")
HOUSEHOLD = "household"


class ToolError(Exception):
    """An expected, explainable failure: returned to the agent as tool output."""


# -- tool schemas ------------------------------------------------------------------

_ID = {"type": "string", "description": "A memory id (uuid) from memory_search"}
_USER = {"type": "string", "description": "Whose memory (default: yours and the household's)"}

TOOLS: list[dict[str, Any]] = [
    {"name": "memory_search",
     "description": "Search long-term memory by meaning. Returns facts (and, if asked, raw"
                    " conversation episodes) ranked by similarity 0-1; judge relevance yourself.",
     "inputSchema": {"type": "object", "required": ["query"], "properties": {
         "query": {"type": "string"},
         "username": _USER,
         "topics": {"type": "array", "items": {"type": "string"}},
         "kinds": {"type": "array", "items": {"type": "string"}},
         "since": {"type": "string", "description": "ISO date or datetime"},
         "until": {"type": "string", "description": "ISO date or datetime"},
         "include_episodes": {"type": "boolean"},
         "limit": {"type": "integer", "minimum": 1, "maximum": MAX_LIMIT}}}},
    {"name": "memory_get",
     "description": "One memory in full: detail, the conversations it came from, and links.",
     "inputSchema": {"type": "object", "required": ["id"], "properties": {"id": _ID}}},
    {"name": "memory_add",
     "description": "Remember a fact. If a near-identical memory exists nothing is added and"
                    " duplicate_of is returned; related memories come back as 'similar' so you"
                    " can memory_supersede one instead if the new fact replaces it.",
     "inputSchema": {"type": "object", "required": ["summary"], "properties": {
         "summary": {"type": "string", "description": "One self-contained sentence, ≤300 chars"},
         "kind": {"type": "string", "default": "fact"},
         "topics": {"type": "array", "items": {"type": "string"}, "maxItems": MAX_TOPICS},
         "importance": {"type": "number", "minimum": 0, "maximum": 1},
         "pinned": {"type": "boolean"},
         "detail": {"type": "object", "description": f"Extra JSON, ≤{MAX_DETAIL_BYTES} bytes"},
         "thread": {"type": "string", "description": "Name of a line of work to group facts"},
         "username": {"type": "string", "description": "Whose memory (default: yours)"}}}},
    {"name": "memory_update",
     "description": "Edit a memory in place (typos, topics, importance, pinning). For a fact"
                    " that is no longer true use memory_supersede so history is kept.",
     "inputSchema": {"type": "object", "required": ["id"], "properties": {
         "id": _ID, "summary": {"type": "string"}, "kind": {"type": "string"},
         "topics": {"type": "array", "items": {"type": "string"}, "maxItems": MAX_TOPICS},
         "importance": {"type": "number", "minimum": 0, "maximum": 1},
         "pinned": {"type": "boolean"}, "detail": {"type": "object"}}}},
    {"name": "memory_supersede",
     "description": "Replace a fact that changed or was wrong with a corrected one.",
     "inputSchema": {"type": "object", "required": ["id", "summary"], "properties": {
         "id": _ID, "summary": {"type": "string"}, "detail": {"type": "object"}}}},
    {"name": "memory_delete",
     "description": "Delete one memory, or every fact in a thread.",
     "inputSchema": {"type": "object", "required": ["reason"], "properties": {
         "id": _ID, "thread": {"type": "string", "description": "Thread name"},
         "username": {"type": "string", "description": "Thread owner (default: you)"},
         "reason": {"type": "string", "enum": list(DELETE_REASONS)}}}},
    {"name": "memory_topics",
     "description": "Topics in memory with how many facts use each.",
     "inputSchema": {"type": "object", "properties": {
         "username": _USER, "prefix": {"type": "string"}}}},
    {"name": "memory_health",
     "description": "Memory counts, pending background jobs and the latest backup.",
     "inputSchema": {"type": "object", "properties": {}}},
]
_SCHEMAS = {t["name"]: t["inputSchema"] for t in TOOLS}
_PY = {"string": str, "integer": int, "number": (int, float), "boolean": bool,
       "array": list, "object": dict}


def validate(name: str, args: Any) -> dict[str, Any]:
    """Check arguments against the tool's schema (the subset these tools use)."""
    schema = _SCHEMAS[name]
    if not isinstance(args, dict):
        raise ToolError("arguments must be an object")
    props = schema.get("properties", {})
    unknown = sorted(set(args) - set(props))
    if unknown:
        raise ToolError(f"unknown argument(s): {', '.join(unknown)}")
    for req in schema.get("required", ()):
        if args.get(req) in (None, ""):
            raise ToolError(f"{req} is required")
    for key, val in args.items():
        if val is None:
            continue
        spec = props[key]
        want = _PY[spec["type"]]
        # bool is an int in Python; a number field must not accept true/false.
        if not isinstance(val, want) or (spec["type"] in ("integer", "number")
                                         and isinstance(val, bool)):
            raise ToolError(f"{key} must be {'an' if spec['type'][0] in 'aeio' else 'a'}"
                            f" {spec['type']}")
        if "enum" in spec and val not in spec["enum"]:
            raise ToolError(f"{key} must be one of: {', '.join(spec['enum'])}")
        if "minimum" in spec and val < spec["minimum"]:
            raise ToolError(f"{key} must be ≥ {spec['minimum']}")
        if "maximum" in spec and val > spec["maximum"]:
            raise ToolError(f"{key} must be ≤ {spec['maximum']}")
        if spec["type"] == "array":
            if "maxItems" in spec and len(val) > spec["maxItems"]:
                raise ToolError(f"{key} takes at most {spec['maxItems']} items")
            if not all(isinstance(v, str) for v in val):
                raise ToolError(f"{key} must be a list of strings")
    return {k: v for k, v in args.items() if v is not None}


def _when(text: str, key: str) -> datetime:
    try:
        if len(text) == 10:
            d = date.fromisoformat(text)
            return datetime(d.year, d.month, d.day, tzinfo=timezone.utc)
        t = datetime.fromisoformat(text)
    except ValueError:
        raise ToolError(f"{key} must be an ISO date like 2025-01-31") from None
    return t if t.tzinfo else t.replace(tzinfo=timezone.utc)


def _uuid(text: str, key: str = "id") -> uuid.UUID:
    try:
        return uuid.UUID(str(text))
    except ValueError:
        raise ToolError(f"{key} is not a memory id") from None


def _clip(text: Any, n: int = MAX_TEXT) -> Any:
    return text[: n - 1] + "…" if isinstance(text, str) and len(text) > n else text


def _json(o: Any) -> str:
    if isinstance(o, datetime):
        return o.isoformat(timespec="seconds")
    return str(o)


def _redact_values(o: Any) -> Any:
    """Secrets out of every string in a JSON value (keys included)."""
    if isinstance(o, str):
        return policy.redact(o)
    if isinstance(o, dict):
        return {policy.redact(str(k)): _redact_values(v) for k, v in o.items()}
    if isinstance(o, list):
        return [_redact_values(v) for v in o]
    return o


def render(result: Any) -> str:
    """JSON for the agent, capped so one call can't flood its context."""
    text = json.dumps(result, default=_json, ensure_ascii=False)
    if len(text) > MAX_RESULT_CHARS:
        text = text[:MAX_RESULT_CHARS] + "… (truncated: narrow the query)"
    return text


# -- the tools -----------------------------------------------------------------------

class MemoryTools:
    """The eight memory tools for one client, user and scope."""

    def __init__(self, store, *, scope: str, user: str, client: str,
                 backup_dir: Path | None = None) -> None:
        if scope not in SCOPES:
            raise ValueError(f"unknown scope {scope!r}")
        self.store, self.scope, self.user = store, scope, user
        self.via = f"mcp:{client}"
        self.backup_dir = backup_dir
        self.handlers: dict[str, Callable[[dict[str, Any]], Any]] = {
            "memory_search": self.search, "memory_get": self.get, "memory_add": self.add,
            "memory_update": self.update, "memory_supersede": self.supersede,
            "memory_delete": self.delete, "memory_topics": self.topics,
            "memory_health": self.health}

    def check(self) -> None:
        """Refuse to serve a user the database doesn't know, or admin scope
        to someone who isn't an admin."""
        p = self.store.person(self.user)
        if p is None:
            raise ToolError(f"{self.user} is not a person in memory (homeai-memory people)")
        if self.scope == "admin" and not p["is_admin"]:
            raise ToolError(f"{self.user} is not an admin: register this client with"
                            " --scope person")

    def call(self, name: str, args: Any) -> Any:
        handler = self.handlers.get(name)
        if handler is None:
            raise ToolError(f"unknown tool: {name}")
        return handler(validate(name, args or {}))

    # -- scope helpers --

    def _readable(self, username: str | None) -> list[str] | None:
        """Owners a read may cover. None = everyone (admin, no filter)."""
        if not username:
            return None if self.scope == "admin" else [self.user, HOUSEHOLD]
        if self.scope != "admin" and username not in (self.user, HOUSEHOLD):
            raise ToolError("you can only read your own and household memories")
        return [username]

    def _writable(self, username: str | None) -> str:
        """The owner a write goes to."""
        username = username or self.user
        if self.scope != "admin" and username != self.user:
            raise ToolError("you can only change your own memories")
        if username != HOUSEHOLD and self.store.person(username) is None:
            raise ToolError(f"no person {username!r} in memory")
        return username

    def _own(self, object_id: uuid.UUID) -> dict[str, Any]:
        """A visible, active fact the caller may change."""
        got = self.store.get(self.user, self.scope, object_id)
        if got is None or got["type"] != "fact" or got["state"] != "active":
            raise ToolError("no active memory you can see has that id")
        self._writable(got["username"])
        return got

    @staticmethod
    def _topics(names: list[str] | None) -> list[str] | None:
        from .extract import slug
        if names is None:
            return None
        out: list[str] = []
        for n in names:
            s = slug(n)
            if s and s not in out:
                out.append(s)
        return out

    @staticmethod
    def _detail(detail: dict[str, Any] | None) -> dict[str, Any] | None:
        if detail is None:
            return None
        raw = json.dumps(detail, ensure_ascii=False)
        if len(raw.encode()) > MAX_DETAIL_BYTES:
            raise ToolError(f"detail is over {MAX_DETAIL_BYTES} bytes: store less")
        return _redact_values(detail)

    @staticmethod
    def _kind(kind: str | None) -> str | None:
        from .store import FACT_KINDS
        if kind is None:
            return None
        if kind in BLOCKED_KINDS:
            raise ToolError(f"agents can't write {kind} memories")
        if kind not in FACT_KINDS:
            allowed = [k for k in FACT_KINDS if k not in BLOCKED_KINDS]
            raise ToolError(f"kind must be one of: {', '.join(allowed)}")
        return kind

    def _vec(self, text: str):
        return self.store.embedder.embed([text], kind="query")[0]

    # -- handlers --

    def search(self, a: dict[str, Any]) -> dict[str, Any]:
        rows = self.store.search(
            self.user, self.scope, self._vec(a["query"]), owners=self._readable(a.get("username")),
            topics=tuple(self._topics(a.get("topics")) or ()), kinds=tuple(a.get("kinds", ())),
            since=a.get("since") and _when(a["since"], "since"),
            until=a.get("until") and _when(a["until"], "until"),
            include_episodes=a.get("include_episodes", False), limit=a.get("limit", 10))
        for r in rows:
            r["text"] = _clip(r["text"])
            r["sim"] = round(float(r["sim"]), 3)
        return {"results": rows}

    def get(self, a: dict[str, Any]) -> dict[str, Any]:
        got = self.store.get(self.user, self.scope, _uuid(a["id"]))
        if got is None:
            raise ToolError("no memory you can see has that id")
        for s in got.get("sources", ()):
            s["said"] = _clip(s["said"])
        return got

    def add(self, a: dict[str, Any]) -> dict[str, Any]:
        from .extract import ASK_SIM, NOOP_SIM
        from .store import thread_id
        owner = self._writable(a.get("username"))
        kind = self._kind(a.get("kind", "fact"))
        summary = policy.redact(" ".join(a["summary"].split()))
        if not summary:
            raise ToolError("summary is empty")
        if len(summary) > 300:
            raise ToolError("summary must be ≤300 characters: put the rest in detail")
        detail = self._detail(a.get("detail"))
        near = self.store.nearest_facts(owner, self.store.embedder.embed([summary], kind="doc")[0])
        if near and near[0]["sim"] >= NOOP_SIM:
            return {"added": None, "duplicate_of": {"id": near[0]["id"],
                                                    "summary": near[0]["summary"]}}
        thread = (self._topics([a["thread"]]) or [None])[0] if a.get("thread") else None
        if a.get("thread") and not thread:
            raise ToolError("thread is not a name")
        tid = thread_id(owner, thread) if thread else None
        fid = self.store.add_fact(owner, summary, kind=kind,
                                  topics=tuple(self._topics(a.get("topics")) or ()),
                                  importance=a.get("importance", 0.5), pinned=a.get("pinned", False),
                                  trust="household", source="mcp", actor=self.via,
                                  detail=detail, thread_id=tid)
        similar = [{"id": n["id"], "summary": n["summary"], "sim": round(float(n["sim"]), 3)}
                   for n in near if ASK_SIM <= n["sim"] < NOOP_SIM]
        return {"added": fid, "thread_id": tid, "similar": similar}

    def update(self, a: dict[str, Any]) -> dict[str, Any]:
        fid = _uuid(a["id"])
        if len(a) == 1:
            raise ToolError("nothing to change")
        self._own(fid)
        from .store import StoreError
        try:
            self.store.update_fact(self.user, self.scope, fid, summary=a.get("summary"),
                                   topics=self._topics(a.get("topics")),
                                   importance=a.get("importance"), pinned=a.get("pinned"),
                                   detail=self._detail(a.get("detail")),
                                   kind=self._kind(a.get("kind")), via=self.via)
        except StoreError as exc:
            raise ToolError(str(exc)) from None
        return {"updated": fid}

    def supersede(self, a: dict[str, Any]) -> dict[str, Any]:
        from .store import StoreError
        fid = _uuid(a["id"])
        old = self._own(fid)
        if old["source"] == "reflection" or old["kind"] in ("profile", "summary"):
            raise ToolError("summaries and profile lines are rebuilt from facts:"
                            " change or delete those")
        try:
            new = self.store.supersede_fact(old["username"], fid, a["summary"], actor=self.via,
                                            source="mcp", trust="household",
                                            detail=self._detail(a.get("detail")))
        except StoreError as exc:
            raise ToolError(str(exc)) from None
        return {"added": new, "superseded": fid}

    def delete(self, a: dict[str, Any]) -> dict[str, Any]:
        from .extract import slug
        from .store import thread_id
        if bool(a.get("id")) == bool(a.get("thread")):
            raise ToolError("give either id or thread")
        if a.get("id"):
            what = self.store.forget(_uuid(a["id"]), self.user, self.scope, via=self.via,
                                     reason=a["reason"])
            if what is None:
                raise ToolError("no memory of yours has that id")
            return {"deleted": what, "id": a["id"]}
        owner = self._writable(a.get("username"))
        name = slug(a["thread"])
        if not name:
            raise ToolError("thread is not a name")
        n = self.store.forget_thread(self.user, self.scope, thread_id(owner, name),
                                     via=self.via, reason=a["reason"])
        if not n:
            raise ToolError(f"no facts of {owner}'s in thread {name!r}")
        return {"deleted": "thread", "facts": n}

    def topics(self, a: dict[str, Any]) -> dict[str, Any]:
        rows = self.store.topic_counts(self.user, self.scope,
                                       owners=self._readable(a.get("username")),
                                       prefix=(a.get("prefix") or "").lower())
        return {"topics": [{"topic": t, "facts": n} for t, n in rows]}

    def health(self, a: dict[str, Any]) -> dict[str, Any]:
        if self.scope != "admin":
            return {"yours": self.store.count_for(self.user)}
        from . import backup, jobs
        out: dict[str, Any] = self.store.stats()
        with self.store.db.scoped(self.user, "admin") as conn:
            pend = jobs.pending(conn)
        out["jobs"] = {"pending": len(pend),
                       "failing": sum(1 for j in pend if j["last_error"])}
        found = backup.dumps(self.backup_dir) if self.backup_dir else []
        if found:
            age = time.time() - found[-1].stat().st_mtime
            out["backup"] = {"newest": found[-1].name, "age_hours": round(age / 3600, 1)}
        else:
            out["backup"] = None
        return out


# -- JSON-RPC over stdio ------------------------------------------------------------------

def _result(request_id: Any, result: dict[str, Any]) -> dict[str, Any]:
    return {"jsonrpc": "2.0", "id": request_id, "result": result}


def _error(request_id: Any, code: int, message: str) -> dict[str, Any]:
    return {"jsonrpc": "2.0", "id": request_id, "error": {"code": code, "message": message}}


def _text(text: str, is_error: bool = False) -> dict[str, Any]:
    return {"content": [{"type": "text", "text": text}], "isError": is_error}


class Server:
    """Dispatches messages. The tools are built on the first call so a down
    database doesn't stop the client starting: it gets an error per call."""

    def __init__(self, make_tools: Callable[[], MemoryTools]) -> None:
        self._make = make_tools
        self._tools: MemoryTools | None = None

    def tools(self) -> MemoryTools:
        if self._tools is None:
            tools = self._make()
            try:
                tools.check()
            except BaseException:
                _close(tools)
                raise
            self._tools = tools
        return self._tools

    def close(self) -> None:
        if self._tools is not None:
            _close(self._tools)
            self._tools = None

    def handle(self, message: dict[str, Any]) -> dict[str, Any] | None:
        method = message.get("method")
        request_id = message.get("id")
        is_notification = "id" not in message
        if not method:
            return None if is_notification else _error(request_id, INVALID_REQUEST,
                                                        "missing method")
        if method == "initialize":
            return _result(request_id, {"protocolVersion": PROTOCOL_VERSION,
                                        "capabilities": {"tools": {}}, "serverInfo": SERVER_INFO})
        if method == "tools/list":
            return _result(request_id, {"tools": TOOLS})
        if method == "tools/call":
            params = message.get("params") or {}
            name = params.get("name")
            if name not in _SCHEMAS:
                return _result(request_id, _text(f"unknown tool: {name}", is_error=True))
            try:
                return _result(request_id, _text(render(
                    self.tools().call(name, params.get("arguments")))))
            except ToolError as exc:
                return _result(request_id, _text(str(exc), is_error=True))
            except Exception as exc:  # noqa: BLE001 - a tool must not kill the server
                log.exception("tool %s failed", name)
                return _result(request_id, _text(f"{name} failed: {type(exc).__name__}: {exc}",
                                                 is_error=True))
        if is_notification:
            return None
        return _error(request_id, METHOD_NOT_FOUND, f"unknown method: {method}")

    def serve(self, stdin=None, stdout=None) -> int:
        stdin = stdin or sys.stdin
        stdout = stdout or sys.stdout
        for line in stdin:
            line = line.strip()
            if not line:
                continue
            try:
                message = json.loads(line)
            except json.JSONDecodeError:
                response = _error(None, PARSE_ERROR, "invalid JSON")
            else:
                if not isinstance(message, dict):
                    response = _error(None, INVALID_REQUEST, "expected an object")
                else:
                    try:
                        response = self.handle(message)
                    except Exception as exc:  # noqa: BLE001
                        log.exception("dispatch failed")
                        response = _error(message.get("id"), INTERNAL_ERROR,
                                          f"internal error: {exc}")
            if response is not None:
                stdout.write(json.dumps(response) + "\n")
                stdout.flush()
        return 0


class _Off:
    def check(self) -> None:
        raise ToolError("long-term memory is off (HOMEAI_LTM=0)")


def _close(tools) -> None:
    db = getattr(getattr(tools, "store", None), "db", None)
    if db is not None:
        db.close()


def build(cfg: Config, *, scope: str, user: str, client: str, embedder=None) -> MemoryTools:
    from .db import Db
    from .embed import DEFAULT_MODEL, OnnxEmbedder
    from .store import PostgresStore

    if not cfg.memory.enabled:
        return _Off()  # type: ignore[return-value]
    db = Db(cfg.memory.dsn, role="homeai_mcp").open()
    store = PostgresStore(db, embedder or OnnxEmbedder(cfg.memory.embed_model or DEFAULT_MODEL))
    return MemoryTools(store, scope=scope, user=user, client=client,
                       backup_dir=Path(cfg.memory.backup_dir).expanduser())


def parse_args(argv: list[str] | None) -> argparse.Namespace:
    ap = argparse.ArgumentParser(prog="homeai-memory-mcp",
                                 description="Long-term memory MCP server (stdio)")
    ap.add_argument("--scope", choices=SCOPES, required=True)
    ap.add_argument("--user", required=True, help="whose memory this client acts as")
    ap.add_argument("--client", required=True, help="name recorded in the audit log")
    ns = ap.parse_args(argv)
    if not policy.USERNAME_RE.fullmatch(ns.user):
        ap.error(f"bad username {ns.user!r}")
    if not policy.USERNAME_RE.fullmatch(ns.client):
        ap.error(f"bad client name {ns.client!r}")
    return ns


def main(argv: list[str] | None = None, *, cfg: Config | None = None, embedder=None,
         stdin=None, stdout=None) -> int:
    ns = parse_args(argv)
    cfg = cfg or Config()
    server = Server(lambda: build(cfg, scope=ns.scope, user=ns.user, client=ns.client,
                                  embedder=embedder))
    try:
        return server.serve(stdin, stdout)
    finally:
        server.close()


if __name__ == "__main__":
    logging.basicConfig(level=logging.WARNING, stream=sys.stderr)
    raise SystemExit(main())
