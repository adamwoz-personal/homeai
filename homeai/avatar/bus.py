"""Event bus: the daemon publishes events to local clients (avatar, debug tools).

Unix domain socket, newline-delimited JSON, one envelope per line:
    {"v": 1, "type": ..., "t_mono": ..., "source": ..., "seq": ..., "data": {...}}
``t_mono`` is ``time.monotonic()`` (system-wide CLOCK_MONOTONIC on Linux), so clients
schedule against it without clock sync.

On connect a client gets ``hello`` (version, capabilities) and ``snapshot`` (the latest
event of each state-like type), so a restarted renderer is correct immediately.

Back-pressure, per client: state-like events coalesce (latest wins, never lost), other
events are dropped oldest-first past ``max_queue``, and a client that hasn't drained its
backlog for ``slow_s`` is disconnected (it may reconnect). ``publish`` never blocks: it
only appends to in-memory queues and pokes the I/O thread.

Standard library only: the daemon imports this.
"""
from __future__ import annotations

import errno
import json
import logging
import os
import selectors
import socket
import threading
import time
from collections import deque
from pathlib import Path

log = logging.getLogger("homeai.avatar.bus")

PROTOCOL_VERSION = 1
# Latest-wins types: a client only ever needs the newest one.
COALESCED = frozenset({"state", "mode", "timer", "art", "script", "recording"})
MAX_QUEUE = 256
SLOW_S = 2.0
MAX_BACKLOG_BYTES = 4 << 20


def default_socket_path() -> Path:
    runtime = os.environ.get("XDG_RUNTIME_DIR") or f"/tmp/homeai-{os.getuid()}"
    return Path(runtime) / "homeai" / "events.sock"


def envelope(etype: str, data: dict, source: str, seq: int, t_mono: float | None = None) -> dict:
    """Build one protocol-v1 event."""
    if not etype or not isinstance(etype, str):
        raise ValueError("event type must be a non-empty string")
    if not isinstance(data, dict):
        raise ValueError("event data must be a dict")
    return {"v": PROTOCOL_VERSION, "type": etype,
            "t_mono": time.monotonic() if t_mono is None else t_mono,
            "source": source, "seq": seq, "data": data}


def encode(event: dict) -> bytes:
    return (json.dumps(event, separators=(",", ":"), ensure_ascii=False) + "\n").encode()


class ClientQueue:
    """Pure per-client outgoing queue with coalescing and drop-oldest (no I/O)."""

    def __init__(self, max_queue: int = MAX_QUEUE) -> None:
        if max_queue < 1:
            raise ValueError("max_queue must be >= 1")
        self.max_queue = max_queue
        self.items: deque[dict] = deque()
        self.dropped = 0

    def push(self, event: dict) -> None:
        if event["type"] in COALESCED:
            self.items = deque(e for e in self.items if e["type"] != event["type"])
        self.items.append(event)
        while len(self.items) > self.max_queue:
            victim = next((e for e in self.items if e["type"] not in COALESCED), None)
            if victim is None:
                victim = self.items[0]
            self.items.remove(victim)
            self.dropped += 1

    def pop_all(self) -> list[dict]:
        out = list(self.items)
        self.items.clear()
        return out

    def __len__(self) -> int:
        return len(self.items)


class _Client:
    def __init__(self, sock: socket.socket, max_queue: int) -> None:
        self.sock = sock
        self.queue = ClientQueue(max_queue)
        self.out = bytearray()
        self.inbuf = bytearray()
        self.blocked_since: float | None = None


# What a client may send (ART_FEATURES_PLAN F6): text typed in the art features window. Anything else is dropped.
ACCEPTED_IN = frozenset({"typed"})
MAX_IN_LINE = 8192


def parse_messages(client: "_Client", data: bytes) -> list[dict] | None:
    """Complete JSON lines a client sent, only the accepted types; None: a line too long (drop the client)."""
    client.inbuf += data
    *lines, rest = bytes(client.inbuf).split(b"\n")
    if len(rest) > MAX_IN_LINE or any(len(line) > MAX_IN_LINE for line in lines):
        return None
    client.inbuf = bytearray(rest)
    out = []
    for line in lines:
        try:
            msg = json.loads(line)
        except ValueError:
            continue
        if isinstance(msg, dict) and msg.get("type") in ACCEPTED_IN and isinstance(msg.get("data"), dict):
            out.append(msg)
    return out


class EventBus:
    """Server side. ``start()`` binds and runs the I/O thread; ``publish()`` is thread-safe."""

    def __init__(self, path: Path | str | None = None, source: str = "daemon",
                 max_queue: int = MAX_QUEUE, slow_s: float = SLOW_S,
                 capabilities: tuple[str, ...] = (), on_message=None) -> None:
        """``on_message(msg)``: called on the I/O thread for each accepted client message; must not block."""
        self.path = Path(path) if path else default_socket_path()
        self.on_message = on_message
        self.source = source
        self.max_queue = max_queue
        self.slow_s = slow_s
        self.capabilities = list(capabilities)
        self._lock = threading.Lock()
        self._seq = 0
        self._latest: dict[str, dict] = {}
        self._clients: dict[int, _Client] = {}
        self._sel: selectors.BaseSelector | None = None
        self._server: socket.socket | None = None
        self._wake_r: socket.socket | None = None
        self._wake_w: socket.socket | None = None
        self._thread: threading.Thread | None = None
        self._stop = threading.Event()

    # ---- lifecycle -------------------------------------------------------------
    def start(self) -> None:
        if self._thread:
            return
        self._prepare_path()
        server = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        old = os.umask(0o177)
        try:
            server.bind(str(self.path))
        finally:
            os.umask(old)
        os.chmod(self.path, 0o600)
        server.listen(8)
        server.setblocking(False)
        self._server = server
        self._wake_r, self._wake_w = socket.socketpair()
        self._wake_r.setblocking(False)
        self._wake_w.setblocking(False)
        self._sel = selectors.DefaultSelector()
        self._sel.register(server, selectors.EVENT_READ, "accept")
        self._sel.register(self._wake_r, selectors.EVENT_READ, "wake")
        self._stop.clear()
        self._thread = threading.Thread(target=self._run, name="homeai-event-bus", daemon=True)
        self._thread.start()
        log.info("event bus listening on %s", self.path)

    def stop(self) -> None:
        if not self._thread:
            return
        self._stop.set()
        self._poke()
        self._thread.join(timeout=2)
        self._thread = None
        for c in list(self._clients.values()):
            self._close(c)
        for s in (self._server, self._wake_r, self._wake_w):
            if s:
                s.close()
        self._wake_w = None
        if self._sel:
            self._sel.close()
        try:
            self.path.unlink()
        except FileNotFoundError:
            pass

    def _prepare_path(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
        if not self.path.exists():
            return
        probe = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        probe.settimeout(0.5)
        try:
            probe.connect(str(self.path))
        except OSError:
            self.path.unlink()          # stale: nobody is listening
            log.info("removed stale event socket %s", self.path)
            return
        finally:
            probe.close()
        raise RuntimeError(f"another event bus is already listening on {self.path}")

    # ---- publishing ------------------------------------------------------------
    def publish(self, etype: str, data: dict | None = None, t_mono: float | None = None) -> dict:
        """Queue an event for every client. Never blocks on I/O."""
        with self._lock:
            self._seq += 1
            event = envelope(etype, data or {}, self.source, self._seq, t_mono)
            if etype in COALESCED:
                self._latest[etype] = event
            for c in self._clients.values():
                c.queue.push(event)
        self._poke()
        return event

    def snapshot(self) -> dict:
        with self._lock:
            return dict(self._latest)

    @property
    def client_count(self) -> int:
        with self._lock:
            return len(self._clients)

    def _poke(self) -> None:
        if self._wake_w is None:
            return
        try:
            self._wake_w.send(b"\0")
        except OSError:
            pass                        # buffer full (already poked) or shutting down

    # ---- I/O thread ------------------------------------------------------------
    def _run(self) -> None:
        while not self._stop.is_set():
            try:
                ready = self._sel.select(timeout=0.25)
            except OSError:
                break
            for key, _mask in ready:
                if key.data == "accept":
                    self._accept()
                elif key.data == "wake":
                    try:
                        while self._wake_r.recv(4096):
                            pass
                    except OSError:
                        pass
                else:
                    self._drain_input(key.data)
            self._flush_all()

    def _accept(self) -> None:
        try:
            sock, _ = self._server.accept()
        except OSError:
            return
        sock.setblocking(False)
        client = _Client(sock, self.max_queue)
        with self._lock:
            hello = envelope("hello", {"protocol": PROTOCOL_VERSION,
                                       "capabilities": self.capabilities}, self.source, 0)
            snap = envelope("snapshot", {"events": dict(self._latest)}, self.source, self._seq)
            client.out += encode(hello) + encode(snap)
            self._clients[sock.fileno()] = client
            count = len(self._clients)
        self._sel.register(sock, selectors.EVENT_READ, client)
        log.info("event bus client connected (%d total)", count)

    def _drain_input(self, client: _Client) -> None:
        """Accepted messages go to ``on_message``; without one, input is read and discarded. EOF closes."""
        try:
            data = client.sock.recv(4096)
        except BlockingIOError:
            return
        except OSError:
            data = b""
        if not data:
            self._close(client)
            return
        if self.on_message is None:
            return
        messages = parse_messages(client, data)
        if messages is None:
            log.warning("event bus client sent a line over %d bytes; disconnected", MAX_IN_LINE)
            self._close(client)
            return
        for msg in messages:
            try:
                self.on_message(msg)
            except Exception:  # noqa: BLE001 - a bad handler must not stop the bus
                log.exception("event bus message handler failed")

    def _flush_all(self) -> None:
        now = time.monotonic()
        with self._lock:
            clients = list(self._clients.values())
            for c in clients:
                for e in c.queue.pop_all():
                    c.out += encode(e)
        for c in clients:
            self._flush(c, now)

    def _flush(self, client: _Client, now: float) -> None:
        if client.out:
            try:
                sent = client.sock.send(client.out)
                del client.out[:sent]
            except BlockingIOError:
                pass
            except OSError as exc:
                if exc.errno not in (errno.EPIPE, errno.ECONNRESET):
                    log.warning("event bus send failed: %s", exc)
                self._close(client)
                return
        if not client.out:
            client.blocked_since = None
            return
        if client.blocked_since is None:
            client.blocked_since = now
        if now - client.blocked_since > self.slow_s or len(client.out) > MAX_BACKLOG_BYTES:
            log.warning("event bus client not draining for %.1f s, disconnecting", self.slow_s)
            self._close(client)

    def _close(self, client: _Client) -> None:
        with self._lock:
            self._clients = {k: v for k, v in self._clients.items() if v is not client}
        try:
            if self._sel:
                self._sel.unregister(client.sock)
        except (KeyError, ValueError, OSError):
            pass
        client.sock.close()
