"""Event bus client: connect, read NDJSON events, reconnect with backoff.

Used by the avatar renderer and debug tools. Standard library only.
"""
from __future__ import annotations

import json
import logging
import socket
import threading
from pathlib import Path
from typing import Callable

from homeai.avatar.bus import PROTOCOL_VERSION, default_socket_path

log = logging.getLogger("homeai.avatar.client")

BACKOFF_MIN_S = 0.2
BACKOFF_MAX_S = 5.0
MAX_LINE = 1 << 20


def next_backoff(current: float) -> float:
    """Doubling backoff, clamped to [BACKOFF_MIN_S, BACKOFF_MAX_S]."""
    return min(BACKOFF_MAX_S, max(BACKOFF_MIN_S, current * 2))


def parse_lines(buffer: bytearray) -> list[dict]:
    """Pop complete lines from ``buffer`` and decode them; bad lines are skipped.

    Raises ValueError if an unterminated line exceeds MAX_LINE (a broken stream).
    """
    events = []
    while (nl := buffer.find(b"\n")) >= 0:
        line = bytes(buffer[:nl])
        del buffer[:nl + 1]
        if not line.strip():
            continue
        try:
            event = json.loads(line)
        except (json.JSONDecodeError, UnicodeDecodeError):
            log.warning("event bus: skipping undecodable line")
            continue
        if isinstance(event, dict) and "type" in event:
            events.append(event)
    if len(buffer) > MAX_LINE:
        raise ValueError("event bus line too long")
    return events


class VersionMismatch(Exception):
    pass


def check_hello(event: dict) -> None:
    """The first event must be a hello with our protocol version."""
    if event.get("type") != "hello":
        raise VersionMismatch(f"expected hello, got {event.get('type')!r}")
    proto = event.get("data", {}).get("protocol")
    if proto != PROTOCOL_VERSION:
        raise VersionMismatch(f"server speaks protocol {proto}, we speak {PROTOCOL_VERSION}")


class BusClient:
    """Background reader. ``on_event(event)`` and ``on_status(connected: bool)`` run on its thread."""

    def __init__(self, on_event: Callable[[dict], None],
                 on_status: Callable[[bool], None] | None = None,
                 path: Path | str | None = None) -> None:
        self.path = Path(path) if path else default_socket_path()
        self.on_event = on_event
        self.on_status = on_status or (lambda _c: None)
        self.connected = False
        self.version_error: str | None = None
        self._stop = threading.Event()
        self._sock: socket.socket | None = None
        self._send_lock = threading.Lock()
        self._thread: threading.Thread | None = None

    def start(self) -> None:
        if self._thread:
            return
        self._stop.clear()
        self._thread = threading.Thread(target=self._run, name="homeai-bus-client", daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        sock = self._sock
        if sock:
            try:
                sock.shutdown(socket.SHUT_RDWR)
            except OSError:
                pass
        if self._thread:
            self._thread.join(timeout=2)
            self._thread = None

    def send(self, etype: str, data: dict) -> bool:
        """Send one message to the daemon (only types the bus accepts, e.g. "typed"); False if not connected."""
        sock = self._sock
        if sock is None or not self.connected:
            return False
        line = (json.dumps({"type": etype, "data": data}, ensure_ascii=False) + "\n").encode()
        try:
            with self._send_lock:
                sock.sendall(line)
        except OSError:
            return False
        return True

    def _set_connected(self, value: bool) -> None:
        if value != self.connected:
            self.connected = value
            try:
                self.on_status(value)
            except Exception:
                log.exception("event bus status callback failed")

    def _run(self) -> None:
        backoff = BACKOFF_MIN_S
        while not self._stop.is_set():
            try:
                self._session()
                backoff = BACKOFF_MIN_S         # had a working session
            except VersionMismatch as exc:
                self.version_error = str(exc)
                log.error("event bus: %s; not reconnecting", exc)
                self._set_connected(False)
                return
            except (OSError, ValueError) as exc:
                log.debug("event bus: %s", exc)
            self._set_connected(False)
            if self._stop.wait(backoff):
                return
            backoff = next_backoff(backoff)

    def _session(self) -> None:
        sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        self._sock = sock
        try:
            sock.connect(str(self.path))
            buffer = bytearray()
            greeted = False
            while not self._stop.is_set():
                chunk = sock.recv(65536)
                if not chunk:
                    return
                buffer += chunk
                for event in parse_lines(buffer):
                    if not greeted:
                        check_hello(event)
                        greeted = True
                        self._set_connected(True)
                        continue
                    try:
                        self.on_event(event)
                    except Exception:
                        log.exception("event bus handler failed on %s", event.get("type"))
        finally:
            self._sock = None
            sock.close()
