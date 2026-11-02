"""Avatar event bus: envelope, queueing, socket server behaviour and the reconnecting client."""

import json
import os
import socket
import stat
import threading
import time

import pytest

from homeai.avatar.bus import ClientQueue, EventBus, envelope
from homeai.avatar.client import (BACKOFF_MAX_S, BACKOFF_MIN_S, BusClient, VersionMismatch,
                                  check_hello, next_backoff, parse_lines)


def wait_until(cond, timeout=3.0):
    end = time.monotonic() + timeout
    while time.monotonic() < end:
        if cond():
            return True
        time.sleep(0.01)
    return cond()


@pytest.fixture
def sock_path(tmp_path):
    return tmp_path / "ev.sock"


@pytest.fixture
def server(sock_path):
    b = EventBus(sock_path, slow_s=0.3)
    b.start()
    yield b
    b.stop()


def read_events(sock, n, timeout=2.0):
    sock.settimeout(timeout)
    buf = bytearray()
    out = []
    while len(out) < n:
        chunk = sock.recv(65536)
        if not chunk:
            break
        buf += chunk
        out += parse_lines(buf)
    return out


def connect(path):
    s = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    s.connect(str(path))
    return s


# ---- pure parts -----------------------------------------------------------------

def test_envelope_fields():
    e = envelope("state", {"state": "idle"}, "daemon", 7, t_mono=12.5)
    assert e == {"v": 1, "type": "state", "t_mono": 12.5, "source": "daemon", "seq": 7,
                 "data": {"state": "idle"}}
    assert envelope("x", {}, "d", 1)["t_mono"] == pytest.approx(time.monotonic(), abs=1)
    with pytest.raises(ValueError):
        envelope("", {}, "d", 1)
    with pytest.raises(ValueError):
        envelope("x", [], "d", 1)


def _ev(etype, n):
    return {"type": etype, "seq": n}


def test_queue_coalesces_state_like_events():
    q = ClientQueue()
    q.push(_ev("state", 1))
    q.push(_ev("caption", 2))
    q.push(_ev("state", 3))
    assert [e["seq"] for e in q.pop_all()] == [2, 3]
    assert len(q) == 0


def test_queue_drops_oldest_bulky_never_state():
    q = ClientQueue(max_queue=3)
    q.push(_ev("state", 1))
    for n in range(2, 6):
        q.push(_ev("utterance", n))
    assert [e["seq"] for e in q.pop_all()] == [1, 4, 5]
    assert q.dropped == 2


def test_queue_all_coalesced_overflow_drops_oldest():
    q = ClientQueue(max_queue=2)
    for n, t in enumerate(["state", "mode", "timer"]):
        q.push(_ev(t, n))
    assert [e["type"] for e in q.pop_all()] == ["mode", "timer"]
    with pytest.raises(ValueError):
        ClientQueue(0)


def test_parse_lines_partial_bad_and_oversize():
    buf = bytearray(b'{"type":"a"}\nnot json\n\n[1]\n{"type":"b"')
    assert [e["type"] for e in parse_lines(buf)] == ["a"]
    assert buf == bytearray(b'{"type":"b"')
    buf += b"}\n"
    assert [e["type"] for e in parse_lines(buf)] == ["b"]
    with pytest.raises(ValueError):
        parse_lines(bytearray(b"x" * ((1 << 20) + 1)))


def test_check_hello_and_backoff():
    check_hello({"type": "hello", "data": {"protocol": 1}})
    with pytest.raises(VersionMismatch):
        check_hello({"type": "hello", "data": {"protocol": 2}})
    with pytest.raises(VersionMismatch):
        check_hello({"type": "state", "data": {}})
    assert next_backoff(0) == BACKOFF_MIN_S
    assert next_backoff(1.0) == 2.0
    assert next_backoff(100) == BACKOFF_MAX_S


# ---- server ---------------------------------------------------------------------

def test_hello_and_snapshot_on_connect(server, sock_path):
    server.publish("state", {"state": "thinking"})
    server.publish("caption", {"text": "hi"})          # not state-like: not in snapshot
    s = connect(sock_path)
    hello, snap = read_events(s, 2)
    assert hello["type"] == "hello" and hello["data"]["protocol"] == 1
    assert snap["type"] == "snapshot"
    assert set(snap["data"]["events"]) == {"state"}
    assert snap["data"]["events"]["state"]["data"] == {"state": "thinking"}
    s.close()


def test_events_delivered_in_order(server, sock_path):
    s = connect(sock_path)
    read_events(s, 2)
    assert wait_until(lambda: server.client_count == 1)
    for i in range(5):
        server.publish("caption", {"i": i})
    got = read_events(s, 5)
    assert [e["data"]["i"] for e in got] == list(range(5))
    assert [e["seq"] for e in got] == sorted(e["seq"] for e in got)
    s.close()


def test_socket_mode_600_and_cleanup(sock_path):
    b = EventBus(sock_path)
    b.start()
    assert stat.S_IMODE(os.stat(sock_path).st_mode) == 0o600
    b.stop()
    assert not sock_path.exists()


def test_stale_socket_removed_live_one_refused(sock_path):
    stale = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    stale.bind(str(sock_path))
    stale.close()                                       # file left behind, nobody listening
    b = EventBus(sock_path)
    b.start()
    try:
        with pytest.raises(RuntimeError, match="already listening"):
            EventBus(sock_path).start()
    finally:
        b.stop()


def test_client_disconnect_is_cleaned_up(server, sock_path):
    s = connect(sock_path)
    assert wait_until(lambda: server.client_count == 1)
    s.close()
    assert wait_until(lambda: server.client_count == 0)


def test_slow_client_disconnected_and_publish_never_blocks(server, sock_path):
    s = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    s.setsockopt(socket.SOL_SOCKET, socket.SO_RCVBUF, 4096)
    s.connect(str(sock_path))                           # never reads
    assert wait_until(lambda: server.client_count == 1)
    payload = {"blob": "x" * 4000}
    start = time.monotonic()
    for _ in range(300):                                # ~1.2 MB: under the backlog cap,
        server.publish("utterance", payload)            # so only the slow_s rule can fire
    assert time.monotonic() - start < 1.0
    assert wait_until(lambda: server.client_count == 0, timeout=3)
    s.close()


def test_publish_without_start_is_harmless(sock_path):
    b = EventBus(sock_path)
    assert b.publish("state", {"state": "idle"})["seq"] == 1
    assert b.snapshot()["state"]["data"] == {"state": "idle"}


# ---- client ---------------------------------------------------------------------

def test_client_receives_and_reconnects_after_server_restart(sock_path, monkeypatch):
    monkeypatch.setattr("homeai.avatar.client.BACKOFF_MIN_S", 0.05)
    got, status = [], []
    c = BusClient(got.append, status.append, path=sock_path)
    c.start()
    try:
        b1 = EventBus(sock_path)
        b1.start()
        assert wait_until(lambda: c.connected)
        b1.publish("caption", {"n": 1})
        assert wait_until(lambda: any(e["type"] == "caption" for e in got))
        b1.stop()
        assert wait_until(lambda: not c.connected)
        b2 = EventBus(sock_path)
        b2.start()
        b2.publish("state", {"state": "idle"})
        assert wait_until(lambda: c.connected, timeout=6)
        assert wait_until(lambda: got[-1]["type"] == "snapshot"
                          and got[-1]["data"]["events"]["state"]["data"] == {"state": "idle"})
        b2.stop()
    finally:
        c.stop()
    assert status[:3] == [True, False, True]


def test_client_stops_on_version_mismatch(sock_path):
    srv = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    srv.bind(str(sock_path))
    srv.listen(1)
    accepted = []

    def serve():
        conn, _ = srv.accept()
        accepted.append(conn)
        conn.sendall((json.dumps({"type": "hello", "data": {"protocol": 99}}) + "\n").encode())

    t = threading.Thread(target=serve, daemon=True)
    t.start()
    c = BusClient(lambda e: None, path=sock_path)
    c.start()
    try:
        assert wait_until(lambda: c.version_error is not None)
        assert "99" in c.version_error
        assert wait_until(lambda: c._thread is not None and not c._thread.is_alive())
    finally:
        c.stop()
        for conn in accepted:
            conn.close()
        srv.close()


def test_handler_exception_does_not_kill_client(server, sock_path):
    seen = []

    def handler(e):
        seen.append(e["type"])
        if e["type"] == "caption":
            raise RuntimeError("boom")

    c = BusClient(handler, path=sock_path)
    c.start()
    try:
        assert wait_until(lambda: c.connected)
        server.publish("caption", {})
        server.publish("state", {"state": "idle"})
        assert wait_until(lambda: "state" in seen)
        assert c.connected
    finally:
        c.stop()
