"""ComfyUI client (homeai/art/comfy.py): unit tests with a fake transport, and a functional test
against a fake ComfyUI HTTP server on a real socket."""
from __future__ import annotations

import io
import json
import threading
import urllib.error
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import pytest

from homeai.art import comfy as C

DONE = {"status": {"status_str": "success", "completed": True},
        "outputs": {"9": {"images": [{"filename": "a_00001_.png", "subfolder": "", "type": "output"}]}}}


class Fake:
    """Answers like ComfyUI; ``history`` is a list of entries returned one per poll."""

    def __init__(self, history=None, raise_on=None):
        self.calls: list[tuple[str, dict | None]] = []
        self.history = list(history or [])
        self.raise_on = raise_on or {}

    def __call__(self, url, payload=None, timeout=30.0):
        path = url.split("8188", 1)[1]
        self.calls.append((path, payload))
        for prefix, exc in self.raise_on.items():
            if path.startswith(prefix):
                raise exc
        if path == "/prompt":
            return {"prompt_id": "p1", "number": 0}
        if path.startswith("/history/"):
            entry = self.history.pop(0) if len(self.history) > 1 else (self.history[0] if self.history else None)
            return {"p1": entry} if entry else {}
        return {}


class Clock:
    def __init__(self):
        self.t = 0.0

    def __call__(self):
        return self.t

    def sleep(self, s):
        self.t += s


def client(fake, out=Path("/out")):
    clock = Clock()
    return C.ComfyClient(http=fake, output_dir=out, clock=clock, sleep=clock.sleep)


def http_error(code, body):
    return urllib.error.HTTPError("u", code, "x", {}, io.BytesIO(body.encode()))


def error_entry(message):
    return {"status": {"status_str": "error", "completed": False,
                       "messages": [["execution_start", {}], ["execution_error", {"exception_message": message}]]}}


def test_run_returns_the_image_path_once_done():
    fake = Fake(history=[None, {"status": {"completed": False}}, DONE])
    result = client(fake).run({"1": {}})
    assert result.image == Path("/out/a_00001_.png") and result.prompt_id == "p1"
    assert fake.calls[0] == ("/prompt", {"prompt": {"1": {}}})
    assert result.seconds == pytest.approx(2 * C.POLL_S)


def test_subfolder_is_kept():
    entry = json.loads(json.dumps(DONE))
    entry["outputs"]["9"]["images"][0]["subfolder"] = "pose"
    assert client(Fake(history=[entry])).poll("p1") == Path("/out/pose/a_00001_.png")


def test_preview_images_are_not_results():
    entry = json.loads(json.dumps(DONE))
    entry["outputs"]["9"]["images"][0]["type"] = "temp"
    with pytest.raises(C.ComfyError) as err:
        client(Fake(history=[entry])).poll("p1")
    assert err.value.spoken == C.MSG_NO_IMAGE


def test_out_of_memory_is_said_plainly():
    with pytest.raises(C.ComfyError) as err:
        client(Fake(history=[error_entry("HIP out of memory. Tried to allocate 2.00 GiB")])).run({})
    assert err.value.spoken == C.MSG_OUT_OF_MEMORY and "2.00 GiB" in err.value.detail


def test_other_failures_are_generic():
    with pytest.raises(C.ComfyError) as err:
        client(Fake(history=[error_entry("shape mismatch")])).run({})
    assert err.value.spoken == C.MSG_FAILED


def test_missing_model_rejected_before_running():
    body = json.dumps({"error": {"type": "prompt_outputs_failed_validation"}, "node_errors": {"1": {"errors": [
        {"type": "value_not_in_list", "details": "unet_name: 'gone.safetensors' not in [...]",
         "extra_info": {"input_name": "unet_name"}, "message": "Value not in list"}]}}})
    with pytest.raises(C.ComfyError) as err:
        client(Fake(raise_on={"/prompt": http_error(400, body)})).run({})
    assert err.value.spoken == C.MSG_MISSING_MODEL


def test_other_rejections_say_the_graph_is_bad():
    body = json.dumps({"error": {"type": "invalid_prompt"}, "node_errors": {}})
    with pytest.raises(C.ComfyError) as err:
        client(Fake(raise_on={"/prompt": http_error(400, body)})).run({})
    assert err.value.spoken == C.MSG_BAD_GRAPH


def test_not_running_says_switch_to_art_mode():
    with pytest.raises(C.ComfyError) as err:
        client(Fake(raise_on={"/prompt": urllib.error.URLError("refused")})).run({})
    assert err.value.spoken == C.MSG_NOT_RUNNING
    assert not client(Fake(raise_on={"/system_stats": ConnectionRefusedError()})).healthy()
    assert client(Fake()).healthy()


def test_timeout_cancels_the_job():
    fake = Fake(history=[{"status": {"completed": False}}])
    with pytest.raises(C.ComfyError) as err:
        client(fake).run({}, timeout=1.0)
    assert err.value.spoken == C.MSG_TIMED_OUT
    assert ("/queue", {"delete": ["p1"]}) in fake.calls and ("/interrupt", {"prompt_id": "p1"}) in fake.calls


def test_cancel_never_raises():
    client(Fake(raise_on={"/queue": urllib.error.URLError("x"), "/interrupt": OSError()})).cancel("p1")


def test_reply_without_an_id_is_an_error():
    with pytest.raises(C.ComfyError, match="no prompt_id"):
        client(lambda url, payload=None, timeout=30.0: {}).submit({})


def test_link_to_a_missing_node_is_refused_before_sending():
    fake = Fake()
    graph = {"1": {"class_type": "VAEDecode", "inputs": {"samples": ["gone", 0], "vae": ["1", 0]}}}
    with pytest.raises(C.ComfyError) as err:
        client(fake).submit(graph)
    assert err.value.spoken == C.MSG_BAD_GRAPH and "1.samples -> gone" in err.value.detail
    assert fake.calls == [], "a graph like this kills ComfyUI's job thread, so it must never be sent"


def test_plain_values_are_not_links():
    graph = {"1": {"inputs": {"text": "a fox", "size": [512, 512], "seed": 7, "model": ["1", 0]}}}
    assert C.dangling_links(graph) == []


def test_classify():
    assert C.classify("CUDA out of memory") == C.MSG_OUT_OF_MEMORY
    assert C.classify("ckpt file not found") == C.MSG_MISSING_MODEL
    assert C.classify("file not found") == C.MSG_FAILED


# -- functionality: a fake ComfyUI on a real socket -------------------------------------------------


class FakeComfy(BaseHTTPRequestHandler):
    polls = 0

    def log_message(self, *args):
        pass

    def _send(self, code, body):
        data = json.dumps(body).encode()
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def do_POST(self):
        payload = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
        if self.path == "/prompt":
            if "bad" in payload["prompt"]:
                return self._send(400, {"error": {"type": "invalid_prompt"}, "node_errors": {}})
            return self._send(200, {"prompt_id": "p1", "number": 0})
        self._send(200, {})

    def do_GET(self):
        if self.path == "/history/p1":
            type(self).polls += 1
            return self._send(200, {"p1": DONE} if self.polls >= 3 else {})
        self._send(200, {"system": {}})


@pytest.fixture
def server():
    srv = ThreadingHTTPServer(("127.0.0.1", 0), FakeComfy)
    FakeComfy.polls = 0
    thread = threading.Thread(target=srv.serve_forever, daemon=True)
    thread.start()
    yield f"http://127.0.0.1:{srv.server_address[1]}"
    srv.shutdown()


def test_over_a_real_socket(server, tmp_path):
    real = C.ComfyClient(url=server, output_dir=tmp_path)
    assert real.healthy()
    result = real.run({"1": {"class_type": "SaveImage"}}, timeout=10)
    assert result.image == tmp_path / "a_00001_.png" and FakeComfy.polls == 3
    with pytest.raises(C.ComfyError) as err:
        real.run({"bad": {}})
    assert err.value.spoken == C.MSG_BAD_GRAPH


def test_closed_port_is_not_running(tmp_path):
    srv = ThreadingHTTPServer(("127.0.0.1", 0), FakeComfy)
    port = srv.server_address[1]
    srv.server_close()
    real = C.ComfyClient(url=f"http://127.0.0.1:{port}", output_dir=tmp_path)
    assert not real.healthy()
    with pytest.raises(C.ComfyError) as err:
        real.submit({})
    assert err.value.spoken == C.MSG_NOT_RUNNING
