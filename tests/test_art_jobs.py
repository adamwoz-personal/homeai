"""Art job queue (homeai/art/jobs.py): order, persistence, cancel, timeouts and restart, worker thread."""
from __future__ import annotations

import json
import threading
import time
from pathlib import Path

import pytest

from homeai.art import comfy as C
from homeai.art import jobs as J
from homeai.art import store as S


class FakeClient:
    """ComfyClient stand-in. ``outcomes`` maps a graph's "name" to an exception, or "ok"."""

    def __init__(self, picture: Path, outcomes=None):
        self.picture, self.outcomes = picture, outcomes or {}
        self.submitted: list[str] = []
        self.cancelled: list[str] = []
        self.gate: threading.Event | None = None

    def submit(self, graph):
        self.submitted.append(graph["name"])
        return f"p-{graph['name']}"

    def wait(self, prompt_id, timeout):
        if self.gate is not None:
            self.gate.wait(5)
        outcome = self.outcomes.get(prompt_id[2:], "ok")
        if outcome != "ok":
            raise outcome
        return C.Result(prompt_id, self.picture, 1.5)

    def cancel(self, prompt_id):
        self.cancelled.append(prompt_id)


@pytest.fixture
def picture(tmp_path):
    path = tmp_path / "out.png"
    path.write_bytes(b"png")
    return path


@pytest.fixture
def store(tmp_path):
    return S.Store(tmp_path / "projects")


@pytest.fixture
def project(store):
    p = store.create("general", "adam")
    for name in "abcde":
        p.add(name, {})
    return p


def make(tmp_path, picture, store, outcomes=None, **kw):
    events = []
    client = FakeClient(picture, outcomes)
    q = J.JobQueue(client, store, path=tmp_path / "queue.json", notify=events.append, **kw)
    return q, client, events


def graph(name):
    return {"name": name}


def test_previews_jump_the_line_then_first_come_first_served(tmp_path, picture, store, project):
    q, client, _ = make(tmp_path, picture, store)
    q.submit(project.id, "001", graph("a"))
    q.submit(project.id, "002", graph("b"))
    q.submit(project.id, "003", graph("c"), priority=J.PREVIEW)
    while q.run_next():
        pass
    assert client.submitted == ["c", "a", "b"]


def test_done_job_lands_in_the_store(tmp_path, picture, store, project):
    q, _, events = make(tmp_path, picture, store)
    job = q.submit(project.id, "001", graph("a"))
    q.run_next()
    node = store.open(project.id).nodes["001"]
    assert node.state == "done" and node.meta["job"] == job.id and node.meta["seconds"] == 1.5
    assert [e["state"] for e in events] == ["queued", "running", "done"]
    assert events[-1]["image"] == str(picture) and events[0]["position"] == 1


def test_failure_is_spoken_and_recorded(tmp_path, picture, store, project):
    q, _, events = make(tmp_path, picture, store, {"a": C.ComfyError(C.MSG_OUT_OF_MEMORY)})
    q.submit(project.id, "001", graph("a"))
    job = q.run_next()
    assert (job.state, job.message) == ("failed", C.MSG_OUT_OF_MEMORY)
    assert store.open(project.id).nodes["001"].meta["error"] == C.MSG_OUT_OF_MEMORY
    assert events[-1]["message"] == C.MSG_OUT_OF_MEMORY


def test_unexpected_crash_fails_only_that_job(tmp_path, picture, store, project):
    q, client, _ = make(tmp_path, picture, store, {"a": RuntimeError("boom")})
    q.submit(project.id, "001", graph("a"))
    q.submit(project.id, "002", graph("b"))
    assert q.run_next().state == "failed"
    assert q.run_next().state == "done"


def test_timeout_restarts_comfy_before_the_next_job(tmp_path, picture, store, project):
    order = []
    q, client, _ = make(tmp_path, picture, store, {"a": C.ComfyError(C.MSG_TIMED_OUT)},
                        restart=lambda: order.append("restart"))
    q.submit(project.id, "001", graph("a"))
    q.submit(project.id, "002", graph("b"))
    q.run_next()
    assert order == []
    q.run_next()
    assert order == ["restart"] and client.submitted == ["a", "b"]
    q.submit(project.id, "003", graph("c"))
    q.run_next()
    assert order == ["restart"], "only once per timeout"


def test_failed_restart_does_not_stop_the_queue(tmp_path, picture, store, project):
    def broken():
        raise RuntimeError("no")
    q, _, _ = make(tmp_path, picture, store, {"a": C.ComfyError(C.MSG_TIMED_OUT)}, restart=broken)
    q.submit(project.id, "001", graph("a"))
    q.submit(project.id, "002", graph("b"))
    q.run_next()
    assert q.run_next().state == "done"


def test_full_line_fails_at_once(tmp_path, picture, store, project):
    q, _, events = make(tmp_path, picture, store, max_queued=2)
    q.submit(project.id, "001", graph("a"))
    q.submit(project.id, "002", graph("b"))
    job = q.submit(project.id, "003", graph("c"))
    assert (job.state, job.message) == ("failed", J.MSG_FULL)
    assert len(q.waiting()) == 2
    assert store.open(project.id).nodes["003"].state == "failed"
    assert events[-1]["state"] == "failed"


def test_cancel_waiting_job_and_positions_move_up(tmp_path, picture, store, project):
    q, client, events = make(tmp_path, picture, store)
    a = q.submit(project.id, "001", graph("a"))
    b = q.submit(project.id, "002", graph("b"))
    assert q.position(b.id) == 2
    assert q.cancel(a.id)
    assert q.position(b.id) == 1 and q.get(a.id).state == "cancelled"
    assert {"job": b.id, "project": project.id, "node": "002", "state": "queued", "position": 1} in events
    assert store.open(project.id).nodes["001"].state == "failed"
    assert not q.cancel(a.id), "already final"
    assert not q.cancel("nope")
    q.run_next()
    assert client.submitted == ["b"]


def test_cancel_running_job_interrupts_comfy(tmp_path, picture, store, project):
    q, client, _ = make(tmp_path, picture, store)
    client.gate = threading.Event()
    job = q.submit(project.id, "001", graph("a"))
    worker = threading.Thread(target=q.run_next)
    worker.start()
    for _ in range(100):
        if q.get(job.id).prompt_id:
            break
        time.sleep(0.01)
    assert q.cancel(job.id)
    client.gate.set()
    worker.join(5)
    assert client.cancelled == ["p-a"]
    assert q.get(job.id).state == "cancelled"


def test_state_survives_a_restart_and_running_goes_first(tmp_path, picture, store, project):
    q, _, _ = make(tmp_path, picture, store)
    a = q.submit(project.id, "001", graph("a"))
    q.submit(project.id, "002", graph("b"))
    data = json.loads((tmp_path / "queue.json").read_text())
    data["jobs"][0]["state"] = "running"          # as if Jarvis died mid-picture
    (tmp_path / "queue.json").write_text(json.dumps(data))
    q2, client, _ = make(tmp_path, picture, store)
    assert q2.waiting()[0] == a.id and q2.get(a.id).message == J.MSG_INTERRUPTED
    later = q2.submit(project.id, "003", graph("c"))
    assert q2.waiting()[-1] == later.id, "numbering continues after a restart"
    while q2.run_next():
        pass
    assert client.submitted == ["a", "b", "c"]


def test_finished_jobs_are_not_kept_on_disk(tmp_path, picture, store, project):
    q, _, _ = make(tmp_path, picture, store)
    q.submit(project.id, "001", graph("a"))
    q.run_next()
    assert json.loads((tmp_path / "queue.json").read_text())["jobs"] == []


def test_unreadable_queue_file_is_set_aside(tmp_path, picture, store):
    (tmp_path / "queue.json").write_text("{")
    q, _, _ = make(tmp_path, picture, store)
    assert q.waiting() == [] and (tmp_path / "queue.json.bad").exists()


def test_a_bad_listener_does_not_stop_the_queue(tmp_path, picture, store, project):
    def bad(event):
        raise ValueError("listener bug")
    q = J.JobQueue(FakeClient(picture), store, path=tmp_path / "queue.json", notify=bad)
    q.submit(project.id, "001", graph("a"))
    assert q.run_next().state == "done"


def test_missing_project_is_logged_not_fatal(tmp_path, picture, store):
    q, _, _ = make(tmp_path, picture, store)
    q.submit("gone", "001", graph("a"))
    assert q.run_next().state == "done"


def test_worker_thread_runs_jobs_and_stops(tmp_path, picture, store, project):
    q, client, events = make(tmp_path, picture, store)
    q.start()
    try:
        q.submit(project.id, "001", graph("a"))
        q.submit(project.id, "002", graph("b"))
        for _ in range(200):
            if sum(e["state"] == "done" for e in events) == 2:
                break
            time.sleep(0.01)
    finally:
        q.stop()
    assert client.submitted == ["a", "b"]
    assert not q._thread.is_alive()


def test_empty_line_runs_nothing(tmp_path, picture, store):
    q, _, _ = make(tmp_path, picture, store)
    assert q.run_next() is None
