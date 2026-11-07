"""The art job queue: one picture at a time on the GPU (AVATAR_PLAN §7.4, 6a step 5).

- Order: previews (priority 0) before normal jobs (1), then first come first served.
- Persisted to ``queue.json`` after every change, atomically. A job found
  ``running`` at start-up was cut off by a restart: it goes back to the front of
  the line and is submitted again (ComfyUI forgets its prompts when it restarts).
- At most ``max_queued`` jobs wait; more fail at once with a message.
- Cancel: a waiting job leaves the line; a running one is interrupted in ComfyUI.
- After a timeout ComfyUI is restarted before the next job, because a dead job
  thread doesn't show in its health check (homeai/art/comfy.py, 2026-11-07).
- Each change is reported through ``notify`` as an event dict; a failing
  listener never stops the queue.

``run_next()`` does one job synchronously, so tests drive it without threads;
``start()`` runs it on a worker thread. All shared state is under one lock, and
nothing slow (ComfyUI, disk copies of pictures) happens while holding it.
"""
from __future__ import annotations

import itertools
import json
import logging
import threading
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path

from homeai.art.comfy import MSG_TIMED_OUT, ComfyError
from homeai.art.store import StoreError, write_json

log = logging.getLogger(__name__)

PATH = Path.home() / ".local/share/homeai/art/queue.json"
FINAL = ("done", "failed", "cancelled")
PREVIEW, NORMAL = 0, 1
MAX_QUEUED = 20
DEFAULT_TIMEOUT_S = 600.0
MSG_FULL = "Too many pictures in line. Try again in a little while."
MSG_CANCELLED = "Stopped."
MSG_INTERRUPTED = "Jarvis restarted, so this picture is being drawn again."


@dataclass
class Job:
    id: str
    project: str
    node: str
    graph: dict
    priority: int = NORMAL
    timeout: float = DEFAULT_TIMEOUT_S
    state: str = "queued"
    seq: int = 0
    created: float = 0.0
    message: str = ""
    image: str = ""
    seconds: float = 0.0
    prompt_id: str = ""
    cancel_asked: bool = field(default=False, repr=False)


class JobQueue:
    def __init__(self, client, store, path: Path = PATH, notify=None, restart=None,
                 max_queued: int = MAX_QUEUED, clock=time.time) -> None:
        self.client, self.store, self.path = client, store, path
        self._notify, self._restart = notify or (lambda event: None), restart
        self.max_queued, self._clock = max_queued, clock
        self._lock = threading.Lock()
        self._wake = threading.Condition(self._lock)
        self._jobs: dict[str, Job] = {}
        self._seq = itertools.count(1)
        self._needs_restart = False
        self._stop = False
        self._thread: threading.Thread | None = None
        self._load()

    # -- persistence -----------------------------------------------------------

    def _load(self) -> None:
        try:
            data = json.loads(self.path.read_text())
            jobs = [Job(**{k: v for k, v in raw.items() if k in Job.__dataclass_fields__}) for raw in data["jobs"]]
        except FileNotFoundError:
            return
        except (ValueError, KeyError, TypeError) as exc:
            log.warning("art queue file unreadable, starting empty: %s", exc)
            self.path.rename(self.path.with_suffix(".json.bad"))
            return
        for job in jobs:
            if job.state == "running":          # cut off by a restart: draw it again, first
                job.state, job.seq, job.prompt_id, job.message = "queued", 0, "", MSG_INTERRUPTED
            if job.state == "queued":
                self._jobs[job.id] = job
        top = max((j.seq for j in self._jobs.values()), default=0)
        self._seq = itertools.count(top + 1)

    def _save(self) -> None:
        live = [asdict(j) for j in self._jobs.values() if j.state not in FINAL]
        for raw in live:
            raw.pop("cancel_asked")
        write_json(self.path, {"version": 1, "jobs": live})

    # -- reading ---------------------------------------------------------------

    def _line(self) -> list[Job]:
        return sorted((j for j in self._jobs.values() if j.state == "queued"), key=lambda j: (j.priority, j.seq))

    def position(self, job_id: str) -> int | None:
        """1 means next; None if it isn't waiting."""
        with self._lock:
            ids = [j.id for j in self._line()]
        return ids.index(job_id) + 1 if job_id in ids else None

    def get(self, job_id: str) -> Job | None:
        with self._lock:
            job = self._jobs.get(job_id)
            return Job(**asdict(job)) if job else None

    def waiting(self) -> list[str]:
        with self._lock:
            return [j.id for j in self._line()]

    # -- events ----------------------------------------------------------------

    def _event(self, job: Job) -> dict:
        event = {"job": job.id, "project": job.project, "node": job.node, "state": job.state}
        if job.state == "queued":
            event["position"] = [j.id for j in self._line()].index(job.id) + 1
        if job.message:
            event["message"] = job.message
        if job.state == "done":
            event["image"] = job.image
        return event

    def _emit(self, events: list[dict]) -> None:
        for event in events:
            try:
                self._notify(event)
            except Exception:  # noqa: BLE001 - a listener must never stop the queue
                log.warning("art queue listener failed", exc_info=True)

    # -- changing --------------------------------------------------------------

    def submit(self, project: str, node: str, graph: dict, priority: int = NORMAL,
               timeout: float = DEFAULT_TIMEOUT_S) -> Job:
        with self._lock:
            seq = next(self._seq)
            job = Job(id=f"j{int(self._clock() * 1000)}-{seq}", project=project, node=node, graph=graph,
                      priority=priority, timeout=timeout, seq=seq, created=self._clock())
            if len(self._line()) >= self.max_queued:
                job.state, job.message = "failed", MSG_FULL
                events = [self._event(job)]
            else:
                self._jobs[job.id] = job
                self._save()
                events = [self._event(job)]
                self._wake.notify_all()
        if job.state == "failed":
            self._fail_node(job)
        self._emit(events)
        return job

    def cancel(self, job_id: str) -> bool:
        """True if there was something to cancel."""
        with self._lock:
            job = self._jobs.get(job_id)
            if job is None or job.state in FINAL:
                return False
            if job.state == "queued":
                job.state, job.message = "cancelled", MSG_CANCELLED
                self._save()
                events = [self._event(job)] + [self._event(j) for j in self._line()]
                running_prompt = ""
            else:
                job.cancel_asked = True
                events, running_prompt = [], job.prompt_id
        if job.state == "cancelled":
            self._fail_node(job)
        if running_prompt:
            self.client.cancel(running_prompt)
        self._emit(events)
        return True

    def _fail_node(self, job: Job) -> None:
        try:
            self.store.open(job.project).fail(job.node, job.message)
        except StoreError as exc:
            log.warning("could not mark %s/%s failed: %s", job.project, job.node, exc)

    def run_next(self) -> Job | None:
        """Run the first job in line to its end; None if the line is empty."""
        with self._lock:
            line = self._line()
            if not line:
                return None
            job = line[0]
            job.state = "running"
            self._save()
            events = [self._event(job)] + [self._event(j) for j in self._line()]
            restart, self._needs_restart = self._needs_restart, False
        self._emit(events)
        if restart and self._restart is not None:
            try:
                self._restart()
            except Exception as exc:  # noqa: BLE001 - reported on the job, the queue carries on
                log.warning("ComfyUI restart failed: %s", exc)
        state, message, image, seconds = self._draw(job)
        with self._lock:
            if job.cancel_asked:
                state, message = "cancelled", MSG_CANCELLED
            if message == MSG_TIMED_OUT:
                self._needs_restart = True
            job.state, job.message, job.image, job.seconds = state, message, image, seconds
            self._save()
            events = [self._event(job)]
        self._record(job)
        self._emit(events)
        return job

    def _draw(self, job: Job) -> tuple[str, str, str, float]:
        try:
            prompt_id = self.client.submit(job.graph)
            with self._lock:
                job.prompt_id = prompt_id
                cancelled_meanwhile = job.cancel_asked
            if cancelled_meanwhile:
                self.client.cancel(prompt_id)
            result = self.client.wait(prompt_id, job.timeout)
            return "done", "", str(result.image), result.seconds
        except ComfyError as exc:
            return "failed", exc.spoken, "", 0.0
        except Exception as exc:  # noqa: BLE001 - one bad job must not kill the worker
            log.exception("art job %s crashed", job.id)
            return "failed", f"The drawing failed. ({type(exc).__name__})", "", 0.0

    def _record(self, job: Job) -> None:
        try:
            project = self.store.open(job.project)
            if job.state == "done":
                project.finish(job.node, Path(job.image), seconds=job.seconds, job=job.id)
            else:
                project.fail(job.node, job.message)
        except (StoreError, OSError) as exc:
            log.warning("could not record %s in %s/%s: %s", job.id, job.project, job.node, exc)

    # -- worker thread ---------------------------------------------------------

    def start(self) -> None:
        self._stop = False
        self._thread = threading.Thread(target=self._loop, name="art-queue", daemon=True)
        self._thread.start()

    def stop(self, timeout: float = 5.0) -> None:
        with self._lock:
            self._stop = True
            self._wake.notify_all()
        if self._thread is not None:
            self._thread.join(timeout)

    def _loop(self) -> None:
        while True:
            with self._lock:
                while not self._stop and not self._line():
                    self._wake.wait()
                if self._stop:
                    return
            self.run_next()
