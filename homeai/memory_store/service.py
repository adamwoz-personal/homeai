# Copyright (c) 2026 Adam Wosotowsky
# SPDX-License-Identifier: BUSL-1.1
# Licensed under the Business Source License 1.1; see LICENSE at the
# repository root. Converts to Apache-2.0 on 2030-09-15.
"""Long-term memory for the voice daemon: record turns, recall for requests.

The voice loop must never wait on, or fail because of, the database:

* ``on_turn`` only queues; a background thread embeds and writes.
* ``recall`` is bounded by ``recall_timeout_ms`` and returns "" on any error,
  before the service is ready, or for anything ``policy`` says to skip.
* Errors are logged once per kind per ten minutes, not on every utterance.
"""

from __future__ import annotations

import logging
import queue
import threading
import time
from datetime import datetime, timezone
from typing import Callable

from ..config import MemoryConfig
from . import commands, policy, recall
from .spoken import MSG_BROKEN
from .store import Episode

log = logging.getLogger(__name__)

MSG_NOT_READY = "Sorry, my long-term memory isn't available right now."

QUEUE_MAX = 200
WARN_EVERY_S = 600.0
WORKER_POLL_S = 30.0
QUIET_S = 60.0          # background jobs wait for this long a pause in talk
TIMINGS_EVERY_S = 3600.0  # recall timings go to health_samples about hourly
TIMINGS_MAX = 2000


class MemoryService:
    def __init__(self, cfg: MemoryConfig, *, store_factory: Callable | None = None,
                 helper_factory: Callable | None = None,
                 clock: Callable[[], datetime] | None = None,
                 monotonic: Callable[[], float] = time.monotonic) -> None:
        self.cfg = cfg
        self._store_factory = store_factory or self._default_store
        self._helper_factory = helper_factory or self._default_helper
        self._now = clock or (lambda: datetime.now().astimezone())
        self._mono = monotonic
        self.store = None
        self.worker = None
        self.spoken = None
        self._ready = threading.Event()
        self._stop = threading.Event()
        self._wake = threading.Event()
        self._q: queue.Queue = queue.Queue(maxsize=QUEUE_MAX)
        self._warned: dict[str, float] = {}
        self._thread: threading.Thread | None = None
        self._worker_thread: threading.Thread | None = None
        self._last_turn = -QUIET_S
        self.last_recall_ms = 0.0
        # Recall timings for the health sampler; touched only by the voice thread.
        self._timings: list[float] = []
        self._empty = 0
        self._timings_since = monotonic()

    # -- lifecycle ---------------------------------------------------------------

    def start(self) -> "MemoryService":
        """Connect and load the embedder in the background; returns at once."""
        self._thread = threading.Thread(target=self._run, name="ltm-recorder", daemon=True)
        self._thread.start()
        return self

    def wait_ready(self, timeout: float) -> bool:
        return self._ready.wait(timeout)

    @property
    def ready(self) -> bool:
        return self._ready.is_set()

    def close(self, timeout: float = 5.0) -> None:
        if self._ready.is_set():
            self.flush(timeout)
        self._stop.set()
        self._wake.set()
        try:
            self._q.put_nowait(None)
        except queue.Full:
            pass
        if self._thread:
            self._thread.join(timeout)
        if self._worker_thread:
            # A helper call can take minutes; the thread is a daemon and its
            # job is recovered on the next start.
            self._worker_thread.join(min(timeout, 1.0))
        if self.store is not None:
            try:
                self.store.db.close()
            except Exception:  # noqa: BLE001
                pass

    def _default_store(self):
        from .db import Db
        from .embed import DEFAULT_MODEL, OnnxEmbedder
        from .store import PostgresStore

        db = Db(self.cfg.dsn, role="homeai_app").open(wait=True)
        try:
            with db.scoped(None, "admin") as conn:
                conn.execute("SELECT 1 FROM memory.schema_version LIMIT 1")
        except Exception:
            db.close()
            raise
        return PostgresStore(db, OnnxEmbedder(self.cfg.embed_model or DEFAULT_MODEL, threads=2))

    def _default_helper(self):
        from .helper import OllamaHelper

        return OllamaHelper(self.cfg.helper_url, self.cfg.helper_model,
                            threads=self.cfg.helper_threads)

    # -- recording -----------------------------------------------------------------

    def on_turn(self, turn) -> None:
        """TranscriptLog listener. Cheap checks here, the work in the thread."""
        self._last_turn = self._mono()
        d = policy.should_record(heard=turn.heard, reply=turn.reply, verdict=turn.verdict,
                                 speaker=turn.speaker, ok=turn.ok,
                                 ignore_speakers=self.cfg.ignore_speakers)
        if not d.record:
            return
        ep = Episode(username=policy.username_for(turn.speaker),
                     user=policy.clean_heard(turn.heard), assistant=turn.reply,
                     occurred_at=datetime.now(timezone.utc),
                     confidence=float(turn.speaker_score or 1.0),
                     extra={"verdict": turn.verdict, "source": turn.source})
        try:
            self._q.put_nowait(("episode", ep, turn.speaker))
        except queue.Full:
            self._warn("full", "memory queue full; dropping a turn")

    def _run(self) -> None:
        while not self._stop.is_set():
            try:
                self.store = self._store_factory()
                break
            except Exception as exc:  # noqa: BLE001
                self._warn("connect", "long-term memory unavailable (%s); retrying in 60 s", exc)
                if self._stop.wait(60):
                    return
        from .spoken import SpokenCommands

        self.spoken = SpokenCommands(self.store, min_similarity=self.cfg.min_similarity,
                                     flush=self.flush, clock=self._now, monotonic=self._mono)
        if self.cfg.extract:
            try:
                from .worker import Worker

                self.worker = Worker(self.store, self._helper_factory(),
                                     batch=self.cfg.extract_batch,
                                     reflect_delay_s=self.cfg.reflect_delay_s, clock=self._now)
                self._worker_thread = threading.Thread(target=self._work, name="ltm-worker",
                                                       daemon=True)
                self._worker_thread.start()
            except Exception as exc:  # noqa: BLE001 - recall and recording still work
                self._warn("worker", "memory extraction disabled: %s", exc)
                self.worker = None
        self._ready.set()
        log.info("long-term memory ready%s", "" if self.worker else " (no extraction)")
        while True:
            item = self._q.get()
            try:
                if item is None or self._stop.is_set() and self._q.empty():
                    return
                self._handle(item)
            except Exception as exc:  # noqa: BLE001
                self._warn("write", "memory write failed: %s", exc)
            finally:
                self._q.task_done()

    def flush(self, timeout: float = 5.0) -> bool:
        """Wait until queued writes are done (tests, shutdown). True if drained."""
        deadline = time.monotonic() + timeout
        while self._q.unfinished_tasks and time.monotonic() < deadline:
            time.sleep(0.01)
        return not self._q.unfinished_tasks

    def _handle(self, item) -> None:
        kind = item[0]
        if kind == "episode":
            _, ep, display = item
            self.store.ensure_person(ep.username, display, voice_handle=display,
                                     is_admin=ep.username in self.cfg.admins)
            if self.store.add_episode(ep) is not None and self.worker is not None:
                self._queue_extract(ep.username)
        elif kind == "touch":
            _, username, ids = item
            with self.store.db.scoped(username, "person") as conn:
                conn.execute("UPDATE memory.facts SET last_accessed = now(),"
                             " access_count = access_count + 1 WHERE id = ANY(%s)", (ids,))
        elif kind == "timings":
            from . import maintain

            _, ms, empty = item
            with self.store.db.scoped(None, "admin") as conn:
                maintain.record(conn, maintain.recall_sample(ms, empty))

    # -- background jobs -------------------------------------------------------------

    def _queue_extract(self, username: str) -> None:
        """Extract once the person has gone quiet, or now if a batch is full."""
        from . import jobs
        from .worker import extract_key

        key = extract_key(username)
        with self.store.db.scoped(None, "admin") as conn:
            jobs.enqueue(conn, "extract", {"username": username}, key=key,
                         delay_s=self.cfg.extract_idle_s)
            if self.store.count_unextracted(username) >= self.cfg.extract_batch:
                jobs.run_now(conn, key)

    def quiet(self) -> bool:
        """Nobody has spoken to Jarvis for QUIET_S: the CPU is free for jobs."""
        return self._mono() - self._last_turn >= QUIET_S

    def _work(self) -> None:
        try:
            n = self.worker.recover()
            if n:
                log.info("memory: requeued %d interrupted jobs", n)
        except Exception as exc:  # noqa: BLE001
            self._warn("worker", "memory job recovery failed: %s", exc)
        while not self._stop.is_set():
            self.work_once()
            self._wake.wait(WORKER_POLL_S)
            self._wake.clear()

    def work_once(self) -> int:
        """Run due jobs while it stays quiet. Returns how many ran."""
        n = 0
        while not self._stop.is_set() and self.quiet():
            try:
                if not self.worker.step():
                    break
            except Exception as exc:  # noqa: BLE001 - e.g. the database went away
                self._warn("job", "memory job loop failed: %s", exc)
                break
            n += 1
        if n:
            try:
                self.worker.helper.release(self.cfg.restore_model or None)
            except Exception as exc:  # noqa: BLE001 - a missed warm-up only costs a slow answer
                self._warn("release", "memory helper release failed: %s", exc)
        return n

    # -- spoken commands -------------------------------------------------------------

    def handle_command(self, speaker: str | None, utterance: str) -> tuple[str, bool] | None:
        """(reply, listen) for a memory command, else None. Never raises."""
        text = policy.clean_heard(utterance)
        if not self._ready.is_set() or self.spoken is None:
            return (MSG_NOT_READY, False) if commands.parse(text) else None
        username = policy.username_for(speaker)
        if username in self.cfg.ignore_speakers:
            username = None
        try:
            return self.spoken.handle(username, speaker, text)
        except Exception as exc:  # noqa: BLE001
            self._warn("command", "memory command failed: %s", exc)
            return (MSG_BROKEN, False) if commands.parse(text) else None

    # -- recall ------------------------------------------------------------------------

    def recall(self, speaker: str | None, utterance: str) -> str:
        """A prompt block of relevant memories, or "". Never raises."""
        start = time.perf_counter()
        queried, block = False, ""
        try:
            if not self._ready.is_set() or not policy.should_recall(utterance, speaker):
                return ""
            username = policy.username_for(speaker)
            if username in self.cfg.ignore_speakers:
                return ""
            now = self._now()
            vec = self.store.embedder.embed([utterance], kind="query")[0]
            left_ms = self.cfg.recall_timeout_ms - (time.perf_counter() - start) * 1000
            if left_ms < 5:
                self._warn("slow-embed", "recall skipped: embedding took the whole budget")
                return ""
            with self.store.db.scoped(username, "person", timeout_ms=int(left_ms),
                                      connect_timeout_s=left_ms / 1000) as conn:
                queried = True
                rows = recall.fetch(conn, username, vec, utterance,
                                    recall.time_window(utterance, now))
            cands = recall.keep(recall.fuse(rows, now), self.cfg.min_similarity)
            block, used = recall.render(cands, speaker, now, self.cfg.budget_tokens * 4)
            facts = [i for k, i in used if k == "fact"]
            if facts:
                try:
                    self._q.put_nowait(("touch", username, facts))
                except queue.Full:
                    pass
            return block
        except Exception as exc:  # noqa: BLE001 - memory must never cost a reply
            self._warn("recall", "recall failed: %s", exc)
            return ""
        finally:
            self.last_recall_ms = (time.perf_counter() - start) * 1000
            if queried:
                self._note_timing(self.last_recall_ms, not block)

    def _note_timing(self, ms: float, empty: bool) -> None:
        """Buffer one recall's time; about hourly hand the batch to the writer."""
        if len(self._timings) < TIMINGS_MAX:
            self._timings.append(ms)
            self._empty += int(empty)
        if self._mono() - self._timings_since < TIMINGS_EVERY_S:
            return
        batch, empty_n = self._timings, self._empty
        self._timings, self._empty, self._timings_since = [], 0, self._mono()
        try:
            self._q.put_nowait(("timings", batch, empty_n))
        except queue.Full:
            pass

    # -- internals -----------------------------------------------------------------------

    def _warn(self, key: str, msg: str, *args) -> None:
        now = time.monotonic()
        if now - self._warned.get(key, -WARN_EVERY_S) >= WARN_EVERY_S:
            self._warned[key] = now
            log.warning(msg, *args)
