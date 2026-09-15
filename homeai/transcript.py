# Copyright (c) 2026 Adam Wosotowsky
# SPDX-License-Identifier: BUSL-1.1
# Licensed under the Business Source License 1.1; see LICENSE at the
# repository root. Converts to Apache-2.0 on 2030-09-15.
"""Append-only transcript log of every voice interaction.

Plan item 5A#5: "Log every spoken command with a timestamp to a file you can
audit." Until now the daemon logged only what it *heard*, never what it
*answered*, which makes an audit trail useless - you could not tell whether a
misheard command produced a harmless reply or a real action.

Format is JSON Lines: one self-contained JSON object per line, appended and
flushed immediately. Chosen because it survives a crash mid-write (you lose at
most the final line), needs no schema migration, and is trivially greppable
and queryable with ``jq``.

Failure policy: logging must never break the assistant. Every failure is
swallowed and reported once, because a full disk should degrade the audit
trail, not silence the house.
"""

from __future__ import annotations

import json
import logging
import os
import threading
import time
from dataclasses import dataclass
from datetime import datetime, timezone

log = logging.getLogger(__name__)


@dataclass
class Turn:
    """One complete interaction, from wake word to spoken reply."""

    heard: str = ""
    reply: str = ""
    verdict: str = ""
    ok: bool = True
    error: str = ""
    attempts: int = 0
    stt_ms: float = 0.0
    agent_ms: float = 0.0
    tts_ms: float = 0.0
    tts_first_audio_ms: float = 0.0
    perceived_ms: float = 0.0
    total_ms: float = 0.0


class TranscriptLog:
    """Thread-safe JSONL writer for conversation turns."""

    def __init__(self, path: str, enabled: bool = True) -> None:
        self._path = os.path.expanduser(path)
        self._enabled = enabled
        self._lock = threading.Lock()
        self._warned = False

        if self._enabled:
            try:
                parent = os.path.dirname(self._path)
                if parent:
                    os.makedirs(parent, exist_ok=True)
                # The transcript contains everything said in the house.
                # Create it private, and tighten it if it already exists.
                fd = os.open(self._path, os.O_CREAT | os.O_APPEND | os.O_WRONLY, 0o600)
                os.close(fd)
                os.chmod(self._path, 0o600)
            except OSError as exc:
                log.warning("transcript log unavailable (%s); continuing without it", exc)
                self._enabled = False

    @property
    def path(self) -> str:
        return self._path

    @property
    def enabled(self) -> bool:
        return self._enabled

    def write(self, turn: Turn) -> None:
        if not self._enabled:
            return

        record = {
            "ts": datetime.now(timezone.utc).isoformat(timespec="milliseconds"),
            "heard": turn.heard,
            "reply": turn.reply,
            "verdict": turn.verdict,
            "ok": turn.ok,
            "attempts": turn.attempts,
            "timings_ms": {
                "stt": round(turn.stt_ms),
                "agent": round(turn.agent_ms),
                "tts": round(turn.tts_ms),
                "tts_first_audio": round(turn.tts_first_audio_ms),
                "perceived": round(turn.perceived_ms),
                "total": round(turn.total_ms),
            },
        }
        if turn.error:
            record["error"] = turn.error

        try:
            line = json.dumps(record, ensure_ascii=False)
        except (TypeError, ValueError) as exc:
            log.warning("could not serialise transcript record: %s", exc)
            return

        try:
            with self._lock:
                with open(self._path, "a", encoding="utf-8") as fh:
                    fh.write(line + "\n")
                    fh.flush()
        except OSError as exc:
            # Warn once, then stay quiet; a failing disk must not spam the log
            # on every single utterance.
            if not self._warned:
                log.warning("transcript write failed (%s); further errors suppressed", exc)
                self._warned = True


class Stopwatch:
    """Monotonic elapsed-milliseconds helper.

    Uses ``time.monotonic`` so a clock adjustment cannot produce a negative or
    wildly wrong duration in the audit record.
    """

    def __init__(self) -> None:
        self._start = time.monotonic()

    def reset(self) -> None:
        self._start = time.monotonic()

    def ms(self) -> float:
        return (time.monotonic() - self._start) * 1000.0
