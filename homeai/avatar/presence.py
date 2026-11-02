"""The daemon's view of the avatar: publish state and captions, never fail the voice loop.

Wraps an EventBus. When the avatar is turned off (HOMEAI_AVATAR=0) no socket is
created and every call is a no-op. A bus that fails to start disables itself
with a warning. Consecutive identical states are not re-sent.
"""
from __future__ import annotations

import logging
import threading

from .bus import EventBus

log = logging.getLogger("homeai.avatar.presence")

STATES = frozenset({"idle", "listening", "thinking", "speaking", "followup", "sleep",
                    "muted", "offline", "generating", "mode_switching"})


class Presence:
    def __init__(self, enabled: bool, socket_path: str = "", bus: EventBus | None = None) -> None:
        self._bus = (bus or EventBus(socket_path or None, capabilities=("state", "utterance",
                                                                        "caption", "speech_stop"))
                     ) if enabled else None
        self._lock = threading.Lock()
        self.current: str | None = None

    @property
    def enabled(self) -> bool:
        return self._bus is not None

    def start(self) -> None:
        if self._bus is None:
            return
        try:
            self._bus.start()
        except (RuntimeError, OSError) as exc:
            log.warning("avatar event bus off: %s", exc)
            self._bus = None
            return
        self.state("idle")

    def stop(self) -> None:
        if self._bus is not None:
            self.state("offline")
            self._bus.stop()

    def publish(self, etype: str, data: dict) -> None:
        bus = self._bus
        if bus is None:
            return
        try:
            bus.publish(etype, data)
        except Exception:  # noqa: BLE001 - the avatar must never break the voice loop
            log.warning("avatar publish failed", exc_info=True)

    def state(self, name: str, **extra) -> None:
        if name not in STATES:
            log.warning("unknown avatar state %r ignored", name)
            return
        with self._lock:
            if name == self.current and not extra:
                return
            self.current = name
        self.publish("state", {"state": name, **extra})

    def caption(self, who: str, text: str) -> None:
        if text and text.strip():
            self.publish("caption", {"who": who, "text": text.strip()})
