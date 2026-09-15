# Copyright (c) 2026 Adam Wosotowsky
# SPDX-License-Identifier: BUSL-1.1
# Licensed under the Business Source License 1.1; see LICENSE at the
# repository root. Converts to Apache-2.0 on 2030-09-15.
"""Microphone capture with a ring buffer.

The capture thread must never block and never die. It writes into a fixed-size
ring buffer; consumers copy out what they need. If the device disappears (the
webcam is unplugged) the thread reports unhealthy and retries rather than
raising into the main loop.
"""

from __future__ import annotations

import logging
import threading
from dataclasses import dataclass

import numpy as np

from .config import AudioConfig

log = logging.getLogger(__name__)


@dataclass(frozen=True)
class DeviceInfo:
    index: int
    name: str
    channels: int
    sample_rate: int


def find_input_device(match: str) -> DeviceInfo | None:
    """Locate an input device by name substring.

    Matching by name rather than index is deliberate: USB device indices shift
    when peripherals are re-enumerated across reboots.
    """
    try:
        import sounddevice as sd
    except (ImportError, OSError) as exc:
        log.error("sounddevice unavailable: %s", exc)
        return None

    try:
        devices = sd.query_devices()
    except Exception as exc:  # noqa: BLE001 - PortAudio raises broadly
        log.error("could not enumerate audio devices: %s", exc)
        return None

    needle = match.lower()
    for index, dev in enumerate(devices):
        if dev.get("max_input_channels", 0) < 1:
            continue
        if needle in str(dev.get("name", "")).lower():
            return DeviceInfo(
                index=index,
                name=str(dev.get("name")),
                channels=int(dev["max_input_channels"]),
                sample_rate=int(dev.get("default_samplerate", 16000)),
            )
    return None


class RingBuffer:
    """Fixed-size circular buffer of mono float32 samples.

    Thread-safe. Overwrites oldest data when full, which is the correct
    behaviour for a live microphone: stale audio is worthless.

    Consumers track an absolute cursor (a count of samples consumed since the
    buffer was created) and call ``read_new``. This guarantees every sample is
    delivered exactly once. Reading with ``read_last`` instead causes the same
    audio to be processed repeatedly, which makes a wake-word detector fire
    over and over on one utterance.
    """

    def __init__(self, capacity: int) -> None:
        if capacity <= 0:
            raise ValueError("capacity must be positive")
        self._capacity = capacity
        self._buffer = np.zeros(capacity, dtype=np.float32)
        self._write = 0
        self._filled = 0
        self._total = 0  # samples ever written; the absolute timeline
        self._lock = threading.Lock()

    @property
    def capacity(self) -> int:
        return self._capacity

    @property
    def total_written(self) -> int:
        with self._lock:
            return self._total

    def __len__(self) -> int:
        with self._lock:
            return self._filled

    def write(self, samples: np.ndarray) -> None:
        flat = np.asarray(samples, dtype=np.float32).reshape(-1)
        if flat.size == 0:
            return

        with self._lock:
            # A write larger than the buffer: keep only the newest tail.
            if flat.size >= self._capacity:
                self._buffer[:] = flat[-self._capacity:]
                self._write = 0
                self._filled = self._capacity
                self._total += flat.size
                return

            end = self._write + flat.size
            if end <= self._capacity:
                self._buffer[self._write:end] = flat
            else:
                split = self._capacity - self._write
                self._buffer[self._write:] = flat[:split]
                self._buffer[: end - self._capacity] = flat[split:]

            self._write = end % self._capacity
            self._filled = min(self._filled + flat.size, self._capacity)
            self._total += flat.size

    def _read_locked(self, n: int) -> np.ndarray:
        """Read the most recent ``n`` samples. Caller must hold the lock."""
        n = min(n, self._filled)
        if n <= 0:
            return np.zeros(0, dtype=np.float32)
        start = (self._write - n) % self._capacity
        if start + n <= self._capacity:
            return self._buffer[start:start + n].copy()
        split = self._capacity - start
        return np.concatenate((self._buffer[start:].copy(), self._buffer[: n - split].copy()))

    def read_last(self, n: int) -> np.ndarray:
        """Return the most recent ``n`` samples in chronological order."""
        with self._lock:
            return self._read_locked(n)

    def read_new(self, cursor: int, max_samples: int | None = None) -> tuple[np.ndarray, int]:
        """Return samples written since ``cursor``, plus the updated cursor.

        If the consumer has fallen so far behind that data was overwritten, the
        cursor jumps forward to the oldest surviving sample rather than
        returning stale or misaligned audio.
        """
        with self._lock:
            oldest = self._total - self._filled
            if cursor < oldest:
                cursor = oldest

            available = self._total - cursor
            if available <= 0:
                return np.zeros(0, dtype=np.float32), cursor

            n = available if max_samples is None else min(available, max_samples)
            # Samples from `cursor` for `n` are the most recent
            # (total - cursor) window, trimmed to n from its start.
            window = self._read_locked(available)
            return window[:n].copy(), cursor + n

    def clear(self) -> int:
        """Empty the buffer. Returns the cursor a consumer should adopt."""
        with self._lock:
            self._buffer[:] = 0
            self._write = 0
            self._filled = 0
            return self._total


def rms(samples: np.ndarray) -> float:
    """Root-mean-square level, used for cheap voice-activity detection."""
    if samples is None or samples.size == 0:
        return 0.0
    return float(np.sqrt(np.mean(np.square(samples, dtype=np.float64))))


class Microphone:
    """Background capture into a ring buffer.

    ``paused`` is set while TTS plays so the assistant does not hear itself.
    """

    def __init__(self, cfg: AudioConfig) -> None:
        self._cfg = cfg
        self.buffer = RingBuffer(cfg.ring_frames)
        self._stream = None
        self._paused = threading.Event()
        self._healthy = threading.Event()
        self.device: DeviceInfo | None = None

    @property
    def healthy(self) -> bool:
        return self._healthy.is_set()

    @property
    def paused(self) -> bool:
        return self._paused.is_set()

    def pause(self) -> None:
        self._paused.set()

    def resume(self) -> None:
        # Discard whatever was captured while muted, so echo of our own speech
        # is never transcribed.
        self.buffer.clear()
        self._paused.clear()

    def _callback(self, indata, frames, time_info, status) -> None:  # noqa: ANN001
        if status:
            log.debug("audio callback status: %s", status)
        if self._paused.is_set():
            return
        try:
            self.buffer.write(indata[:, 0] if indata.ndim > 1 else indata)
        except Exception:  # noqa: BLE001 - callback must never propagate
            log.exception("ring buffer write failed")

    def start(self) -> tuple[bool, str]:
        try:
            import sounddevice as sd
        except (ImportError, OSError) as exc:
            return False, f"sounddevice unavailable: {exc}"

        self.device = find_input_device(self._cfg.input_match)
        if self.device is None:
            return False, f"no input device matching {self._cfg.input_match!r}"

        try:
            self._stream = sd.InputStream(
                device=self.device.index,
                channels=1,
                samplerate=self._cfg.sample_rate,
                blocksize=self._cfg.block_size,
                dtype="float32",
                callback=self._callback,
            )
            self._stream.start()
        except Exception as exc:  # noqa: BLE001 - PortAudio raises broadly
            return False, f"could not open {self.device.name}: {exc}"

        self._healthy.set()
        log.info("microphone started: %s", self.device.name)
        return True, ""

    def stop(self) -> None:
        self._healthy.clear()
        if self._stream is not None:
            try:
                self._stream.stop()
                self._stream.close()
            except Exception:  # noqa: BLE001
                log.exception("error closing audio stream")
            finally:
                self._stream = None
