# Copyright (c) 2026 Adam Wosotowsky
# SPDX-License-Identifier: BUSL-1.1
# Licensed under the Business Source License 1.1; see LICENSE at the
# repository root. Converts to Apache-2.0 on 2030-09-15.
"""Eye-follow: the avatar looks at the person in front of the webcam.

Two halves. ``python -m homeai.avatar.eyes`` is a child process that holds the
camera, finds the biggest face and prints where it is, about ten times a second.
``EyeFollow`` runs in the avatar window: it starts that child while she is awake
and ends it otherwise, so the camera is open exactly as long as the process lives.

Privacy (AVATAR_PLAN §9): off unless HOMEAI_EYE_FOLLOW=1. One frame at a time in
memory, never saved or sent; only a position leaves the child. Nothing recognises
who it is. The child exits if the window goes away.
"""
from __future__ import annotations

import logging
import os
import subprocess
import sys
import threading
import time
from collections.abc import Callable, Iterable

log = logging.getLogger("homeai.avatar.eyes")

CAMERA_STATES = frozenset({"idle", "listening", "thinking", "speaking", "followup"})   # awake: everything else closes it
FPS = 10.0
SIZE = (640, 360)           # measured 2026-11-05: 5 ms a frame, 7% of a core (tools/avatar/probe_camera.py)
GAIN = (2.0, 2.0)           # x gaze per unit of image offset from the centre
CENTRE_Y = 0.5              # image height where she looks straight ahead
LIMIT = 0.9                 # gaze never quite reaches the edge of the eye
DEADBAND = 0.04             # gaze: smaller moves are ignored, so the eyes fix on you rather than tremble
LOST_S = 1.0                # no face for this long and she goes back to her own gaze
RETRY_S = 30.0              # after the camera process dies
MAX_MISSED_READS = 30
NO_FACE = "-"


def look_from(x: float, y: float) -> tuple[float, float]:
    """Gaze (+x screen right, +y up) toward a face at image position ``x``, ``y`` (0-1, the camera's own view).

    The camera faces the viewer, so someone at the screen's right is on the image's left.
    """
    def fit(v: float) -> float:
        return max(-LIMIT, min(LIMIT, v))

    return fit((0.5 - x) * GAIN[0]), fit((CENTRE_Y - y) * GAIN[1])


def biggest_face(faces: Iterable) -> tuple[float, float] | None:
    """Nose (x, y) of the widest face in ``faces`` (lists of landmarks with .x and .y); None if there are none."""
    best, width = None, 0.0
    for face in faces:
        xs = [p.x for p in face]
        if max(xs) - min(xs) > width:
            best, width = (face[1].x, face[1].y), max(xs) - min(xs)
    return best


def parse(line: str) -> tuple[float, float] | None:
    """A child's output line as (x, y); None for no face or anything malformed."""
    parts = line.split()
    if len(parts) != 2:
        return None
    try:
        return float(parts[0]), float(parts[1])
    except ValueError:
        return None


def track(read: Callable[[], object | None], detect: Callable[[object, int], Iterable], emit: Callable[[str], None],
          clock: Callable[[], float] = time.monotonic, sleep: Callable[[float], None] = time.sleep,
          running: Callable[[], bool] = lambda: True) -> int:
    """The child's loop: read a frame, emit a position. Returns its exit code."""
    missed = 0
    start = clock()
    while running():
        begun = clock()
        frame = read()
        if frame is None:
            missed += 1
            if missed >= MAX_MISSED_READS:
                return 1
            sleep(1.0 / FPS)
            continue
        missed = 0
        nose = biggest_face(detect(frame, int((begun - start) * 1000)))
        emit(NO_FACE if nose is None else "%.4f %.4f" % nose)
        sleep(max(0.0, 1.0 / FPS - (clock() - begun)))
    return 0


def _exit_with_parent() -> None:
    """The window holds our stdin: when it closes, the window has gone, so the camera must go too."""
    sys.stdin.read()
    os._exit(0)


def main() -> int:
    import cv2
    import mediapipe as mp
    from mediapipe.tasks.python import BaseOptions
    from mediapipe.tasks.python.vision import FaceLandmarker, FaceLandmarkerOptions, RunningMode

    from .facebuild import MODEL_PATH

    threading.Thread(target=_exit_with_parent, name="parent-watch", daemon=True).start()
    camera = cv2.VideoCapture(0, cv2.CAP_V4L2)
    camera.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
    camera.set(cv2.CAP_PROP_FRAME_WIDTH, SIZE[0])
    camera.set(cv2.CAP_PROP_FRAME_HEIGHT, SIZE[1])
    camera.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    if not camera.isOpened():
        print("eyes: could not open the camera", file=sys.stderr)
        return 1

    def read():
        ok, frame = camera.read()
        return cv2.cvtColor(frame, cv2.COLOR_BGR2RGB) if ok else None

    def emit(line: str) -> None:
        print(line, flush=True)

    options = FaceLandmarkerOptions(base_options=BaseOptions(model_asset_path=str(MODEL_PATH)),
                                    running_mode=RunningMode.VIDEO, num_faces=3)
    try:
        with FaceLandmarker.create_from_options(options) as landmarker:
            def detect(frame, ms):
                image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame)
                return landmarker.detect_for_video(image, ms).face_landmarks

            return track(read, detect, emit)
    except BrokenPipeError:
        return 0
    finally:
        camera.release()


def spawn() -> subprocess.Popen:
    env = {**os.environ, "GLOG_minloglevel": "2"}
    return subprocess.Popen([sys.executable, "-m", "homeai.avatar.eyes"], stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                            text=True, env=env)


class EyeFollow:
    """The window's side: keeps the camera process running while she's awake, and gives the gaze to look at."""

    def __init__(self, start: Callable[[], object] = spawn, clock: Callable[[], float] = time.monotonic) -> None:
        self._start = start
        self._clock = clock
        self._proc = None
        self._retry_at = 0.0
        self._look: tuple[float, float] | None = None
        self._seen_at = float("-inf")
        self._lock = threading.Lock()

    def see(self, line: str) -> None:
        """Take one line from the child."""
        face = parse(line)
        if face is None:
            return
        look = look_from(*face)
        with self._lock:
            self._seen_at = self._clock()
            old = self._look
            if old is None or max(abs(look[0] - old[0]), abs(look[1] - old[1])) >= DEADBAND:
                self._look = look

    def _read(self, proc) -> None:
        try:
            for line in proc.stdout:
                self.see(line)
        except (OSError, ValueError):
            pass                                # closed under us: stopping

    def update(self, now: float, state: str) -> tuple[float, float] | None:
        """Start or stop the camera for shown ``state``; the gaze to look at, or None for her own."""
        awake = state in CAMERA_STATES
        if self._proc is not None and self._proc.poll() is not None:
            log.warning("eye-follow camera process ended (%s): retrying in %.0f s", self._proc.returncode, RETRY_S)
            self._proc = None
            self._retry_at = now + RETRY_S
        if not awake:
            self.stop()
            return None
        if self._proc is None and now >= self._retry_at:
            self._proc = self._start()
            threading.Thread(target=self._read, args=(self._proc,), name="eyes-read", daemon=True).start()
        with self._lock:
            if self._clock() - self._seen_at > LOST_S:
                self._look = None
            return self._look

    def stop(self) -> None:
        """Close the camera. Never waits: this runs on the render loop."""
        proc, self._proc = self._proc, None
        with self._lock:
            self._look, self._seen_at = None, float("-inf")
        if proc is not None:
            proc.terminate()
            threading.Thread(target=self._reap, args=(proc,), name="eyes-reap", daemon=True).start()

    @staticmethod
    def _reap(proc) -> None:
        try:
            proc.wait(timeout=2.0)
        except subprocess.TimeoutExpired:
            proc.kill()
            proc.wait()


if __name__ == "__main__":
    sys.exit(main())
