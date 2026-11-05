#!/usr/bin/env python3
"""Measure what eye-follow would cost: webcam, face detection, and the shared microphone.

Why: the eyes should follow Adam using the NexiGO webcam, but that device also
carries Jarvis's only microphone, which stalled silently on 2026-11-05. This
reads the mic's capture rate from ALSA with the camera off, on, and off again,
so any harm shows as a number, and measures frame rate, detection time, CPU and
how steady the face position is. Frames stay in memory and are never saved or
sent; only numbers go to ~/homeai-bench/camera-<time>.json. The camera is in use
while it runs.

Usage:
    .venv/bin/python tools/avatar/probe_camera.py [--seconds 60] [--size 640x360] [--fps 10] [--quiet-s 10]
"""
from __future__ import annotations

import argparse
import json
import re
import sys
import time
from datetime import datetime
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

from homeai.avatar.facebuild import MODEL_PATH  # noqa: E402

BENCH_DIR = Path.home() / "homeai-bench"
MIC_RATE = 16000


def capture_status() -> Path | None:
    """The ALSA status file of the running capture stream on the webcam's card."""
    for card in Path("/proc/asound").glob("card*"):
        if "WEBCAM" in (card / "id").read_text().upper():
            return card / "pcm0c" / "sub0" / "status"
    return None


def hw_ptr(path: Path) -> int | None:
    found = re.search(r"hw_ptr\s*:\s*(\d+)", path.read_text())
    return int(found.group(1)) if found else None


class MicRate:
    """Samples per second from successive hw_ptr readings, one figure per ``tick``."""

    def __init__(self, path: Path | None) -> None:
        self.path, self.last, self.rates = path, None, []

    def tick(self) -> None:
        if self.path is None:
            return
        now, ptr = time.monotonic(), hw_ptr(self.path)
        if ptr is not None and self.last is not None and now > self.last[0]:
            self.rates.append((ptr - self.last[1]) / (now - self.last[0]))
        self.last = (now, ptr) if ptr is not None else None

    def summary(self) -> dict:
        r = np.array(self.rates or [0.0])
        return {"seconds": len(self.rates), "mean": round(float(r.mean())), "min": round(float(r.min())),
                "stalled_s": int((r < 0.5 * MIC_RATE).sum())}


def quiet(rate: MicRate, seconds: float) -> dict:
    rate.rates, rate.last = [], None
    end = time.monotonic() + seconds
    while time.monotonic() < end:
        rate.tick()
        time.sleep(1.0)
    return rate.summary()


def watch(rate: MicRate, seconds: float, size: tuple[int, int], fps: float) -> dict:
    import cv2
    import mediapipe as mp
    from mediapipe.tasks.python import BaseOptions
    from mediapipe.tasks.python.vision import FaceLandmarker, FaceLandmarkerOptions, RunningMode

    cam = cv2.VideoCapture(0, cv2.CAP_V4L2)
    cam.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
    cam.set(cv2.CAP_PROP_FRAME_WIDTH, size[0])
    cam.set(cv2.CAP_PROP_FRAME_HEIGHT, size[1])
    cam.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    if not cam.isOpened():
        return {"error": "could not open the camera"}
    options = FaceLandmarkerOptions(base_options=BaseOptions(model_asset_path=str(MODEL_PATH)),
                                    running_mode=RunningMode.VIDEO, num_faces=1)
    rate.rates, rate.last = [], None
    grabs, detects, centres, frames, found = [], [], [], 0, 0
    cpu0, wall0 = time.process_time(), time.monotonic()
    next_mic = wall0
    with FaceLandmarker.create_from_options(options) as landmarker:
        while (now := time.monotonic()) < wall0 + seconds:
            if now >= next_mic:
                rate.tick()
                next_mic += 1.0
            t = time.monotonic()
            ok, frame = cam.read()
            grabs.append(time.monotonic() - t)
            if not ok:
                continue
            frames += 1
            t = time.monotonic()
            image = mp.Image(image_format=mp.ImageFormat.SRGB, data=cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            result = landmarker.detect_for_video(image, int((t - wall0) * 1000))
            detects.append(time.monotonic() - t)
            if result.face_landmarks:
                found += 1
                nose = result.face_landmarks[0][1]
                centres.append((nose.x, nose.y))
            time.sleep(max(0.0, 1.0 / fps - (time.monotonic() - t)))
    wall = time.monotonic() - wall0
    cpu = time.process_time() - cpu0
    cam.release()
    c = np.array(centres or [(0.0, 0.0)])
    d = np.array(detects or [0.0]) * 1000
    return {"fps": round(frames / wall, 1), "found_share": round(found / max(frames, 1), 3),
            "detect_ms_p50": round(float(np.median(d)), 1), "detect_ms_p95": round(float(np.percentile(d, 95)), 1),
            "cpu_percent_of_one_core": round(100 * cpu / wall, 1), "nose_step_median": round(float(np.median(
                np.linalg.norm(np.diff(c, axis=0), axis=1))) if len(c) > 1 else 0.0, 5),
            "mic": rate.summary()}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--seconds", type=float, default=60.0)
    ap.add_argument("--size", default="640x360")
    ap.add_argument("--fps", type=float, default=10.0)
    ap.add_argument("--quiet-s", type=float, default=10.0)
    args = ap.parse_args()
    size = tuple(int(v) for v in args.size.split("x"))
    path = capture_status()
    print("mic status:", path, flush=True)
    rate = MicRate(path)
    result = {"size": size, "target_fps": args.fps, "before": quiet(rate, args.quiet_s)}
    print(result, flush=True)
    result["camera"] = watch(rate, args.seconds, size, args.fps)
    print(result["camera"], flush=True)
    result["after"] = quiet(rate, args.quiet_s)
    print(result["after"], flush=True)
    out = BENCH_DIR / f"camera-{datetime.now():%Y%m%d-%H%M%S}.json"
    out.write_text(json.dumps(result, indent=1))
    print(out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
