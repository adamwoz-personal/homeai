#!/usr/bin/env python3
# Copyright (c) 2026 Adam Wosotowsky
# SPDX-License-Identifier: BUSL-1.1
# Licensed under the Business Source License 1.1; see LICENSE at the
# repository root. Converts to Apache-2.0 on 2030-09-15.
"""Measure whether barge-in is viable without acoustic echo cancellation.

The question
------------
For the assistant to be interruptible, the microphone must stay open while it
is speaking. The risk is obvious: openWakeWord may hear *Jarvis* saying
something and treat it as a wake word, cutting off its own answer at random.

This has already happened once in production -- a wake fired 0.487 s after a
reply finished, from residual audio in the detector's internal context. What
is unknown is the *rate* while audio is actively playing, and that rate
decides the architecture:

* **Low false-trigger rate** -> keep raw ALSA, open the mic during playback,
  and gate on the wake word. Cheap and low-risk.
* **High rate** -> acoustic echo cancellation is mandatory, which on this box
  means switching the sound card to its ``pro-audio`` profile and re-plumbing
  desktop audio through PipeWire. Expensive and risky.

Measuring first is much cheaper than implementing the wrong one.

Method
------
Speak a long passage through the real speakers while capturing the real
microphone, then run the detector over the captured audio and count how many
times it would have fired. This is the true acoustic path -- room, speaker,
and microphone -- not a simulation.

Reported alongside the count is the peak detector score, which matters just as
much: a peak far below threshold means headroom to keep the mic open safely,
whereas a peak that merely grazes the threshold means the margin is luck.
"""

from __future__ import annotations

import argparse
import sys
import threading
import time

import numpy as np

from homeai.config import Config
from homeai.audio import Microphone
from homeai.tts import Speaker
from homeai.wake import OpenWakeWordDetector

# Deliberately contains phonetically similar material. "Hey" and "Jarvis"
# fragments in ordinary speech are the realistic worst case, and a passage of
# unrelated prose would understate the risk.
PASSAGE = (
    "Here is a long explanation, the kind that would normally run for over a "
    "minute without pause. Hey, that reminds me of something worth saying. "
    "Jarvis is a name that appears in fiction quite often. "
    "Einstein proposed the theory of relativity in nineteen oh five, and it "
    "changed how we understand space and time. Special relativity deals with "
    "objects moving at constant speed, especially near the speed of light. "
    "General relativity extended that to gravity and acceleration. "
    "Heavy objects curve spacetime, and that curvature is what we experience "
    "as gravitational attraction between masses."
)


ORDINARY_PASSAGE = (
    "Einstein proposed the theory of relativity in nineteen oh five, and it "
    "changed how we understand space and time. Special relativity deals with "
    "objects moving at constant speed, especially near the speed of light. "
    "General relativity extended that to gravity and acceleration. "
    "Heavy objects curve spacetime, and that curvature is what we experience "
    "as gravitational attraction between masses. The bending of starlight "
    "during a solar eclipse confirmed the prediction in nineteen nineteen, "
    "and satellite navigation must correct for these effects every day."
)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--threshold",
        type=float,
        default=None,
        help="override the configured wake threshold",
    )
    parser.add_argument(
        "--ordinary",
        action="store_true",
        help="speak neutral prose containing no wake-word fragments, to "
             "measure the realistic (not adversarial) false-trigger risk",
    )
    parser.add_argument(
        "--silence-only",
        action="store_true",
        help="capture without speaking, to establish a baseline",
    )
    args = parser.parse_args()

    passage = ORDINARY_PASSAGE if args.ordinary else PASSAGE

    cfg = Config()
    threshold = args.threshold if args.threshold is not None else cfg.wake.threshold

    print(f"wake model:     {cfg.wake.model}")
    print(f"threshold:      {threshold}")
    print(f"input device:   {cfg.audio.input_match}")
    print(f"output device:  {cfg.audio.output_device}")
    print()

    detector = OpenWakeWordDetector(cfg.wake.model, threshold)
    loaded, problem = detector.load()
    if not loaded:
        print(f"ERROR: wake model did not load: {problem}", file=sys.stderr)
        print("Refusing to report a verdict from a detector that cannot fire.",
              file=sys.stderr)
        return 1
    mic = Microphone(cfg.audio)

    captured: list[np.ndarray] = []
    stop = threading.Event()

    def capture() -> None:
        """Drain the microphone continuously until told to stop."""
        cursor = 0
        while not stop.is_set():
            frames, cursor = mic.buffer.read_new(cursor)
            if frames is not None and len(frames):
                captured.append(frames.copy())
            else:
                time.sleep(0.01)

    ok, detail = mic.start()
    if not ok:
        print(f'ERROR: microphone did not start: {detail}', file=sys.stderr)
        return 1
    worker = threading.Thread(target=capture, daemon=True)
    worker.start()

    # Let the capture path settle before making any sound, so start-up
    # transients are not counted as detections.
    time.sleep(1.0)

    if args.silence_only:
        print("capturing 20s of silence (baseline)...")
        time.sleep(20.0)
    else:
        print("speaking through the real speakers with the mic OPEN...")
        speaker = Speaker(cfg.tts, cfg.audio.output_device)
        result = speaker.say_safe(passage)
        print(f"  spoke for {result.total_ms / 1000:.1f}s (ok={result.ok})")
        # Trailing capture: the detector holds ~1s of internal context, so a
        # late trigger is exactly the failure already seen in production.
        time.sleep(2.0)

    stop.set()
    worker.join(timeout=5.0)
    mic.stop()

    if not captured:
        print("ERROR: captured no audio at all", file=sys.stderr)
        return 1

    audio = np.concatenate(captured)
    duration = len(audio) / cfg.audio.sample_rate
    rms = float(np.sqrt(np.mean(audio.astype(np.float64) ** 2)))
    print(f"\ncaptured {duration:.1f}s, rms={rms:.4f}")

    # Replay the captured audio through the detector in the same frame size
    # the daemon uses, so the result reflects real behaviour.
    block = cfg.audio.block_size
    detections = 0
    peak = 0.0
    detection_times: list[float] = []

    for start in range(0, len(audio) - block + 1, block):
        frame = audio[start : start + block]
        score = _score(detector, frame)
        peak = max(peak, score)
        if score >= threshold:
            detections += 1
            detection_times.append(start / cfg.audio.sample_rate)

    print(f"peak score:     {peak:.4f}")
    print(f"threshold:      {threshold}")
    print(f"margin:         {threshold - peak:+.4f}")
    print(f"false triggers: {detections}")
    if detection_times:
        preview = ", ".join(f"{t:.1f}s" for t in detection_times[:10])
        print(f"  at: {preview}")

    print()
    if detections:
        print("VERDICT: mic cannot stay open without echo cancellation.")
    elif peak > threshold * 0.7:
        print("VERDICT: no triggers, but margin is thin. Risky without AEC.")
    else:
        print("VERDICT: comfortable margin. Open mic during playback is viable.")
    return 0


def _score(detector: OpenWakeWordDetector, frame: np.ndarray) -> float:
    """Get the raw detector score. Requires an already-loaded model."""
    model = getattr(detector, "_model", None)
    if model is None:
        raise RuntimeError("wake model not loaded; scores would be meaningless")
    pcm = (np.clip(frame, -1.0, 1.0) * 32767).astype(np.int16)
    scores = model.predict(pcm)
    return max(scores.values()) if scores else 0.0


if __name__ == "__main__":
    raise SystemExit(main())
