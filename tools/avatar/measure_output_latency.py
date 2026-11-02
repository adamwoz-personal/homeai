#!/usr/bin/env python3
# Copyright (c) 2026 Adam Wosotowsky
# SPDX-License-Identifier: BUSL-1.1
"""Avatar Phase 0 spike: how long from writing PCM to aplay until it is heard?

Why: the avatar schedules the mouth from the speech timeline (AVATAR_PLAN
§3.2): start_mono = first PCM write + HOMEAI_AVATAR_LAG_MS. The relay writes
PCM faster than real time, so this lag is the device's output latency, which
only a measurement can give. This tool produces that number.

How: a train of short tone bursts is written to the exact aplay command
tts.py uses (same device, 22050 Hz S16 mono) while the Jarvis mic records.
For each burst, lag = heard (mic sample time on the monotonic clock) -
(first PCM write + the burst's offset in the stream). The mic path adds its
own latency; PortAudio's ADC timestamps are used to remove it when the
backend provides them (reported as "adc"), otherwise callback time
("callback"; an upper bound). Pass: 20 bursts found, spread (max-min) within
+/-10 ms of the median.

Usage:
    .venv/bin/python tools/avatar/measure_output_latency.py [--clicks 20] [--volume 0.3]
Audible: ~13 s of quiet beeps. Writes ~/homeai-bench/output_latency-<stamp>.json.
Exit 0 pass, 1 unstable/missing clicks, 2 setup error.
"""

from __future__ import annotations

import argparse
import json
import statistics
import subprocess
import sys
import threading
import time
from datetime import datetime
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

PLAY_RATE = 22050
LEAD_S = 0.5
SPACING_S = 0.6
BURST_MS = 6.0
TONE_HZ = 2000.0
SEARCH_S = 0.45  # a lag beyond this is a failure, and stays clear of the next burst
STABLE_MS = 10.0
BENCH_DIR = Path.home() / "homeai-bench"


def click_train(n: int, volume: float, rate: int = PLAY_RATE) -> tuple[np.ndarray, list[float]]:
    """Int16 mono PCM with ``n`` Hann-windowed tone bursts; returns (pcm, offsets_s)."""
    length = int(rate * (LEAD_S + n * SPACING_S + 0.5))
    pcm = np.zeros(length, dtype=np.float32)
    burst_len = int(rate * BURST_MS / 1000)
    t = np.arange(burst_len) / rate
    burst = np.sin(2 * np.pi * TONE_HZ * t) * np.hanning(burst_len) * volume
    offsets = []
    for k in range(n):
        start = int(rate * (LEAD_S + k * SPACING_S))
        pcm[start:start + burst_len] = burst
        offsets.append(start / rate)
    return (pcm * 32767).astype(np.int16), offsets


def onsets(signal: np.ndarray, rate: int, t0: float, expected: list[float],
           search_s: float = SEARCH_S) -> list[float | None]:
    """Monotonic time of the first loud sample after each expected time.

    ``signal`` is float mono whose sample i was captured at t0 + i/rate. A
    first difference acts as a high-pass so room hum doesn't count; the
    threshold sits well above the noise but below the burst peak.
    """
    hp = np.abs(np.diff(signal, prepend=signal[:1]))
    noise = float(np.median(hp)) + 1e-9
    found: list[float | None] = []
    for when in expected:
        a = int((when - t0) * rate)
        b = int((when + search_s - t0) * rate)
        if a < 0 or b > len(hp):
            found.append(None)
            continue
        window = hp[a:b]
        peak = float(window.max())
        if peak < 8 * noise:
            found.append(None)
            continue
        threshold = max(8 * noise, 0.3 * peak)
        idx = int(np.argmax(window >= threshold))
        found.append(t0 + (a + idx) / rate)
    return found


def verdict(lags_ms: list[float | None], n: int) -> dict:
    """Pure summary; stable when every burst was heard within +/-STABLE_MS of the median."""
    heard = [x for x in lags_ms if x is not None]
    if not heard:
        return {"passed": False, "heard": 0, "expected": n, "reason": "no bursts heard"}
    med = statistics.median(heard)
    spread = max(abs(x - med) for x in heard)
    reasons = []
    if len(heard) < n:
        reasons.append(f"heard {len(heard)} of {n}")
    if spread > STABLE_MS:
        reasons.append(f"spread +/-{spread:.1f} ms > {STABLE_MS:.0f}")
    return {"passed": not reasons, "heard": len(heard), "expected": n,
            "median_ms": round(med, 1), "spread_ms": round(spread, 1),
            "min_ms": round(min(heard), 1), "max_ms": round(max(heard), 1),
            "reason": "; ".join(reasons)}


class Recorder:
    """Mic capture with each block stamped on the monotonic clock."""

    def __init__(self, sd, device: int, rate: int):
        self.rate = rate
        self.blocks: list[np.ndarray] = []
        self.t0_adc: float | None = None
        self.t0_cb: float | None = None
        self.adc_supported = False
        self._lock = threading.Lock()
        self.stream = sd.InputStream(device=device, channels=1, samplerate=rate,
                                     dtype="float32", callback=self._cb)

    def _cb(self, data, frames, tinfo, status):  # noqa: ARG002 - PortAudio signature
        now = time.monotonic()
        with self._lock:
            if self.t0_cb is None:
                self.t0_cb = now - frames / self.rate
                adc = tinfo.inputBufferAdcTime
                if adc and tinfo.currentTime and 0 < tinfo.currentTime - adc < 1.0:
                    self.adc_supported = True
                    self.t0_adc = now - (tinfo.currentTime - adc)
            self.blocks.append(data[:, 0].copy())

    def signal(self) -> np.ndarray:
        with self._lock:
            return np.concatenate(self.blocks) if self.blocks else np.zeros(0, np.float32)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[1])
    parser.add_argument("--clicks", type=int, default=20)
    parser.add_argument("--volume", type=float, default=0.3)
    parser.add_argument("--input-match", default=None,
                        help="mic name substring (default: Jarvis's HOMEAI_INPUT_MATCH)")
    args = parser.parse_args()

    try:
        import sounddevice as sd
        from homeai.audio import find_input_device
        from homeai.config import Config
    except (ImportError, OSError) as exc:
        print(f"setup error: {exc}")
        return 2
    cfg = Config()
    match = args.input_match or cfg.audio.input_match
    mic = find_input_device(match)
    if mic is None:
        print(f"setup error: no input device matching {match!r} (the daemon may hold it;"
              " try --input-match HD_WEBCAM or stop homeai briefly)")
        return 2
    device = cfg.audio.output_device

    pcm, offsets = click_train(args.clicks, args.volume)
    rec = Recorder(sd, mic.index, mic.sample_rate)
    aplay = None
    try:
        rec.stream.start()
        time.sleep(0.5)  # let capture settle before anything plays
        aplay = subprocess.Popen(
            ["aplay", "-q", "-D", device, "-f", "S16_LE", "-r", str(PLAY_RATE), "-c", "1"],
            stdin=subprocess.PIPE, stderr=subprocess.PIPE)
        data = pcm.tobytes()
        first_write = time.monotonic()
        aplay.stdin.write(data[:4096])
        aplay.stdin.flush()
        aplay.stdin.write(data[4096:])
        aplay.stdin.close()
        rc = aplay.wait(timeout=len(pcm) / PLAY_RATE + 10)
        time.sleep(0.3)
    except (OSError, subprocess.TimeoutExpired) as exc:
        if aplay and aplay.poll() is None:
            aplay.kill()
        print(f"setup error: playback failed: {exc}")
        return 2
    finally:
        rec.stream.stop()
        rec.stream.close()
    if rc != 0:
        print(f"setup error: aplay exited {rc}: {aplay.stderr.read().decode(errors='replace')[-200:]}")
        return 2

    signal = rec.signal()
    if rec.t0_cb is None or not len(signal):
        print("setup error: the mic delivered no audio (is the daemon holding it? "
              "stop homeai for the run)")
        return 2
    expected = [first_write + o for o in offsets]
    results = {}
    for clock, t0 in (("adc", rec.t0_adc), ("callback", rec.t0_cb)):
        if t0 is None:
            continue
        heard = onsets(signal, rec.rate, t0, expected)
        lags = [None if h is None else round((h - e) * 1000, 1) for h, e in zip(heard, expected)]
        results[clock] = {"lags_ms": lags, **verdict(lags, args.clicks)}

    best = results.get("adc") or results["callback"]
    report = {"when": datetime.now().isoformat(timespec="seconds"), "output_device": device,
              "mic": mic.name, "mic_rate": rec.rate, "clicks": args.clicks,
              "adc_timestamps": rec.adc_supported, "results": results,
              "recommended_lag_ms": best.get("median_ms"), "passed": best["passed"]}
    BENCH_DIR.mkdir(exist_ok=True)
    out = BENCH_DIR / f"output_latency-{datetime.now():%Y%m%d-%H%M%S}.json"
    out.write_text(json.dumps(report, indent=2))

    print(f"output {device} -> mic {mic.name} @ {rec.rate} Hz; adc timestamps: {rec.adc_supported}")
    for clock, r in results.items():
        if r["heard"]:
            print(f"  {clock:8} heard {r['heard']}/{r['expected']}  median {r['median_ms']} ms"
                  f"  spread +/-{r['spread_ms']} ms  range {r['min_ms']}..{r['max_ms']}")
        else:
            print(f"  {clock:8} {r['reason']}")
    print(("PASS" if best["passed"] else f"FAIL: {best['reason']}") + f"  -> {out}")
    return 0 if best["passed"] else 1


if __name__ == "__main__":
    sys.exit(main())
