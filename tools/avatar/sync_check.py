#!/usr/bin/env python3
# Copyright (c) 2026 Adam Wosotowsky
# SPDX-License-Identifier: BUSL-1.1
"""Avatar Phase 1 functionality test: does the drawn sphere move with the voice?

Why: AVATAR_PLAN §10 Phase 1 exit. The scene's spring, lookahead and viseme
mapping could make the sphere open early or late without any unit test
noticing. This synthesises real speech with the real Piper voice, builds the
same utterance events the daemon publishes (timeline.chunk_utterance), plays
them through the real Scene, renders every frame headless (EGL) with the real
shaders, measures the drawn sphere's size from the pixels, and
cross-correlates it with the audio's loudness. Pass: best lag within
+/- --tolerance-ms and correlation >= --min-corr. Playback latency is
NOT tested here (it's HOMEAI_AVATAR_LAG_MS, measured by measure_output_latency.py).

Usage:
    .venv/bin/python tools/avatar/sync_check.py
    .venv/bin/python tools/avatar/sync_check.py --text "..." --rate 240 --no-render
Writes ~/homeai-bench/sync_check-<stamp>.json. Exit 0 pass, 1 fail, 2 setup error.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import datetime
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

BENCH_DIR = Path.home() / "homeai-bench"
DEFAULT_TEXT = ("Good morning, Adam. The kitchen lights are off, and it's going to rain "
                "this afternoon. Should I remind you to bring an umbrella?")
START = 1.0


def best_lag(audio: np.ndarray, mouth: np.ndarray, rate: float, max_lag_s: float = 0.25
             ) -> tuple[float, float]:
    """Lag (s) by which ``mouth`` trails ``audio`` at peak normalised correlation, and that peak.

    Positive lag: the sphere moves late. Pure; both series sampled at ``rate``.
    """
    if audio.shape != mouth.shape or audio.size < 3:
        raise ValueError("series must have the same length (>= 3)")
    a = (audio - audio.mean()) / (audio.std() or 1.0)
    m = (mouth - mouth.mean()) / (mouth.std() or 1.0)
    best = (0.0, -2.0)
    k = int(round(max_lag_s * rate))
    for lag in range(-k, k + 1):
        if lag >= 0:
            x, y = a[:a.size - lag], m[lag:]
        else:
            x, y = a[-lag:], m[:m.size + lag]
        c = float(np.dot(x, y) / x.size)
        if c > best[1]:
            best = (lag / rate, c)
    return best


def audio_track(chunks: list[tuple[float, bytes, int]], times: np.ndarray, window_s: float
                ) -> np.ndarray:
    """Loudness (dB, floored at -60) of the audio heard during [t, t+window) for each t."""
    out = np.full(times.size, -60.0)
    for start, pcm, sr in chunks:
        x = np.frombuffer(pcm, dtype=np.int16).astype(np.float64) / 32768.0
        for i, t in enumerate(times):
            a, b = int((t - start) * sr), int((t + window_s - start) * sr)
            if b <= 0 or a >= x.size:
                continue
            seg = x[max(a, 0):min(b, x.size)]
            rms = float(np.sqrt(np.mean(seg * seg))) if seg.size else 0.0
            if rms > 0:
                out[i] = max(out[i], max(-60.0, 20 * np.log10(rms)))
    return out


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--text", default=DEFAULT_TEXT)
    parser.add_argument("--rate", type=float, default=240.0, help="simulated frames per second")
    parser.add_argument("--size", type=int, default=240)
    parser.add_argument("--tolerance-ms", type=float, default=40.0)
    parser.add_argument("--min-corr", type=float, default=0.5)
    parser.add_argument("--no-render", action="store_true", help="use Frame.radius, skip GL")
    parser.add_argument("--lookahead-ms", type=float, default=None,
                        help="override scene.LOOKAHEAD_S (tuning; 0 shows the spring's raw lag)")
    args = parser.parse_args()

    from homeai.avatar import scene as scene_mod
    from homeai.avatar import timeline
    from homeai.avatar.scene import Scene
    from homeai.config import load
    from homeai.tts_engine import PiperEngine

    if args.lookahead_ms is not None:
        scene_mod.LOOKAHEAD_S = args.lookahead_ms / 1000

    engine = PiperEngine(load().tts)
    ok, why = engine.available()
    if not ok:
        print(f"setup: {why}")
        return 2
    t0 = time.monotonic()
    pcm_chunks = list(engine.synthesize(args.text))
    synth_s = time.monotonic() - t0
    starts = timeline.chain_starts(START, [c.num_samples for c in pcm_chunks],
                                   pcm_chunks[0].sample_rate)
    scene = Scene()
    scene.set_connected(True)
    scene.apply({"type": "state", "data": {"state": "speaking"}})
    for i, (c, s) in enumerate(zip(pcm_chunks, starts)):
        scene.apply({"type": "utterance",
                     "data": timeline.chunk_utterance(f"sync-{i}", s, c.pcm, c.sample_rate,
                                                      c.alignments)})
    end = starts[-1] + pcm_chunks[-1].num_samples / pcm_chunks[-1].sample_rate + 0.5
    times = np.arange(START - 0.5, end, 1 / args.rate)

    renderer = fbo = ctx = None
    if not args.no_render:
        try:
            from homeai.avatar import gl
            ctx, fbo = gl.headless((args.size, args.size))
            renderer = gl.SphereRenderer(ctx, points=6000)
        except Exception as exc:  # noqa: BLE001
            print(f"setup: headless GL unavailable ({exc}); use --no-render")
            return 2
    mouth = np.empty(times.size)
    for i, t in enumerate(times):
        frame = scene.frame(float(t))
        if renderer is None:
            mouth[i] = frame.radius
        else:
            renderer.draw(frame, float(t), fbo.size)
            px = np.frombuffer(fbo.read(components=3), dtype=np.uint8).reshape(args.size, args.size, 3)
            mouth[i] = gl.lit_radius(px)
    if renderer is not None:
        renderer.release()
        ctx.release()

    audio = audio_track([(s, c.pcm, c.sample_rate) for c, s in zip(pcm_chunks, starts)],
                        times, 1 / args.rate)
    lag, corr = best_lag(audio, mouth, args.rate)
    per_chunk = []
    for c, s in zip(pcm_chunks, starts):
        sel = (times >= s) & (times < s + c.num_samples / c.sample_rate)
        if sel.sum() > args.rate * 0.3:
            cl, cc = best_lag(audio[sel], mouth[sel], args.rate, 0.15)
            per_chunk.append({"start_s": round(s - START, 3), "lag_ms": round(cl * 1000, 1),
                              "corr": round(cc, 3),
                              "mouth_range": round(float(np.ptp(mouth[sel])), 4)})
    passed = abs(lag) * 1000 <= args.tolerance_ms and corr >= args.min_corr
    result = {"when": datetime.now().isoformat(timespec="seconds"), "text": args.text,
              "chunks": len(pcm_chunks), "aligned": engine.aligned,
              "audio_s": round(float(end - START - 0.5), 2), "synth_s": round(synth_s, 3),
              "rate": args.rate, "rendered": renderer is not None or not args.no_render,
              "lookahead_ms": round(scene_mod.LOOKAHEAD_S * 1000, 1),
              "lag_ms": round(lag * 1000, 1), "corr": round(corr, 3),
              "tolerance_ms": args.tolerance_ms, "min_corr": args.min_corr, "passed": passed,
              "per_chunk": per_chunk}
    BENCH_DIR.mkdir(exist_ok=True)
    out = BENCH_DIR / f"sync_check-{datetime.now():%Y%m%d-%H%M%S}.json"
    out.write_text(json.dumps(result, indent=2))
    print(f"{'PASS' if passed else 'FAIL'}  lag {result['lag_ms']:+.1f} ms (tolerance "
          f"{args.tolerance_ms:.0f})  corr {corr:.2f}  {len(pcm_chunks)} chunks, "
          f"alignments {'on' if engine.aligned else 'off'}  -> {out}")
    for pc in per_chunk:
        print(f"    chunk at {pc['start_s']:6.2f} s: lag {pc['lag_ms']:+.1f} ms  corr {pc['corr']:.2f}"
              f"  sphere range {pc['mouth_range']:.3f}")
    return 0 if passed else 1


if __name__ == "__main__":
    sys.exit(main())
