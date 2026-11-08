# Copyright (c) 2026 Adam Wosotowsky
# SPDX-License-Identifier: BUSL-1.1
# Licensed under the Business Source License 1.1; see LICENSE at the
# repository root. Converts to Apache-2.0 on 2030-09-15.
"""Keep only the owner's words from a recording two people spoke into (ART_FEATURES_PLAN S1).

In Adam's battleship session (2026-11-07) a child's "it has two heads..." and
Adam's "again" came in one recording, scored 0.44-0.52 as a whole, and the
child's words changed the picture. A whole-sentence speaker check can't split
that, so this scores the audio in short windows against the owner's profile:

1. ``window_sims``: 1.5 s windows every 0.25 s, cosine to the owner.
2. ``other_spans``: runs of at least ``min_windows`` windows that are all
   confidently someone else (below ``different_below``). A single low window
   is never enough: the owner's own short or quiet stretch can score low.
3. Only if there are such spans AND some window is clearly the owner is the
   recording mixed; then whisper is asked for word timings and
   ``keep_owner_words`` drops the words whose middle falls in another voice's span.

Bench (tools/bench_segment_speakers.py, Piper voices, 2026-11-08): owner
windows >= 0.68, other <= 0.16; owner words kept 99/99, other words dropped
35/44 (leaks at the very start of a clip). Synthetic: unverified on the household.
"""
from __future__ import annotations

import re
from typing import Callable

import numpy as np

SR = 16000
WINDOW_S = 1.5
HOP_S = 0.25
MIN_WINDOWS = 2
STAMP = re.compile(r"^\[(\d+):(\d+):([\d.]+) --> (\d+):(\d+):([\d.]+)\]\s*(.*)$")


def window_sims(similarity: Callable[[np.ndarray], float | None], audio: np.ndarray,
                window_s: float = WINDOW_S, hop_s: float = HOP_S) -> list[tuple[float, float, float | None]]:
    """(start s, end s, similarity to the owner or None) for each window."""
    n, hop = int(window_s * SR), int(hop_s * SR)
    if len(audio) < n:
        return []
    return [(start / SR, (start + n) / SR, similarity(audio[start:start + n]))
            for start in range(0, len(audio) - n + 1, hop)]


def other_spans(sims, below: float, min_windows: int = MIN_WINDOWS) -> list[tuple[float, float]]:
    """Merged time spans covered by runs of >= ``min_windows`` windows all confidently not the owner."""
    spans, run = [], []
    for item in sims + [(0.0, 0.0, None)]:
        if item[2] is not None and item[2] < below:
            run.append(item)
            continue
        if len(run) >= min_windows:
            start, end = run[0][0], run[-1][1]
            if spans and start <= spans[-1][1]:
                spans[-1] = (spans[-1][0], end)
            else:
                spans.append((start, end))
        run = []
    return spans


def is_mixed(sims, spans, same_above: float) -> bool:
    """Another voice's span plus at least one window that is clearly the owner."""
    return bool(spans) and any(s is not None and s >= same_above for _, _, s in sims)


def parse_words(stdout: str) -> list[tuple[float, float, str]]:
    """whisper-cli ``-ml 1 -sow`` output: one ``[hh:mm:ss.mmm --> hh:mm:ss.mmm]  word`` per line."""
    words = []
    for line in stdout.splitlines():
        m = STAMP.match(line.strip())
        if m and m.group(7).strip():
            h1, m1, s1, h2, m2, s2 = (float(x) for x in m.groups()[:6])
            words.append((h1 * 3600 + m1 * 60 + s1, h2 * 3600 + m2 * 60 + s2, m.group(7).strip()))
    return words


def keep_owner_words(words, spans) -> tuple[str, str]:
    """(kept text, dropped text): a word is dropped if its middle falls inside another voice's span."""
    kept, dropped = [], []
    for start, end, word in words:
        mid = (start + end) / 2
        (dropped if any(a <= mid <= b for a, b in spans) else kept).append(word)
    return re.sub(r"\s+", " ", " ".join(kept)).strip(), " ".join(dropped)
