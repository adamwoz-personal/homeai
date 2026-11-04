"""Caption text for the avatar window: wrap, lay out and rasterise (AVATAR_PLAN §4.7).

``wrap`` and ``layout`` are pure (any ``measure`` callable), so the rules are
unit tested without fonts; ``Rasteriser`` turns the frame's captions into an
RGBA image with pygame's font module (no display needed), cached so an
unchanged caption costs nothing per frame.

Layout, bottom up: Jarvis's current sentence (larger, up to 3 lines), what was
heard above it (smaller, dimmer, up to 2 lines); a status line ("coding mode")
sits under the sphere's centre instead.
"""
from __future__ import annotations

from typing import Callable

import numpy as np

ELLIPSIS = "\u2026"
MAX_LINES = {"jarvis": 3, "user": 2, "status": 1}
SCALE = {"jarvis": 1 / 20, "user": 1 / 26, "status": 1 / 22}    # font px per window px (height)
TINT = {"jarvis": (235, 245, 255), "user": (160, 175, 190), "status": (190, 180, 230)}
MARGIN = 0.06               # of the width, each side
ALPHA_STEPS = 20            # fades re-rasterise at most this many times


def wrap(text: str, measure: Callable[[str], int], max_w: int, max_lines: int) -> list[str]:
    """Greedy word wrap into at most ``max_lines``; overflow ends the last line with an ellipsis.

    A single word wider than ``max_w`` gets a line of its own (never split).
    """
    if max_lines < 1:
        return []
    lines: list[str] = []
    words = text.split()
    current = ""
    for word in words:
        trial = f"{current} {word}" if current else word
        if not current or measure(trial) <= max_w:
            current = trial
            continue
        lines.append(current)
        current = word
        if len(lines) == max_lines:
            break
    else:
        if current:
            lines.append(current)
        return lines
    # Overflow: trim the last allowed line until the ellipsis fits.
    last = lines[-1]
    while last and measure(last + ELLIPSIS) > max_w and " " in last:
        last = last.rsplit(" ", 1)[0]
    lines[-1] = last + ELLIPSIS
    return lines


def layout(captions: list[tuple[str, str, float]], size: tuple[int, int],
           measure: Callable[[str, str], int]) -> list[tuple[str, str, float, int]]:
    """Lines to draw: ``(who, line, alpha, baseline_y_px)``; y grows downwards.

    ``measure(who, text)`` gives the pixel width in that speaker's font.
    """
    w, h = size
    max_w = int(w * (1 - 2 * MARGIN))
    out = []
    bottom = int(h * 0.96)
    for who in ("jarvis", "user"):          # stacked upwards from the bottom
        for cap_who, text, alpha in captions:
            if cap_who != who:
                continue
            lines = wrap(text, lambda s, who=who: measure(who, s), max_w, MAX_LINES[who])
            step = int(h * SCALE[who] * 1.25)
            for i, line in enumerate(reversed(lines)):
                out.append((who, line, alpha, bottom - i * step))
            bottom -= step * len(lines) + int(h * 0.015)
    for cap_who, text, alpha in captions:
        if cap_who == "status":
            out.append(("status", text, alpha, int(h * 0.5 + h * 0.30)))
    return out


class Rasteriser:
    """Captions -> RGBA uint8 image (h, w, 4), top row first; cached by content."""

    def __init__(self) -> None:
        import pygame

        self._pg = pygame
        pygame.font.init()
        self._fonts: dict[tuple[str, int], object] = {}
        self._key = None
        self._image: np.ndarray | None = None

    def _font(self, who: str, h: int):
        px = max(10, int(h * SCALE[who]))
        key = (who, px)
        if key not in self._fonts:
            self._fonts[key] = self._pg.font.Font(None, int(px * 1.35))   # default font is small
        return self._fonts[key]

    def image(self, captions: list[tuple[str, str, float]], size: tuple[int, int]) -> np.ndarray | None:
        """None when there is nothing to draw."""
        q = tuple((w, t, round(a * ALPHA_STEPS)) for w, t, a in captions if a > 0)
        if not q:
            return None
        key = (q, size)
        if key == self._key:
            return self._image
        w, h = size
        surf = self._pg.Surface((w, h), self._pg.SRCALPHA)
        lines = layout(captions, size, lambda who, s: self._font(who, h).size(s)[0])
        for who, line, alpha, baseline in lines:
            font = self._font(who, h)
            text = font.render(line, True, TINT[who])
            text.set_alpha(int(255 * alpha))
            x = (w - text.get_width()) // 2
            surf.blit(text, (x, baseline - text.get_height()))
        rgba = np.dstack([self._pg.surfarray.array3d(surf).transpose(1, 0, 2),
                          self._pg.surfarray.array_alpha(surf).T])
        self._image = np.ascontiguousarray(np.flipud(rgba))     # GL textures start at the bottom
        self._key = key
        return self._image
