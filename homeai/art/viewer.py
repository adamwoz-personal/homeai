"""The picture window for art mode (AVATAR_PLAN 6a step 8a).

A separate window and process, so the avatar keeps running as it is. It
follows the latest ``art`` event on the event bus (published by
homeai/art/voice.py): the current picture large, a status line, and the
pictures that led to it as a strip of thumbnails along the bottom.

Pure parts (``layout``, ``status_line``, ``fit``) are unit-tested; ``main``
only draws. Started by art mode as the transient unit ``homeai-art-viewer``;
to try it by hand: ``.venv/bin/python -m homeai.art.viewer``.
"""
from __future__ import annotations

import logging
import os
import queue
import sys

log = logging.getLogger("homeai.art.viewer")

SIZE = (1000, 1000)
STRIP_H = 150
STATUS_H = 56
PAD = 12
MAX_THUMBS = 8
BG, FG, DIM = (14, 16, 22), (230, 232, 240), (130, 136, 150)


def fit(w: int, h: int, box_w: int, box_h: int) -> tuple[int, int]:
    """Largest size with the picture's shape that fits the box (never zero)."""
    if w <= 0 or h <= 0 or box_w <= 0 or box_h <= 0:
        return (max(1, box_w), max(1, box_h))
    scale = min(box_w / w, box_h / h)
    return (max(1, round(w * scale)), max(1, round(h * scale)))


def layout(width: int, height: int, thumbs: int) -> dict:
    """Rectangles (x, y, w, h): the main picture, the status line and one box per thumbnail."""
    strip = STRIP_H if thumbs > 1 else 0
    main_h = max(1, height - STATUS_H - strip - PAD * (4 if strip else 3))
    out = {"main": (PAD, PAD, max(1, width - 2 * PAD), main_h),
           "status": (PAD, PAD + main_h + PAD, max(1, width - 2 * PAD), STATUS_H), "thumbs": []}
    if strip:
        n = min(thumbs, MAX_THUMBS)
        box = min(STRIP_H, (width - PAD * (n + 1)) // n)
        y = height - PAD - STRIP_H
        out["thumbs"] = [(PAD + i * (box + PAD), y, box, STRIP_H) for i in range(n)]
    return out


def status_line(state: dict) -> str:
    if not state.get("active"):
        return "Art mode is off."
    parts = []
    current = state.get("current") or {}
    if current.get("instruction"):
        parts.append(f"“{current['instruction']}”")
    if state.get("drawing"):
        parts.append("drawing")
    if state.get("waiting"):
        parts.append(f"{state['waiting']} waiting")
    if state.get("status"):
        parts.append(state["status"])
    return "  ·  ".join(parts) or "Say what to draw."


def art_state(event: dict, current: dict) -> dict:
    """The window's state after a bus event: an ``art`` event, or the one inside a ``snapshot``."""
    if event.get("type") == "snapshot":
        event = ((event.get("data") or {}).get("events") or {}).get("art") or {}
    if event.get("type") != "art" or not isinstance(event.get("data"), dict):
        return current
    return event["data"]


def thumbs_of(state: dict) -> list[dict]:
    """The last MAX_THUMBS steps of the lineage, oldest first."""
    return list(state.get("lineage") or [])[-MAX_THUMBS:]


class Images:
    """Loads each picture once per size; a missing or half-written file is skipped, not fatal."""

    def __init__(self, pygame) -> None:
        self.pg, self._cache = pygame, {}

    def get(self, path: str | None, box: tuple[int, int]):
        if not path:
            return None
        key = (path, box)
        if key not in self._cache:
            try:
                img = self.pg.image.load(path).convert()
            except (OSError, self.pg.error):
                return None
            self._cache[key] = self.pg.transform.smoothscale(img, fit(*img.get_size(), *box))
            if len(self._cache) > 64:
                self._cache.pop(next(iter(self._cache)))
        return self._cache[key]


def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)-7s %(name)s: %(message)s")
    os.environ.setdefault("SDL_VIDEODRIVER", "x11")
    os.environ.setdefault("SDL_VIDEO_X11_WMCLASS", "homeai-art")
    import pygame

    from homeai.avatar.client import BusClient

    events: queue.SimpleQueue = queue.SimpleQueue()
    client = BusClient(lambda event: events.put(event) if event.get("type") in ("art", "snapshot") else None,
                       path=os.environ.get("HOMEAI_AVATAR_SOCKET") or None)
    pygame.init()
    screen = pygame.display.set_mode(SIZE, pygame.RESIZABLE)
    pygame.display.set_caption("Jarvis — pictures")
    font = pygame.font.SysFont("dejavusans", 22)
    images = Images(pygame)
    state: dict = {"active": True, "status": "Starting..."}
    client.start()
    clock = pygame.time.Clock()
    shots = os.environ.get("HOMEAI_ART_VIEWER_SHOTS")    # live checks: save the window after each new picture
    shot_of = None
    try:
        while True:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    return 0
                if event.type == pygame.KEYDOWN and event.key in (pygame.K_ESCAPE, pygame.K_q):
                    return 0
            while not events.empty():
                state = art_state(events.get(), state)
            width, height = screen.get_size()
            thumbs = thumbs_of(state)
            boxes = layout(width, height, len(thumbs))
            screen.fill(BG)
            x, y, w, h = boxes["main"]
            picture = images.get((state.get("current") or {}).get("image"), (w, h))
            if picture is not None:
                pw, ph = picture.get_size()
                screen.blit(picture, (x + (w - pw) // 2, y + (h - ph) // 2))
            x, y, w, h = boxes["status"]
            text = font.render(status_line(state)[:160], True, FG)
            screen.blit(text, (x, y + (h - text.get_height()) // 2))
            current = (state.get("current") or {}).get("node")
            for (x, y, w, h), step in zip(boxes["thumbs"], thumbs):
                thumb = images.get(step.get("image"), (w, h - 4))
                if thumb is not None:
                    screen.blit(thumb, (x + (w - thumb.get_width()) // 2, y))
                colour = FG if step.get("node") == current else DIM
                pygame.draw.rect(screen, colour, (x, y, w, h - 4), 2 if step.get("node") == current else 1)
            pygame.display.flip()
            shown = (state.get("current") or {}).get("image") if picture is not None else None
            if shots and shown and shown != shot_of:
                shot_of = shown
                n = len(list(os.scandir(shots))) + 1 if os.path.isdir(shots) else 1
                os.makedirs(shots, exist_ok=True)
                pygame.image.save(screen, os.path.join(shots, f"window-{n}.png"))
            clock.tick(10)
    finally:
        client.stop()
        pygame.quit()


if __name__ == "__main__":
    sys.exit(main())
