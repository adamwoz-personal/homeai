#!/usr/bin/env python3
"""Correct face landmarks by hand when detection misplaces them (AVATAR_PLAN §5 step 2).

Why: MediaPipe is trained on photos; on stylised art it can put a lip or lid
a few pixels off, and the rig then moves the wrong dots. This shows the crop
the build will use with its detected landmarks; drag any that are wrong and
save. The overrides file feeds ``build_face.py --overrides``.

Usage:
    .venv/bin/python tools/avatar/landmark_editor.py IMAGE [--out overrides.json]
Keys: drag = move the nearest landmark, R = reset all, Z = undo last drag,
S = save, Q/Esc = quit. Moved landmarks are drawn cyan, rigged ones yellow.
"""
from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

PICK_RADIUS = 12        # screen px
MOVED_TOL = 0.25        # image px; smaller moves are not saved
WINDOW = 900


@dataclass(frozen=True)
class Fit:
    """Image <-> screen mapping for an image scaled to fit a square window."""
    scale: float
    ox: float
    oy: float

    @classmethod
    def of(cls, image_w: int, image_h: int, window: int) -> "Fit":
        s = window / max(image_w, image_h)
        return cls(s, (window - image_w * s) / 2, (window - image_h * s) / 2)

    def to_screen(self, pts: np.ndarray) -> np.ndarray:
        return np.asarray(pts, dtype=np.float64) * self.scale + (self.ox, self.oy)

    def to_image(self, pts: np.ndarray) -> np.ndarray:
        return (np.asarray(pts, dtype=np.float64) - (self.ox, self.oy)) / self.scale


def pick(screen_pts: np.ndarray, at: tuple[float, float], radius: float = PICK_RADIUS) -> int | None:
    """Index of the nearest point within ``radius`` of ``at``, or None."""
    if not len(screen_pts):
        return None
    d = np.linalg.norm(screen_pts - np.asarray(at, dtype=np.float64), axis=1)
    i = int(np.argmin(d))
    return i if d[i] <= radius else None


def overrides(detected: np.ndarray, edited: np.ndarray, image_sha256: str, crop: dict) -> dict:
    """The file build_face.py reads: only landmarks moved more than MOVED_TOL."""
    moved = np.nonzero(np.linalg.norm(edited - detected, axis=1) > MOVED_TOL)[0]
    return {"image_sha256": image_sha256, "crop": crop,
            "landmarks": {str(int(i)): [round(float(v), 2) for v in edited[i]] for i in moved}}


def load_existing(path: Path, detected: np.ndarray, crop: dict) -> np.ndarray:
    """Start from a saved overrides file if it was made on the same crop."""
    from homeai.avatar import facebuild

    if not path.exists():
        return detected.copy()
    data = json.loads(path.read_text())
    if data.get("crop") != crop:
        print(f"{path}: made on another crop; starting from the detected landmarks")
        return detected.copy()
    return facebuild.apply_overrides(detected, data)


def run(image: Path, out: Path) -> int:  # pragma: no cover - interactive
    import pygame

    from homeai.avatar import facebuild, rig
    from homeai.avatar.assets import sha256

    crop, cut = facebuild.load_crop(image)
    detected = facebuild.detect_landmarks(crop)
    edited = load_existing(out, detected, cut)
    rigged = {i for idx in rig.REGIONS.values() for i in idx}
    fit = Fit.of(crop.shape[1], crop.shape[0], WINDOW)

    pygame.init()
    screen = pygame.display.set_mode((WINDOW, WINDOW))
    pygame.display.set_caption(f"landmarks: {image.name}  (drag, S save, R reset, Z undo, Q quit)")
    bg = pygame.transform.smoothscale(pygame.surfarray.make_surface(crop.swapaxes(0, 1)),
                                      (round(crop.shape[1] * fit.scale), round(crop.shape[0] * fit.scale)))
    held, undo, dirty = None, [], False
    clock = pygame.time.Clock()
    while True:
        for ev in pygame.event.get():
            if ev.type == pygame.QUIT or (ev.type == pygame.KEYDOWN and ev.key in (pygame.K_q, pygame.K_ESCAPE)):
                if dirty:
                    print("quit without saving the last changes")
                pygame.quit()
                return 0
            if ev.type == pygame.MOUSEBUTTONDOWN and ev.button == 1:
                held = pick(fit.to_screen(edited), ev.pos)
                if held is not None:
                    undo.append((held, edited[held].copy()))
            elif ev.type == pygame.MOUSEBUTTONUP:
                held = None
            elif ev.type == pygame.MOUSEMOTION and held is not None:
                edited[held] = np.clip(fit.to_image(ev.pos), 0, cut["side"])
                dirty = True
            elif ev.type == pygame.KEYDOWN and ev.key == pygame.K_z and undo:
                i, xy = undo.pop()
                edited[i] = xy
            elif ev.type == pygame.KEYDOWN and ev.key == pygame.K_r:
                edited, undo, dirty = detected.copy(), [], True
            elif ev.type == pygame.KEYDOWN and ev.key == pygame.K_s:
                data = overrides(detected, edited, sha256(image), cut)
                out.write_text(json.dumps(data, indent=1))
                print(f"saved {len(data['landmarks'])} moved landmarks -> {out}")
                dirty = False
        screen.fill((0, 0, 0))
        screen.blit(bg, (fit.ox, fit.oy))
        moved = np.linalg.norm(edited - detected, axis=1) > MOVED_TOL
        for i, (x, y) in enumerate(fit.to_screen(edited)):
            colour = (0, 255, 255) if moved[i] else (255, 220, 0) if i in rigged else (255, 60, 60)
            pygame.draw.circle(screen, colour, (round(x), round(y)), 4 if i == held else 2)
        pygame.display.flip()
        clock.tick(60)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("image", type=Path)
    parser.add_argument("--out", type=Path, help="default: <image stem>.landmarks.json next to the image")
    args = parser.parse_args()
    from homeai.avatar.facebuild import BuildError
    try:
        return run(args.image, args.out or args.image.with_suffix(".landmarks.json"))
    except BuildError as exc:
        print(f"error: {exc}")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
