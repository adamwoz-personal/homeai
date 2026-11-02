#!/usr/bin/env python3
# Copyright (c) 2026 Adam Wosotowsky
# SPDX-License-Identifier: BUSL-1.1
"""Avatar Phase 0 spike: does MediaPipe find face landmarks on the avatar art?

Why: the asset pipeline (AVATAR_PLAN §5) binds stippled points to a face mesh.
If MediaPipe's FaceLandmarker finds the 478 landmarks on stylised pointillist
art, the mesh is automatic; if not, the landmark editor is required.

Pointillist dots can hide edges, so each image is tried as-is and with
preprocessing (Gaussian blur to merge dots, contrast boost, downscale), with
low confidence floors. The first variant that finds a face wins. Overlay PNGs
(landmarks drawn) go to ~/homeai-bench/landmarks-<stamp>/ for a human to
judge whether the points sit on the eyes, lips and jaw.

Usage:
    .venv/bin/python tools/avatar/landmarks_probe.py --download      # fetch the model once
    .venv/bin/python tools/avatar/landmarks_probe.py [IMAGE ...]     # default: Adam's candidates
Needs the avatar-build extra (mediapipe, opencv-python-headless).
Exit 0 all images found, 1 some not found, 2 setup error.
"""

from __future__ import annotations

import argparse
import json
import sys
import urllib.request
from datetime import datetime
from pathlib import Path

MODEL_URL = ("https://storage.googleapis.com/mediapipe-models/face_landmarker/"
             "face_landmarker/float16/1/face_landmarker.task")
MODEL_PATH = Path.home() / ".local/share/homeai/models/face_landmarker.task"
CANDIDATES = sorted((Path.home() / "snap/firefox/common/Downloads").glob("image_*.jpg"))
BENCH_DIR = Path.home() / "homeai-bench"
# Landmark indices for a quick plausibility check: eyes above the mouth.
LEFT_EYE, RIGHT_EYE, UPPER_LIP, LOWER_LIP, CHIN = 33, 263, 13, 14, 152


def variants(img, cv2):
    """(name, image) pairs to try, cheapest first."""
    yield "raw", img
    for k in (9, 21):
        yield f"blur{k}", cv2.GaussianBlur(img, (k, k), 0)
    lab = cv2.cvtColor(cv2.GaussianBlur(img, (9, 9), 0), cv2.COLOR_BGR2LAB)
    lab[..., 0] = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8)).apply(lab[..., 0])
    yield "blur9+clahe", cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)
    gray = cv2.cvtColor(cv2.GaussianBlur(img, (15, 15), 0), cv2.COLOR_BGR2GRAY)
    yield "blur15+gray", cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
    h, w = img.shape[:2]
    yield "half+blur5", cv2.GaussianBlur(cv2.resize(img, (w // 2, h // 2)), (5, 5), 0)


def bbox(points: list[tuple[float, float]]) -> list[float]:
    """Normalised [x0, y0, x1, y1] of landmark points."""
    xs = [p[0] for p in points]
    ys = [p[1] for p in points]
    return [round(min(xs), 3), round(min(ys), 3), round(max(xs), 3), round(max(ys), 3)]


def plausible(points: list[tuple[float, float]]) -> bool:
    """Eyes above lips above chin, eyes left/right apart: a face, not noise."""
    if len(points) <= CHIN:
        return False
    eye_y = max(points[LEFT_EYE][1], points[RIGHT_EYE][1])
    return (eye_y < points[UPPER_LIP][1] < points[LOWER_LIP][1] < points[CHIN][1]
            and abs(points[LEFT_EYE][0] - points[RIGHT_EYE][0]) > 0.02)


def download() -> int:
    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    tmp = MODEL_PATH.with_suffix(".part")
    try:
        urllib.request.urlretrieve(MODEL_URL, tmp)
    except OSError as exc:
        print(f"download failed: {exc}")
        return 2
    tmp.replace(MODEL_PATH)
    print(f"model saved: {MODEL_PATH} ({MODEL_PATH.stat().st_size // 1024} KB)")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[1])
    parser.add_argument("images", nargs="*", type=Path)
    parser.add_argument("--download", action="store_true")
    parser.add_argument("--min-confidence", type=float, default=0.1)
    args = parser.parse_args()
    if args.download:
        return download()

    try:
        import cv2
        import mediapipe as mp
        from mediapipe.tasks.python import BaseOptions
        from mediapipe.tasks.python.vision import FaceLandmarker, FaceLandmarkerOptions
    except ImportError as exc:
        print(f"setup error: {exc} (install the avatar-build extra)")
        return 2
    if not MODEL_PATH.exists():
        print(f"setup error: {MODEL_PATH} missing; run with --download")
        return 2
    images = args.images or CANDIDATES
    if not images:
        print("setup error: no images given and no candidates found")
        return 2

    floor = args.min_confidence
    options = FaceLandmarkerOptions(
        base_options=BaseOptions(model_asset_path=str(MODEL_PATH)),
        num_faces=1, min_face_detection_confidence=floor,
        min_face_presence_confidence=floor, output_face_blendshapes=True)
    out_dir = BENCH_DIR / f"landmarks-{datetime.now():%Y%m%d-%H%M%S}"
    out_dir.mkdir(parents=True, exist_ok=True)

    results = []
    with FaceLandmarker.create_from_options(options) as landmarker:
        for path in images:
            img = cv2.imread(str(path))
            if img is None:
                results.append({"image": path.name, "error": "unreadable"})
                continue
            row = {"image": path.name, "size": [img.shape[1], img.shape[0]], "tried": []}
            for name, variant in variants(img, cv2):
                rgb = cv2.cvtColor(variant, cv2.COLOR_BGR2RGB)
                found = landmarker.detect(mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb))
                row["tried"].append(name)
                if not found.face_landmarks:
                    continue
                pts = [(p.x, p.y) for p in found.face_landmarks[0]]
                row.update(variant=name, landmarks=len(pts), bbox=bbox(pts),
                           plausible=plausible(pts))
                h, w = img.shape[:2]
                overlay = img.copy()
                for x, y in pts:
                    cv2.circle(overlay, (int(x * w), int(y * h)), 2, (0, 0, 255), -1)
                cv2.imwrite(str(out_dir / f"{path.stem}-{name}.png"), overlay)
                break
            results.append(row)

    (out_dir / "results.json").write_text(json.dumps(results, indent=2))
    found_all = True
    for r in results:
        if "landmarks" in r:
            print(f"{r['image']:22} FOUND via {r['variant']:12} {r['landmarks']} pts"
                  f"  bbox {r['bbox']}  plausible={r['plausible']}")
        else:
            found_all = False
            print(f"{r['image']:22} NOT FOUND ({r.get('error') or ', '.join(r['tried'])})")
    print(f"overlays + results.json -> {out_dir}")
    return 0 if found_all else 1


if __name__ == "__main__":
    sys.exit(main())
