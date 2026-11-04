#!/usr/bin/env python3
"""Build the avatar face from art (AVATAR_PLAN §5, Phase 3).

Why: the face is a deterministic build (art -> landmarks -> mesh -> stipple ->
binding) so it can be rebuilt, compared and rolled back. Two uses:

1. Compare candidates (scratch builds + a contact sheet for Adam; nothing is
   installed): source | mesh and landmarks | still | jaw open | blink + brows.
2. Build a version into ~/.local/share/homeai/avatar/vN (never overwritten),
   optionally making it current.

Usage:
    .venv/bin/python tools/avatar/build_face.py --candidates [IMAGE ...]
    .venv/bin/python tools/avatar/build_face.py IMAGE [--version v1] [--use] \
        [--overrides landmarks.json] [--points 14000] [--seed 7]
Needs the avatar-build extra. Exit 0 ok, 1 a build failed, 2 setup error.
"""
from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

from homeai.avatar import assets, facebuild  # noqa: E402

BENCH_DIR = Path.home() / "homeai-bench"
CANDIDATES = sorted((Path.home() / "snap/firefox/common/Downloads").glob("image_*.jpg"))
TILE = 360
POSES = (("still", {}), ("jaw open", {"jaw_open": 1.0}),
         ("blink + brows", {"blink_l": 1.0, "blink_r": 1.0, "brow_raise": 1.0}))


def contact_row(out_dir: Path) -> "np.ndarray":  # noqa: F821
    """One row of tiles for a built directory."""
    import cv2
    import numpy as np
    from PIL import Image

    a = assets.load(out_dir)
    side = a.image_size[0]
    src = np.asarray(Image.open(out_dir / "source.png").convert("RGB"))
    tiles = [("source", src), ("mesh", facebuild.draw_mesh(src, a.verts, a.triangles))]
    for name, params in POSES:
        img = facebuild.render_points(a.points_for(params), a.size, a.colour, side)
        Image.fromarray(img).save(out_dir / f"pose_{name.replace(' + ', '_').replace(' ', '_')}.png")
        tiles.append((name, img))
    row = []
    for name, img in tiles:
        tile = cv2.resize(img, (TILE, TILE), interpolation=cv2.INTER_AREA)
        cv2.putText(tile, name, (8, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1, cv2.LINE_AA)
        row.append(tile)
    strip = np.hstack(row)
    cv2.putText(strip, out_dir.name, (8, TILE - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 1,
                cv2.LINE_AA)
    return strip


def export_landmarks(build_dir: Path, out: Path) -> None:
    """Save a build's 478 crop landmarks as a test fixture (tests/fixtures/avatar/)."""
    a = assets.load(build_dir)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({"source": a.build["inputs"]["image"], "image_size": list(a.image_size),
                               "landmarks": a.verts[:facebuild.rig.N_FACE].round(2).tolist()}))


def candidates(images: list[Path], params: dict) -> int:
    import numpy as np
    from PIL import Image

    out = BENCH_DIR / f"face-candidates-{datetime.now():%Y%m%d-%H%M%S}"
    rows, failed = [], 0
    for image in images:
        target = out / image.stem
        try:
            manifest = facebuild.build(image, target, **params)
        except (facebuild.BuildError, assets.AssetError) as exc:
            print(json.dumps({"image": image.name, "error": str(exc)}))
            failed += 1
            continue
        print(json.dumps({"image": image.name, **manifest["counts"], "seconds": manifest["seconds"]}))
        rows.append(contact_row(target))
    if rows:
        sheet = out / "contact_sheet.png"
        Image.fromarray(np.vstack(rows)).save(sheet)
        print(f"-> {sheet}")
    return 1 if failed else 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("images", nargs="*", type=Path)
    parser.add_argument("--candidates", action="store_true")
    parser.add_argument("--version", help="default: the next free vN")
    parser.add_argument("--use", action="store_true", help="make the new version current")
    parser.add_argument("--overrides", type=Path, help="landmark_editor.py output")
    parser.add_argument("--export-landmarks", nargs=2, type=Path, metavar=("BUILD_DIR", "OUT_JSON"),
                        help="write a build's landmarks as a test fixture and exit")
    for key, value in facebuild.DEFAULTS.items():
        parser.add_argument(f"--{key.replace('_', '-')}", type=type(value), default=value)
    args = parser.parse_args()
    params = {k: getattr(args, k) for k in facebuild.DEFAULTS}
    if args.export_landmarks:
        try:
            export_landmarks(*args.export_landmarks)
        except assets.AssetError as exc:
            print(f"export failed: {exc}")
            return 1
        print(f"-> {args.export_landmarks[1]}")
        return 0
    try:
        import cv2  # noqa: F401
        import mediapipe  # noqa: F401
        import scipy  # noqa: F401
    except ImportError as exc:
        print(f"setup error: {exc} (uv pip install --python .venv/bin/python -e '.[avatar-build]')")
        return 2

    if args.candidates:
        return candidates(args.images or CANDIDATES, params)
    if len(args.images) != 1:
        parser.error("give one IMAGE (or --candidates)")
    overrides = None
    if args.overrides:
        overrides = json.loads(args.overrides.read_text())
    root = assets.asset_root()
    version = args.version or assets.next_version(root)
    try:
        manifest = facebuild.build(args.images[0], root / version, overrides=overrides, **params)
        assets.load(root / version)
        print(json.dumps({"version": version, **manifest["counts"], "seconds": manifest["seconds"]}))
        if args.use:
            before = assets.use(root, version)
            print(f"current: {version} (was {before or 'none'})")
    except (facebuild.BuildError, assets.AssetError) as exc:
        print(f"build failed: {exc}")
        return 1
    print(f"-> {root / version}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
