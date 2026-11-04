"""Built face assets: load, validate, list, switch versions (AVATAR_PLAN §5 step 7).

Pure numpy (the window loads faces). A version is a directory
``<root>/vN/`` written by ``tools/avatar/build_face.py``:

- ``rig.json``: image size, vertices (landmarks + boundary ring), triangles,
  regions, displacement fields
- ``points.npz``: xy, size, colour, tri, bary per stipple point
- ``preview.png``, ``build.json`` (inputs, seed, tool versions, sha256 of each file)

``<root>/current`` is a symlink to the version in use, ``<root>/previous`` to
the one before; both are swapped with ``os.replace`` so a reader never sees a
half-switched state. Usage: ``homeai-avatar list|current|check [vN]|use vN|rollback``.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from . import rig

FORMAT = 1
DEFAULT_ROOT = Path.home() / ".local" / "share" / "homeai" / "avatar"
HASHED = ("rig.json", "points.npz")
POINT_KEYS = ("xy", "size", "colour", "tri", "bary")
_VERSION = re.compile(r"^v(\d+)$")


class AssetError(ValueError):
    pass


def asset_root() -> Path:
    return Path(os.environ.get("HOMEAI_AVATAR_ASSETS") or DEFAULT_ROOT)


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for block in iter(lambda: fh.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


@dataclass
class FaceAsset:
    version: str
    image_size: tuple[int, int]
    verts: np.ndarray            # (V, 2) rest positions
    triangles: np.ndarray        # (T, 3) int32
    regions: dict[str, list[int]]
    fields: dict[str, np.ndarray]
    xy: np.ndarray               # (N, 2) rest point positions
    size: np.ndarray             # (N,)
    colour: np.ndarray           # (N, 3) 0-1
    tri: np.ndarray              # (N,) triangle per point
    bary: np.ndarray             # (N, 3)
    build: dict

    @property
    def face_height(self) -> float:
        return rig.face_height(self.verts)

    def points_for(self, params: dict[str, float]) -> np.ndarray:
        return rig.positions(rig.deform(self.verts, self.fields, params), self.triangles,
                             self.tri, self.bary)


def validate(asset: FaceAsset) -> None:
    """Raise AssetError if the parts do not fit together."""
    v, t, n = len(asset.verts), len(asset.triangles), len(asset.xy)
    try:
        rig.validate_regions(asset.regions, v)
        rig.validate_fields(asset.fields, v, asset.face_height)
    except rig.RigError as exc:
        raise AssetError(str(exc)) from exc
    if v < rig.N_FACE or asset.verts.shape != (v, 2):
        raise AssetError(f"vertices: shape {asset.verts.shape}")
    if t == 0 or asset.triangles.shape != (t, 3) or asset.triangles.min() < 0 or asset.triangles.max() >= v:
        raise AssetError("triangles: bad shape or vertex index")
    shapes = {"size": (n,), "colour": (n, 3), "tri": (n,), "bary": (n, 3)}
    for key, shape in shapes.items():
        if getattr(asset, key).shape != shape:
            raise AssetError(f"{key}: shape {getattr(asset, key).shape}, want {shape}")
    if n == 0:
        raise AssetError("no points")
    if asset.tri.min() < 0 or asset.tri.max() >= t:
        raise AssetError("points bound to a triangle that does not exist")
    if not np.allclose(asset.bary.sum(axis=1), 1.0, atol=1e-3):
        raise AssetError("barycentric weights do not sum to 1")
    if not all(np.isfinite(a).all() for a in (asset.verts, asset.xy, asset.size, asset.colour, asset.bary)):
        raise AssetError("non-finite values")


def load(path: Path, verify_hashes: bool = True) -> FaceAsset:
    """Load and validate one version directory."""
    path = Path(path)
    try:
        build = json.loads((path / "build.json").read_text())
        rig_data = json.loads((path / "rig.json").read_text())
        with np.load(path / "points.npz") as npz:
            pts = {k: npz[k] for k in POINT_KEYS}
    except (OSError, ValueError, KeyError) as exc:
        raise AssetError(f"{path}: unreadable ({exc})") from exc
    if build.get("format") != FORMAT or rig_data.get("format") != FORMAT:
        raise AssetError(f"{path}: format {build.get('format')}/{rig_data.get('format')}, want {FORMAT}")
    if verify_hashes:
        for name in HASHED:
            want = (build.get("files") or {}).get(name)
            if want != sha256(path / name):
                raise AssetError(f"{path / name}: does not match build.json (edited after the build?)")
    try:
        asset = FaceAsset(
            version=path.name, image_size=tuple(rig_data["image_size"]),
            verts=np.asarray(rig_data["vertices"], dtype=np.float32),
            triangles=np.asarray(rig_data["triangles"], dtype=np.int32),
            regions={k: list(v) for k, v in rig_data["regions"].items()},
            fields={k: np.asarray(v, dtype=np.float32) for k, v in rig_data["fields"].items()},
            xy=pts["xy"].astype(np.float32), size=pts["size"].astype(np.float32),
            colour=pts["colour"].astype(np.float32), tri=pts["tri"].astype(np.int32),
            bary=pts["bary"].astype(np.float32), build=build)
    except (KeyError, TypeError, ValueError) as exc:
        raise AssetError(f"{path}: rig.json incomplete ({exc})") from exc
    validate(asset)
    return asset


def versions(root: Path) -> list[str]:
    if not root.is_dir():
        return []
    found = [p.name for p in root.iterdir() if p.is_dir() and not p.is_symlink() and _VERSION.match(p.name)]
    return sorted(found, key=lambda name: int(name[1:]))


def next_version(root: Path) -> str:
    have = versions(root)
    return f"v{int(have[-1][1:]) + 1}" if have else "v1"


def _link(root: Path, name: str) -> str | None:
    try:
        return os.readlink(root / name)
    except OSError:
        return None


def current(root: Path) -> str | None:
    return _link(root, "current")


def _swap(root: Path, name: str, target: str) -> None:
    tmp = root / f".{name}.{os.getpid()}.tmp"
    tmp.unlink(missing_ok=True)
    os.symlink(target, tmp)
    os.replace(tmp, root / name)            # atomic: readers see old or new, never neither


def use(root: Path, version: str) -> str | None:
    """Make ``version`` current (it must load cleanly); returns the version it replaced."""
    if not _VERSION.match(version) or not (root / version).is_dir():
        raise AssetError(f"no such version: {version} (have: {', '.join(versions(root)) or 'none'})")
    load(root / version)
    before = current(root)
    if before == version:
        return before
    if before:
        _swap(root, "previous", before)
    _swap(root, "current", version)
    return before


def rollback(root: Path) -> str:
    """Switch back to ``previous``; returns the version now current."""
    prev = _link(root, "previous")
    if not prev:
        raise AssetError("nothing to roll back to")
    use(root, prev)
    return prev


def load_current(root: Path | None = None) -> FaceAsset | None:
    """The face in use, or None when none has been built or chosen."""
    root = root or asset_root()
    name = current(root)
    return load(root / name) if name else None


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="homeai-avatar", description="Manage built avatar faces")
    sub = parser.add_subparsers(dest="cmd", required=True)
    sub.add_parser("list")
    sub.add_parser("current")
    p = sub.add_parser("check")
    p.add_argument("version", nargs="?")
    p = sub.add_parser("use")
    p.add_argument("version")
    sub.add_parser("rollback")
    args = parser.parse_args(argv)
    root = asset_root()
    try:
        if args.cmd == "list":
            cur = current(root)
            for v in versions(root):
                print(f"{'*' if v == cur else ' '} {v}")
            if not versions(root):
                print(f"no faces built in {root}")
        elif args.cmd == "current":
            print(current(root) or "none")
        elif args.cmd == "check":
            name = args.version or current(root)
            if not name:
                print("no version given and none current")
                return 1
            a = load(root / name)
            print(f"{name}: OK ({len(a.xy)} points, {len(a.triangles)} triangles)")
        elif args.cmd == "use":
            before = use(root, args.version)
            print(f"current: {args.version} (was {before or 'none'}); restart the window: "
                  "systemctl --user restart homeai-avatar")
        elif args.cmd == "rollback":
            print(f"current: {rollback(root)}")
    except AssetError as exc:
        print(f"error: {exc}")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
