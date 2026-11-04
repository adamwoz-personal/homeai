"""Weighted Voronoi stippling of the face art (AVATAR_PLAN §5 step 5). Build time only (scipy).

The art is already pointillist; blurring merges its dots into tone (as for
landmark detection) and the stippler re-places points by that tone: bright
areas get dense points, dark panel seams and background get none. Weighted
Lloyd relaxation (Secord 2002) spreads them evenly. Deterministic for a seed.
"""
from __future__ import annotations

import numpy as np

LUMA = np.array([0.299, 0.587, 0.114])
SMALL, LARGE = 0.6, 1.5         # point size factors, as the sphere's (gl.point_attributes)


def luminance(rgb: np.ndarray) -> np.ndarray:
    """(h, w) 0-1 from uint8 RGB."""
    if rgb.ndim != 3 or rgb.shape[2] != 3:
        raise ValueError(f"want an (h, w, 3) image, got {rgb.shape}")
    return (rgb.astype(np.float64) @ LUMA) / 255.0


def density_map(rgb: np.ndarray, blur_sigma: float = 2.0, gamma: float = 1.0,
                floor: float = 0.06) -> np.ndarray:
    """(h, w) float32 0-1: blurred luminance normalised to its 99.5th percentile; below ``floor`` is 0."""
    from scipy.ndimage import gaussian_filter

    lum = gaussian_filter(luminance(rgb), blur_sigma) if blur_sigma > 0 else luminance(rgb)
    top = float(np.percentile(lum, 99.5))
    if top <= 0:
        return np.zeros(lum.shape, dtype=np.float32)
    d = np.clip(lum / top, 0, 1) ** gamma
    d[d < floor] = 0.0
    return d.astype(np.float32)


def seed_points(density: np.ndarray, n: int, rng: np.random.Generator) -> np.ndarray:
    """(n, 2) x, y pixel coordinates drawn in proportion to density."""
    flat = density.ravel().astype(np.float64)
    total = flat.sum()
    if n <= 0 or total <= 0:
        raise ValueError("need n > 0 and some non-zero density")
    nonzero = int((flat > 0).sum())
    idx = rng.choice(flat.size, size=n, replace=n > nonzero, p=flat / total)
    ys, xs = np.divmod(idx, density.shape[1])
    return np.column_stack([xs, ys]) + rng.random((n, 2))


def relax(points: np.ndarray, density: np.ndarray, iterations: int, upsample: int = 2) -> np.ndarray:
    """Weighted Lloyd: move each point to the density-weighted centroid of its Voronoi cell.

    The cells are measured on a grid ``upsample`` times finer than the image, so
    small cells still have enough samples for a smooth centroid.
    """
    from scipy.ndimage import zoom
    from scipy.spatial import cKDTree

    pts = np.array(points, dtype=np.float64)
    fine = zoom(density.astype(np.float64), upsample, order=1) if upsample > 1 else density
    ys, xs = np.nonzero(fine > 0)
    w = fine[ys, xs]
    cx, cy = (xs + 0.5) / upsample, (ys + 0.5) / upsample
    sample = np.column_stack([cx, cy])
    n = len(pts)
    for _ in range(iterations):
        _, owner = cKDTree(pts).query(sample, workers=-1)
        mass = np.bincount(owner, w, n)
        moved = mass > 0
        pts[moved, 0] = np.bincount(owner, w * cx, n)[moved] / mass[moved]
        pts[moved, 1] = np.bincount(owner, w * cy, n)[moved] / mass[moved]
    return pts


def stipple(density: np.ndarray, n: int, seed: int, iterations: int = 30, upsample: int = 2) -> np.ndarray:
    """(n, 2) float64 point positions; the same inputs give the same points."""
    rng = np.random.default_rng(seed)
    return relax(seed_points(density, n, rng), density, iterations, upsample)


def sample(image: np.ndarray, pts: np.ndarray) -> np.ndarray:
    """Bilinear sample of an (h, w) or (h, w, c) array at pixel coordinates (centres at +0.5)."""
    img = image if image.ndim == 3 else image[..., None]
    h, w = img.shape[:2]
    x = np.clip(pts[:, 0] - 0.5, 0, w - 1)
    y = np.clip(pts[:, 1] - 0.5, 0, h - 1)
    x0, y0 = np.floor(x).astype(int), np.floor(y).astype(int)
    x1, y1 = np.minimum(x0 + 1, w - 1), np.minimum(y0 + 1, h - 1)
    fx, fy = (x - x0)[:, None], (y - y0)[:, None]
    out = (img[y0, x0] * (1 - fx) * (1 - fy) + img[y0, x1] * fx * (1 - fy)
           + img[y1, x0] * (1 - fx) * fy + img[y1, x1] * fx * fy)
    return out if image.ndim == 3 else out[:, 0]


def sizes(lum: np.ndarray) -> np.ndarray:
    """Size factor per point: darker points smaller (SMALL), the brightest LARGE."""
    top = float(lum.max()) if lum.size else 0.0
    t = np.clip(lum / top, 0, 1) if top > 0 else np.zeros_like(lum)
    return (SMALL + (LARGE - SMALL) * t).astype(np.float32)
