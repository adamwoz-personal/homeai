"""Draw Ani's wooden mannequin from joints alone (pose library, ART_FEATURES_PLAN F10).

Why: klein follows a mannequin picture given as a reference (pose probe 2026-11-06, REMOTE_APP_PLAN section 9),
and forms from motion capture can reach forward, which her dials can't make yet, so home draws them itself. The
look is her ``bodyShapes`` (pose-core.js, skeleton2-turn): wood-coloured limbs 16 wide with round ends, joint balls,
a chest and hips with thickness, feet pointing forward, an oval head and a small nose for the facing. Shapes are
painted back to front by depth (z toward the viewer). Several figures can share one picture (a scene).
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
from PIL import Image, ImageDraw

WOOD, JOINT_COLOUR, FLOOR, PAPER = (217, 179, 130), (169, 124, 80), (216, 201, 174), (255, 250, 240)
HALF_DEPTH = 14
SUPER = 2                                           # drawn at twice the size, then shrunk: smooth edges


@dataclass(frozen=True)
class Placed:
    """A figure in a picture: its joints (her canvas units), where its hips go across the picture (0..1), and a
    scale on top of the fit."""
    joints: dict
    x: float = 0.5
    scale: float = 1.0


def _v(q) -> np.ndarray:
    return np.array([q["x"], q["y"], q["z"]], dtype=float)


def _unit(v):
    n = float(np.linalg.norm(v))
    return v / n if n > 1e-9 else v * 0


def _facing(left, right, up):
    """Which way a body part faces: square to its left-right line and to up (pose-core.js ``facing``)."""
    return _unit(np.cross(up, _unit(left - right)))


def _hull(points: list[tuple[float, float]]) -> list[tuple[float, float]]:
    pts = sorted(set((round(x, 6), round(y, 6)) for x, y in points))
    if len(pts) < 3:
        return pts

    def turn(o, a, b):
        return (a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0])

    def half(seq):
        out = []
        for p in seq:
            while len(out) >= 2 and turn(out[-2], out[-1], p) <= 0:
                out.pop()
            out.append(p)
        return out[:-1]
    return half(pts) + half(pts[::-1])


def shapes(joints: dict) -> list[dict]:
    """Her shapes for one figure, in her canvas units, each {kind, depth, ...}, back to front."""
    j = {n: _v(q) for n, q in joints.items()}
    down = np.array([0.0, 1.0, 0.0])
    out = []

    def limb(a, b, depth=None):
        out.append({"kind": "line", "a": a, "b": b, "width": 16, "colour": WOOD,
                    "depth": (a[2] + b[2]) / 2 if depth is None else depth})

    def ball(p, r, colour=JOINT_COLOUR, bias=2):
        out.append({"kind": "ball", "c": p, "r": r, "colour": colour, "depth": p[2] + bias})

    hips_fwd = _facing(j["hipL"], j["hipR"], -down)
    across = _unit(j["hipL"] - j["hipR"])
    for s, side in (("L", 1), ("R", -1)):
        ankle = j["ankle" + s]
        limb(j["hip" + s], j["knee" + s])
        limb(j["knee" + s], ankle)
        limb(ankle, ankle + across * side * 16 + down * 2 + hips_fwd * 14, depth=ankle[2])
        ball(j["knee" + s], 7)
    corners = [j["hips"] + across * side * w + down * dy + hips_fwd * deep * HALF_DEPTH
               for w, dy in ((28, -20), (22, 6)) for side in (1, -1) for deep in (1, -1)]
    out.append({"kind": "block", "points": corners, "colour": WOOD, "depth": float(np.mean([c[2] for c in corners]))})
    ball(j["belly"], 13)
    chest_fwd = _facing(j["shoulderL"], j["shoulderR"], j["chestTop"] - j["hips"])
    chest = [p + chest_fwd * deep * HALF_DEPTH for p in (j["shoulderL"], j["shoulderR"], j["ribR"], j["ribL"])
             for deep in (1, -1)]
    out.append({"kind": "block", "points": chest, "colour": WOOD, "depth": float(np.mean([c[2] for c in chest]))})
    for s in ("L", "R"):
        limb(j["shoulder" + s], j["elbow" + s])
        limb(j["elbow" + s], j["hand" + s])
        ball(j["shoulder" + s], 8)
        ball(j["elbow" + s], 7)
        ball(j["hand" + s], 9)
    out.append({"kind": "line", "a": j["chestTop"], "b": j["neckTop"], "width": 11, "colour": JOINT_COLOUR,
                "depth": (j["chestTop"][2] + j["neckTop"][2]) / 2})
    tip = math.degrees(math.atan2(j["head"][0] - j["neckTop"][0], j["neckTop"][1] - j["head"][1]))
    out.append({"kind": "head", "c": j["head"], "angle": tip, "colour": WOOD, "depth": j["head"][2] + 2})
    ball(j["head"] + chest_fwd * 21, 5)
    order = sorted(range(len(out)), key=lambda i: (round(out[i]["depth"], 2), i))
    return [out[i] for i in order]


def _fit(placed: Placed, width: int, height: int) -> tuple[float, float, float]:
    """Scale and offset so the figure's 400 x 520 canvas fits the picture's height, hips at ``placed.x``."""
    s = height / 520 * placed.scale
    return s, placed.x * width - placed.joints["hips"]["x"] * s, height - 520 * s


def draw(figures: list[Placed], width: int = 400, height: int = 520, floor: bool = True) -> Image.Image:
    """One picture of the figures, nearest painted last across all of them."""
    big = Image.new("RGB", (width * SUPER, height * SUPER), PAPER)
    pen = ImageDraw.Draw(big)
    if floor:
        y = (height - 20 * height / 520) * SUPER
        pen.line([(30 * SUPER * width / 400, y), (width * SUPER - 30 * SUPER * width / 400, y)], fill=FLOOR,
                 width=max(1, round(3 * SUPER * height / 520)))
    queue = []
    for n, placed in enumerate(figures):
        s, ox, oy = _fit(placed, width, height)
        for shape in shapes(placed.joints):
            queue.append((round(shape["depth"], 2), n, shape, s, ox, oy))
    queue.sort(key=lambda item: (item[0], item[1]))
    for _, _, shape, s, ox, oy in queue:
        k = s * SUPER

        def at(p):
            return ((p[0] * s + ox) * SUPER, (p[1] * s + oy) * SUPER)
        if shape["kind"] == "line":
            a, b, w = at(shape["a"]), at(shape["b"]), shape["width"] * k
            pen.line([a, b], fill=shape["colour"], width=max(1, round(w)))
            for c in (a, b):
                pen.ellipse([c[0] - w / 2, c[1] - w / 2, c[0] + w / 2, c[1] + w / 2], fill=shape["colour"])
        elif shape["kind"] == "ball":
            c, r = at(shape["c"]), shape["r"] * k
            pen.ellipse([c[0] - r, c[1] - r, c[0] + r, c[1] + r], fill=shape["colour"])
        elif shape["kind"] == "block":
            outline = _hull([at(p) for p in shape["points"]])
            if len(outline) >= 3:
                pen.polygon(outline, fill=shape["colour"])
                pen.line(outline + outline[:1], fill=shape["colour"], width=max(1, round(12 * k)), joint="curve")
        else:                                            # the head: an oval turned with the neck
            c, a = at(shape["c"]), math.radians(shape["angle"])
            ring = [(c[0] + 20 * k * math.cos(t) * math.cos(a) - 26 * k * math.sin(t) * math.sin(a),
                     c[1] + 20 * k * math.cos(t) * math.sin(a) + 26 * k * math.sin(t) * math.cos(a))
                    for t in np.linspace(0, 2 * math.pi, 48, endpoint=False)]
            pen.polygon(ring, fill=shape["colour"])
    return big.resize((width, height), Image.LANCZOS)
