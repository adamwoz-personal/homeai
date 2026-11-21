"""Pose forms: motion-capture frames as Ani's mannequin in `pose-spec/1` (ART_FEATURES_PLAN F10).

The contract (~/ani/pose-app/plans/HOMEAI_CONTRACT.md) is the format: rig ``mannequin-2d/2``, ``sides:
anatomical`` (L is the figure's own left), her 19 joint names, canvas 400 x 520, y down, z toward the viewer,
joints as drawn after ``turn``. A form keeps each bone's *direction* from the capture and Ani's bone *lengths*
(pose-core.js, Skeleton 2.0), so it is her mannequin doing the captured move.

Retargeting, per frame:
1. Face the viewer: the figure's left is its hip line (left hip - right hip, flattened); forward = left x up.
   Picture x = the figure's left, y = down, z = forward.
2. Rebuild her skeleton from the hips out with her lengths along the captured directions.
3. Stand it on her floor: the lowest ankle at GROUND - 6, hips over the middle of the canvas.
Then ``turn`` spins it about the vertical line through the hips, exactly as her ``findJoints(pose, turn)`` does.
"""
from __future__ import annotations

import math

import numpy as np

from homeai.poses import mocap as M

RIG, SCHEMA, SIDES = "mannequin-2d/2", "pose-spec/1", "anatomical"
CANVAS = {"width": 400, "height": 520}
GROUND, MIDDLE = 500, 200
# Her sizes (pose-core.js section 6 and findJoints).
THIGH = SHIN = 85
SPINE, BELLY, RIB_UP, RIB_SIDE = 120, 36, 56, 28
HIP_SIDE, SHOULDER_OUT, SHOULDER_DROP = 22, 46, 6
UPPER_ARM, FOREARM, NECK, HEAD = 62, 58, 18, 24
JOINTS = ("hips", "hipL", "kneeL", "ankleL", "hipR", "kneeR", "ankleR", "belly", "chestTop", "ribL", "ribR",
          "shoulderL", "elbowL", "handL", "shoulderR", "elbowR", "handR", "neckTop", "head")
LIMITS = {"x": (-400, 800), "y": (-520, 1040), "z": (-400, 400)}
CREDIT = "The data used in this project was obtained from mocap.cs.cmu.edu."


class FormError(ValueError):
    pass


def _unit(v: np.ndarray) -> np.ndarray:
    n = float(np.linalg.norm(v))
    if n < 1e-9:
        raise FormError("two captured points are on top of each other")
    return v / n


def from_capture(p: dict[str, np.ndarray]) -> dict[str, dict[str, float]]:
    """Her 19 joints (x, y, z on her canvas, facing the viewer) from one frame of CMU bone ends."""
    up = np.array([0.0, 1.0, 0.0])
    hip_line = p["lhipjoint"] - p["rhipjoint"]
    left = _unit(hip_line - up * float(hip_line @ up))
    forward = np.cross(left, up)                     # facing +z with up +y, the left is +x: forward = left x up
    frame = np.stack([left, -up, forward])           # rows: picture x, y (down), z (toward the viewer)

    def d(a: str, b: str) -> np.ndarray:             # captured direction from a to b, in picture axes
        return frame @ _unit(p[b] - p[a])

    j = {"hips": np.zeros(3)}
    hip_across = frame @ _unit(p["lhipjoint"] - p["rhipjoint"])      # her hips sit level with the pelvis
    for s, side, sign in (("L", "l", 1), ("R", "r", -1)):
        j["hip" + s] = j["hips"] + hip_across * sign * HIP_SIDE
        j["knee" + s] = j["hip" + s] + d(f"{side}hipjoint", f"{side}femur") * THIGH
        j["ankle" + s] = j["knee" + s] + d(f"{side}femur", f"{side}tibia") * SHIN
    spine = d("root", "thorax")
    j["chestTop"] = spine * SPINE
    j["belly"] = spine * BELLY
    across = frame @ _unit(p["lclavicle"] - p["rclavicle"])
    for s, side, sign in (("L", "l", 1), ("R", "r", -1)):
        j["rib" + s] = spine * RIB_UP + across * sign * RIB_SIDE
        # Her shoulder at rest: 46 out along the shoulder line and 6 down the spine from the top of the chest.
        j["shoulder" + s] = j["chestTop"] + across * sign * SHOULDER_OUT - spine * SHOULDER_DROP
        j["elbow" + s] = j["shoulder" + s] + d(f"{side}clavicle", f"{side}humerus") * UPPER_ARM
        j["hand" + s] = j["elbow" + s] + d(f"{side}humerus", f"{side}radius") * FOREARM
    j["neckTop"] = j["chestTop"] + d("thorax", "upperneck") * NECK
    j["head"] = j["neckTop"] + d("upperneck", "head") * HEAD
    lift = GROUND - 6 - max(j["ankleL"][1], j["ankleR"][1])
    return {name: {"x": float(v[0]) + MIDDLE, "y": float(v[1]) + lift, "z": float(v[2])} for name, v in j.items()}


def turned(joints: dict[str, dict[str, float]], turn: int) -> dict[str, dict[str, float]]:
    """The figure spun by ``turn`` degrees about the vertical through its hips: 90 faces the right of the picture."""
    check_turn(turn)
    a = math.radians(turn)
    cx, cz = joints["hips"]["x"], joints["hips"]["z"]
    out = {}
    for name, q in joints.items():
        dx, dz = q["x"] - cx, q["z"] - cz
        out[name] = {"x": cx + dx * math.cos(a) + dz * math.sin(a), "y": q["y"],
                     "z": cz - dx * math.sin(a) + dz * math.cos(a)}
    return out


def check_turn(turn) -> None:
    if not isinstance(turn, int) or isinstance(turn, bool) or not 0 <= turn <= 359:
        raise FormError("the turn must be a whole number from 0 to 359")


def view_of(turn: int) -> str:
    """The contract's rule: the nearest of 0, 45, ... 315; 0 front, 90 and 270 side, 180 back, else three-quarter."""
    check_turn(turn)
    nearest = round(turn / 45) % 8
    return {0: "front", 4: "back", 2: "side", 6: "side"}.get(nearest, "three-quarter")


def _round(v: float) -> float:
    return round(v, 2) + 0.0                         # + 0.0 turns -0.0 into 0.0


def to_spec(joints: dict[str, dict[str, float]], *, name: str, description: str, turn: int = 0,
            pose: dict | None = None, extra: dict | None = None) -> dict:
    """A pose-spec/1 for a form, in the contract's field order. ``joints`` face the viewer; the turn is applied here."""
    drawn = turned(joints, turn)
    spec = {"schema": SCHEMA, "rig": RIG, "name": name, "description": description, "view": view_of(turn),
            "turn": turn, "sides": SIDES, "canvas": dict(CANVAS)}
    if pose is not None:
        spec["pose"] = dict(pose)
    spec["joints"] = {n: {k: _round(drawn[n][k]) for k in ("x", "y", "z")} for n in JOINTS}
    if extra:
        spec["extra"] = dict(extra)
    check_spec(spec)
    return spec


def check_spec(spec: dict) -> None:
    """The contract's rules for what home accepts, for the fields a form uses. Anything wrong is a FormError."""
    for key in ("schema", "rig", "view", "sides", "canvas", "joints", "turn"):
        if key not in spec:
            raise FormError(f"a form needs {key}")
    if (spec["schema"], spec["rig"], spec["sides"]) != (SCHEMA, RIG, SIDES):
        raise FormError("a form is pose-spec/1, rig mannequin-2d/2, anatomical")
    if spec["view"] != view_of(spec["turn"]):
        raise FormError(f"view {spec['view']!r} does not match turn {spec['turn']}")
    if spec["canvas"] != CANVAS:
        raise FormError("the canvas is 400 x 520")
    if not isinstance(spec.get("name", ""), str) or len(spec.get("name", "")) > 80:
        raise FormError("a name is text of at most 80 letters")
    if not isinstance(spec.get("description", ""), str) or len(spec.get("description", "")) > 500:
        raise FormError("a description is text of at most 500 letters")
    joints = spec["joints"]
    if set(joints) != set(JOINTS):
        raise FormError(f"joints must be exactly her 19: missing {sorted(set(JOINTS) - set(joints))}, "
                        f"extra {sorted(set(joints) - set(JOINTS))}")
    for name, q in joints.items():
        for axis, (low, high) in LIMITS.items():
            v = q.get(axis)
            if not isinstance(v, (int, float)) or isinstance(v, bool) or not math.isfinite(v) or not low <= v <= high:
                raise FormError(f"{name}.{axis} = {v!r} is outside {low}..{high}")


def capture_frame(skeleton: M.Skeleton, frames: list, number: int) -> dict[str, np.ndarray]:
    """Bone ends for a frame by its 1-based number, as the AMC file counts."""
    if not 1 <= number <= len(frames):
        raise FormError(f"frame {number} is outside 1..{len(frames)}")
    return M.positions(skeleton, frames[number - 1])
