"""Read CMU motion capture (ASF skeleton + AMC motion) into joint positions (pose library, ART_FEATURES_PLAN F10).

An ASF file describes the bones: each has a direction and length in the rest pose, an ``axis`` (its local frame,
Euler XYZ in degrees) and the rotations it may take (``dof``). An AMC file gives, per frame, the root's position
and every bone's rotation. Forward kinematics, as in the common AMC readers:

    C    = euler(axis)                              (the bone's frame)
    R    = R_parent @ C @ euler(rotation) @ C.T     (the bone's rotation in the world)
    end  = start + length * R @ direction

with euler(x, y, z) = Rz @ Ry @ Rx (static X, then Y, then Z). Positions come out in the ASF's length units, Y up.
Pure maths on parsed text: no file reading here beyond ``load``.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np


class MocapError(ValueError):
    pass


def euler(x: float, y: float, z: float) -> np.ndarray:
    """Rotation for angles in degrees about X, then Y, then Z (fixed axes)."""
    a, b, c = np.radians([x, y, z])
    rx = np.array([[1, 0, 0], [0, np.cos(a), -np.sin(a)], [0, np.sin(a), np.cos(a)]])
    ry = np.array([[np.cos(b), 0, np.sin(b)], [0, 1, 0], [-np.sin(b), 0, np.cos(b)]])
    rz = np.array([[np.cos(c), -np.sin(c), 0], [np.sin(c), np.cos(c), 0], [0, 0, 1]])
    return rz @ ry @ rx


@dataclass
class Bone:
    name: str
    direction: np.ndarray = field(default_factory=lambda: np.zeros(3))
    length: float = 0.0
    axis: np.ndarray = field(default_factory=lambda: np.eye(3))
    dof: tuple[str, ...] = ()
    children: list[str] = field(default_factory=list)


@dataclass
class Skeleton:
    bones: dict[str, Bone]
    root_order: tuple[str, ...]
    root_axis: np.ndarray


def _sections(text: str) -> dict[str, list[str]]:
    out, name = {}, None
    for raw in text.splitlines():
        line = raw.split("#", 1)[0].strip()
        if not line:
            continue
        if line.startswith(":"):
            name, _, rest = line[1:].partition(" ")
            out[name] = [rest] if rest else []
        elif name is not None:
            out[name].append(line)
    return out


def parse_asf(text: str) -> Skeleton:
    parts = _sections(text)
    if "bonedata" not in parts or "hierarchy" not in parts:
        raise MocapError("not an ASF skeleton (no bonedata or hierarchy)")
    root_order, root_axis = ("TX", "TY", "TZ", "RX", "RY", "RZ"), np.eye(3)
    for line in parts.get("root", []):
        key, *vals = line.split()
        if key == "order":
            root_order = tuple(v.upper() for v in vals)
        elif key == "orientation":
            root_axis = euler(*map(float, vals))
    bones, bone = {"root": Bone("root")}, None
    for line in parts["bonedata"]:
        key, *vals = line.split()
        if key == "begin":
            bone = Bone("")
        elif key == "end":
            if bone is None or not bone.name:
                raise MocapError("a bone without a name")
            bones[bone.name], bone = bone, None
        elif bone is None:
            continue
        elif key == "name":
            bone.name = vals[0]
        elif key == "direction":
            bone.direction = np.array(list(map(float, vals)))
        elif key == "length":
            bone.length = float(vals[0])
        elif key == "axis":
            bone.axis = euler(*map(float, vals[:3]))
        elif key == "dof":
            bone.dof = tuple(v.lower() for v in vals)
    for line in parts["hierarchy"]:
        if line in ("begin", "end"):
            continue
        parent, *kids = line.split()
        if parent not in bones or any(k not in bones for k in kids):
            raise MocapError(f"hierarchy names an unknown bone: {line!r}")
        bones[parent].children.extend(kids)
    return Skeleton(bones, root_order, root_axis)


def parse_amc(text: str) -> list[dict[str, list[float]]]:
    """Frames, each {bone: [values in the order the file gives them]}."""
    frames, frame = [], None
    for raw in text.splitlines():
        line = raw.strip()
        if not line or line.startswith(("#", ":")):
            continue
        if line.isdigit():
            frame = {}
            frames.append(frame)
            continue
        if frame is None:
            raise MocapError("motion values before the first frame number")
        name, *vals = line.split()
        frame[name] = [float(v) for v in vals]
    if not frames:
        raise MocapError("no frames")
    return frames


def positions(skeleton: Skeleton, frame: dict[str, list[float]]) -> dict[str, np.ndarray]:
    """Where every bone ends in one frame (the root: its position). Keys: bone names, plus "root"."""
    values = dict(zip(skeleton.root_order, frame.get("root", [0.0] * 6)))
    root_rot = skeleton.root_axis @ euler(values.get("RX", 0.0), values.get("RY", 0.0), values.get("RZ", 0.0)) \
        @ skeleton.root_axis.T
    out = {"root": np.array([values.get("TX", 0.0), values.get("TY", 0.0), values.get("TZ", 0.0)])}
    stack = [(kid, root_rot, out["root"]) for kid in skeleton.bones["root"].children]
    while stack:
        name, parent_rot, start = stack.pop()
        bone = skeleton.bones[name]
        angles = dict(zip(bone.dof, frame.get(name, [])))
        local = euler(angles.get("rx", 0.0), angles.get("ry", 0.0), angles.get("rz", 0.0))
        rot = parent_rot @ bone.axis @ local @ bone.axis.T
        end = start + bone.length * rot @ bone.direction
        out[name] = end
        stack.extend((kid, rot, end) for kid in bone.children)
    return out


def load(asf: Path, amc: Path) -> tuple[Skeleton, list[dict[str, list[float]]]]:
    return parse_asf(Path(asf).read_text()), parse_amc(Path(amc).read_text())
