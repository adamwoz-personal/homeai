"""The pose probe must wire the reference picture in and line joints up with the fitted export."""
from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools" / "art"))


def load(name: str):
    spec = importlib.util.spec_from_file_location(name, ROOT / "tools" / "art" / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


probe = load("probe_pose_control")
fixture = load("check_pose_fixture")


def graph():
    return probe.klein_reference("u", "c", "pose.png", "p", 7, 800, 1040, "x")


def test_reference_reaches_both_sides_of_the_guider():
    g = graph()
    assert g["guider"]["inputs"]["positive"] == ["ref_pos", 0]
    assert g["guider"]["inputs"]["negative"] == ["ref_neg", 0]
    assert g["ref_pos"]["inputs"]["latent"] == ["encode", 0]
    assert g["ref_neg"]["inputs"]["latent"] == ["encode", 0]
    assert g["encode"]["inputs"]["pixels"] == ["load", 0]


def test_every_link_points_at_a_node():
    g = graph()
    for node in g.values():
        for value in node["inputs"].values():
            if isinstance(value, list):
                assert value[0] in g


def test_joints_follow_the_fit():
    spec = {"joints": {"handR": {"x": 409.87, "y": 100.0}}}
    fit = {"imageWidth": 800, "scale": 2, "offsetX": -96.36, "offsetY": -10.0}
    x, y = probe.joints_in_pixels(spec, fit, 800)["handR"]
    assert round(x, 2) == 723.38
    assert y == 190.0


def test_joints_shrink_with_the_picture():
    spec = {"joints": {"head": {"x": 100.0, "y": 100.0}}}
    fit = {"imageWidth": 800, "scale": 2, "offsetX": 0, "offsetY": 0}
    assert probe.joints_in_pixels(spec, fit, 400)["head"] == (100.0, 100.0)


def test_bones_name_real_joints():
    example = json.loads((ROOT / "tests/fixtures/pose-spec-example.json").read_text())
    assert {name for bone in probe.BONES for name in bone} <= set(example["joints"])


def test_fixture_same(tmp_path):
    a, b = tmp_path / "a", tmp_path / "b"
    a.write_text("x")
    b.write_text("x")
    assert fixture.compare(a, b)[0] == 0


def test_fixture_differs(tmp_path):
    a, b = tmp_path / "a", tmp_path / "b"
    a.write_text("x")
    b.write_text("y")
    assert fixture.compare(a, b)[0] == 1


def test_fixture_without_her_repo_is_skipped(tmp_path):
    a = tmp_path / "a"
    a.write_text("x")
    code, line = fixture.compare(a, tmp_path / "missing")
    assert code == 0 and "skipped" in line
