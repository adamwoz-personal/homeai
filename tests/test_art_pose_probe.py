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


def joints(**moved):
    base = {"head": {"x": 200, "y": 160}, "neckTop": {"x": 200, "y": 195}, "shoulderL": {"x": 160, "y": 215},
            "shoulderR": {"x": 240, "y": 215}, "handL": {"x": 150, "y": 330}, "handR": {"x": 250, "y": 330}, "hips": {"x": 200, "y": 325},
            "chestTop": {"x": 200, "y": 200},
            "elbowL": {"x": 155, "y": 272}, "elbowR": {"x": 245, "y": 272}}
    return {"joints": {**base, **moved}}


def test_resting_pose_needs_no_words():
    assert probe.pose_words(joints()) == ""


def test_raised_arm_is_named_on_its_side():
    words = probe.pose_words(joints(handL={"x": 150, "y": 40}))
    assert "left of the picture is raised" in words
    assert "right of the picture" not in words


def test_arm_straight_out_is_named():
    assert "right of the picture is stretched" in probe.pose_words(joints(handR={"x": 360, "y": 220}))


def test_head_tilt_side():
    assert "tilts toward the right" in probe.pose_words(joints(head={"x": 215, "y": 160}))
    assert "tilts toward the left" in probe.pose_words(joints(head={"x": 185, "y": 160}))


def test_bent_arms_are_named():
    assert "hand at the waist" in probe.pose_words(joints(elbowL={"x": 110, "y": 270}, handL={"x": 170, "y": 320}))
    assert "front of the chest" in probe.pose_words(joints(elbowR={"x": 290, "y": 270}, handR={"x": 215, "y": 240}))


def test_straight_arm_angle():
    assert round(probe.elbow_angle({"x": 0, "y": 0}, {"x": 0, "y": 10}, {"x": 0, "y": 20})) == 180


def test_right_angle_elbow():
    assert round(probe.elbow_angle({"x": 0, "y": 0}, {"x": 0, "y": 10}, {"x": 10, "y": 10})) == 90


def test_feeling_strength_words():
    assert "a little sad" in probe.feeling_words({"feeling": "sad", "strength": 50})
    assert "very sad" in probe.feeling_words({"feeling": "sad", "strength": 100})
    assert "overwhelmingly sad" in probe.feeling_words({"feeling": "sad", "strength": 150})


def test_no_feeling_no_words():
    assert probe.feeling_words({}) == ""


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
