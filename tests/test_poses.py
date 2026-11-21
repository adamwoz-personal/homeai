"""The pose library (homeai/poses, tools/poses; ART_FEATURES_PLAN F10): motion capture read right, retargeted to
Ani's mannequin, written in her contract's pose-spec/1, drawn back to front; and the shipped library is valid.
No downloads and none of Ani's files: synthetic skeletons, and the committed library."""
from __future__ import annotations

import json
import math
import shutil
import subprocess

import numpy as np
import pytest

from homeai.poses import forms as FM
from homeai.poses import mocap as M
from homeai.poses import render as R
from tools.poses import build_forms as B
from tools.poses import fetch_cmu as C
from tools.poses import sheet_clips as S

# -- reading motion capture ---------------------------------------------------------------------------------------


def test_euler_turns_x_toward_y_about_z():
    assert np.allclose(M.euler(0, 0, 90) @ [1, 0, 0], [0, 1, 0])
    assert np.allclose(M.euler(90, 0, 0) @ [0, 1, 0], [0, 0, 1])
    assert np.allclose(M.euler(10, 20, 30), M.euler(0, 0, 30) @ M.euler(0, 20, 0) @ M.euler(10, 0, 0))


ASF = """:version 1.10
:units
  length 1.0
:root
   order TX TY TZ RX RY RZ
   axis XYZ
   position 0 0 0
   orientation 0 0 0
:bonedata
  begin
     id 1
     name upper
     direction 1 0 0
     length 2
     axis 0 0 90  XYZ
     dof rx ry rz
  end
  begin
     id 2
     name lower
     direction 1 0 0
     length 1
     axis 0 0 0  XYZ
     dof rz
  end
:hierarchy
  begin
    root upper
    upper lower
  end
"""


def test_a_bone_turns_in_its_own_frame_and_its_child_follows():
    sk = M.parse_asf(ASF)
    rest = M.positions(sk, {"root": [1, 2, 3, 0, 0, 0]})
    assert np.allclose(rest["upper"], [3, 2, 3]) and np.allclose(rest["lower"], [4, 2, 3])
    # The upper bone's frame is turned 90 about z, so its own x axis is the world's y: "rx 90" spins it about y.
    turned = M.positions(sk, {"root": [0, 0, 0, 0, 0, 0], "upper": [90, 0, 0], "lower": [0]})
    assert np.allclose(turned["upper"], [0, 0, -2]) and np.allclose(turned["lower"], [0, 0, -3])
    bent = M.positions(sk, {"root": [0, 0, 0, 0, 0, 0], "upper": [0, 0, 0], "lower": [90]})
    assert np.allclose(bent["lower"], [2, 1, 0])
    spun = M.positions(sk, {"root": [0, 0, 0, 0, 0, 90]})
    assert np.allclose(spun["upper"], [0, 2, 0])


def test_frames_are_read_in_order():
    frames = M.parse_amc(":FULLY-SPECIFIED\n:DEGREES\n1\nroot 0 0 0 0 0 0\nupper 1 2 3\n2\nroot 1 1 1 0 0 0\n")
    assert len(frames) == 2 and frames[0]["upper"] == [1.0, 2.0, 3.0] and frames[1]["root"][0] == 1.0


@pytest.mark.parametrize("asf, amc", [
    (ASF.replace(":hierarchy", ":nothing"), None),
    (ASF.replace("upper lower", "upper elbow"), None),
    (None, "root 0 0 0 0 0 0\n"),
    (None, ":DEGREES\n"),
])
def test_broken_files_say_so(asf, amc):
    with pytest.raises(M.MocapError):
        M.parse_asf(asf) if asf else M.parse_amc(amc)


# -- retargeting to Ani's mannequin ------------------------------------------------------------------------------

def capture(facing_deg: float = 0.0, arm_forward: bool = False) -> dict[str, np.ndarray]:
    """A person in CMU's axes (y up), arms down or the left arm straight ahead, facing ``facing_deg`` about y."""
    left, up = np.array([1.0, 0, 0]), np.array([0, 1.0, 0])
    fwd = np.cross(left, up)
    p = {"root": up * 10, "lhipjoint": up * 9 + left * 1.2, "rhipjoint": up * 9 - left * 1.2,
         "lfemur": up * 5 + left * 1.2, "rfemur": up * 5 - left * 1.2, "ltibia": up * 1 + left * 1.2,
         "rtibia": up * 1 - left * 1.2, "thorax": up * 16, "lclavicle": up * 15.6 + left * 2.5,
         "rclavicle": up * 15.6 - left * 2.5, "upperneck": up * 17, "head": up * 18.5,
         "lhumerus": up * 12.6 + left * 2.5, "rhumerus": up * 12.6 - left * 2.5,
         "lradius": up * 9.8 + left * 2.5, "rradius": up * 9.8 - left * 2.5}
    if arm_forward:
        p["lhumerus"] = p["lclavicle"] + fwd * 3
        p["lradius"] = p["lhumerus"] + fwd * 2.8
    turn = M.euler(0, facing_deg, 0)
    return {k: turn @ v for k, v in p.items()}


def length(j, a, b):
    return math.dist([j[a][k] for k in "xyz"], [j[b][k] for k in "xyz"])


@pytest.mark.parametrize("facing", [0, 37, 180, 290])
def test_her_bone_lengths_her_floor_and_facing_the_viewer_whichever_way_they_faced(facing):
    j = FM.from_capture(capture(facing))
    assert set(j) == set(FM.JOINTS)
    for a, b, want in (("hipL", "kneeL", FM.THIGH), ("kneeR", "ankleR", FM.SHIN), ("hips", "chestTop", FM.SPINE),
                       ("shoulderL", "elbowL", FM.UPPER_ARM), ("elbowR", "handR", FM.FOREARM),
                       ("chestTop", "neckTop", FM.NECK), ("neckTop", "head", FM.HEAD), ("hips", "hipL", FM.HIP_SIDE)):
        assert length(j, a, b) == pytest.approx(want)
    assert max(j["ankleL"]["y"], j["ankleR"]["y"]) == pytest.approx(FM.GROUND - 6)
    assert j["hips"]["x"] == FM.MIDDLE and j["hips"]["z"] == 0
    assert j["hipL"]["x"] > FM.MIDDLE and j["hipL"]["y"] == pytest.approx(j["hips"]["y"])   # level, left on the right
    assert j["shoulderL"]["y"] == pytest.approx(j["chestTop"]["y"] + FM.SHOULDER_DROP)
    assert all(abs(q["z"]) < 1e-9 for q in j.values())                                    # a flat pose stays flat


def test_an_arm_reaching_forward_comes_toward_the_viewer():
    j = FM.from_capture(capture(120, arm_forward=True))
    assert j["handL"]["z"] == pytest.approx(FM.UPPER_ARM + FM.FOREARM) and j["handR"]["z"] == pytest.approx(0)


def test_the_turn_is_the_contract_s():
    j = FM.from_capture(capture())
    j["handL"] = {"x": 300.0, "y": 200.0, "z": 0.0}                           # arm straight out to her left
    side = FM.turned(j, 90)
    assert side["handL"]["x"] == pytest.approx(FM.MIDDLE) and side["handL"]["z"] == pytest.approx(-100)
    assert FM.turned(j, 0) == j
    assert all(side[n]["y"] == j[n]["y"] for n in j)
    back = FM.turned(j, 180)
    assert back["handL"]["x"] == pytest.approx(100)


@pytest.mark.parametrize("turn, view", [(0, "front"), (22, "front"), (23, "three-quarter"), (68, "side"),
                                        (90, "side"), (158, "back"), (270, "side"), (338, "front")])
def test_view_names(turn, view):
    assert FM.view_of(turn) == view


@pytest.mark.parametrize("bad", [-1, 360, 1.5, "90", True])
def test_a_bad_turn_is_refused(bad):
    with pytest.raises(FM.FormError):
        FM.view_of(bad)


def spec_of(**kw):
    return FM.to_spec(FM.from_capture(capture(arm_forward=True)), name="Test", description="a figure", **kw)


def test_a_form_is_a_pose_spec_in_her_field_order():
    spec = spec_of(turn=180, extra={"tags": ["x"]})
    assert list(spec) == ["schema", "rig", "name", "description", "view", "turn", "sides", "canvas", "joints", "extra"]
    assert (spec["schema"], spec["rig"], spec["sides"], spec["view"]) == ("pose-spec/1", "mannequin-2d/2",
                                                                          "anatomical", "back")
    for q in spec["joints"].values():
        for v in q.values():
            assert v == round(v, 2) and not (v == 0 and math.copysign(1, v) < 0)        # two decimals, no -0
    assert spec["joints"]["handL"]["z"] < 0                                    # from the back the reach goes away
    assert "pose" in spec_of(pose={"headTilt": 0}) and list(spec_of(pose={})).index("pose") == 8


@pytest.mark.parametrize("change, why", [
    (lambda s: s["joints"].pop("head"), "missing"),
    (lambda s: s["joints"].update(tail={"x": 1, "y": 1, "z": 1}), "extra"),
    (lambda s: s["joints"]["head"].update(z=401), "outside"),
    (lambda s: s["joints"]["head"].update(x=float("nan")), "outside"),
    (lambda s: s.update(view="side"), "does not match"),
    (lambda s: s.update(canvas={"width": 800, "height": 520}), "canvas"),
    (lambda s: s.update(sides="mirror"), "anatomical"),
    (lambda s: s.update(name="x" * 81), "80"),
])
def test_the_contract_s_limits_hold(change, why):
    spec = spec_of()
    change(spec)
    with pytest.raises(FM.FormError, match=why):
        FM.check_spec(spec)


# -- drawing -----------------------------------------------------------------------------------------------------

def test_from_the_side_the_far_arm_is_painted_before_the_chest_and_the_near_arm_after():
    j = FM.from_capture(capture())
    j["handL"] = {"x": 300.0, "y": 200.0, "z": 0.0}
    j["handR"] = {"x": 100.0, "y": 200.0, "z": 0.0}
    order = R.shapes(FM.turned(j, 90))
    hands = [i for i, s in enumerate(order) if s["kind"] == "ball" and s["r"] == 9]
    chest = max(i for i, s in enumerate(order) if s["kind"] == "block")
    depths = [s["depth"] for s in order]
    assert depths == sorted(depths) and hands[0] < chest < hands[1]


def test_the_nose_shows_the_facing():
    nose = [s for s in R.shapes(FM.turned(FM.from_capture(capture()), 90)) if s["kind"] == "ball" and s["r"] == 5][0]
    assert nose["c"][0] > FM.MIDDLE + 15


def test_figures_land_where_they_are_placed():
    j = FM.from_capture(capture())
    picture = np.asarray(R.draw([R.Placed(j, x=0.25), R.Placed(j, x=0.75, scale=0.8)], 800, 520)).astype(int)
    wood = (abs(picture - np.array(R.WOOD)).sum(axis=2) < 40)
    columns = np.flatnonzero(wood.any(axis=0))
    assert columns.min() > 100 and columns.max() < 700
    assert wood[:, 180:220].any() and wood[:, 580:620].any() and not wood[:, 390:410].any()
    rows_left, rows_right = np.flatnonzero(wood[:, :400].any(axis=1)), np.flatnonzero(wood[:, 400:].any(axis=1))
    assert rows_right.min() > rows_left.min() + 30                              # the smaller one is shorter


def test_the_outline_keeps_only_the_edge():
    assert sorted(R._hull([(0, 0), (2, 0), (2, 2), (0, 2), (1, 1), (2, 2)])) == [(0, 0), (0, 2), (2, 0), (2, 2)]


# -- the shipped library -----------------------------------------------------------------------------------------

def library(folder):
    return {p.name: json.loads(p.read_text()) for p in folder.glob("*.json") if p.name != "index.json"}


def test_every_form_in_the_library_keeps_the_contract():
    forms = library(B.LIBRARY)
    listed = json.loads(B.LIST.read_text())["forms"]
    assert set(forms) == {B.slug(e["name"]) + ".json" for e in listed}
    for name, spec in forms.items():
        FM.check_spec(spec)
        if spec["extra"]["source"].startswith("CMU "):
            assert spec["extra"]["credit"] == FM.CREDIT, name
        else:                                                       # set by hand on her dials: nothing to credit
            assert spec["extra"]["source"] == "set by hand on her dials" and "credit" not in spec["extra"], name
        assert "pose" not in spec, name                          # Jarvis's copies keep the capture, no dials
    index = json.loads((B.LIBRARY / "index.json").read_text())
    assert index["credit"] == FM.CREDIT and [f["file"] for f in index["forms"]] == sorted(forms, key=[
        B.slug(e["name"]) + ".json" for e in listed].index)


def test_ani_gets_only_forms_her_dials_make_and_that_still_read_as_the_pose():
    forms = library(B.FOR_ANI)
    assert 3 <= len(forms) < len(library(B.LIBRARY))
    for spec in forms.values():
        FM.check_spec(spec)
        assert spec["pose"] and spec["extra"]["fit"]["mean"] <= B.ANI_MEAN
        assert "shoulderForwardL" in spec["pose"]                  # fitted with her forward dials (skeleton2-turn)
    names = {s["name"] for s in forms.values()}
    assert {"Standing", "Waving one arm high", "Sword thrust"} <= names and "Kneeling on one knee" not in names
    assert any(abs(q["z"]) > 20 for s in forms.values() for q in s["joints"].values())    # some reach toward us
    assert json.loads((B.FOR_ANI / "index.json").read_text())["fitted_with"] == "pose-core.js @ skeleton2-turn"


def test_descriptions_name_the_pose_not_a_prop():
    for entry in json.loads(B.LIST.read_text())["forms"]:
        assert not any(w in entry["description"] for w in ("sword", "shield", "bow", "staff", "gun")), entry["name"]


@pytest.mark.parametrize("mean, kept", [(10, True), (27.1, True), (27.5, True), (28.03, False)])
def test_her_copy_is_kept_up_to_the_cutoff(mean, kept, monkeypatch):
    entry = {"name": "N", "trial": "01_01", "frame": 1, "tags": [], "description": "a figure"}
    fitted = {"pose": {"headTilt": 0}, "joints": FM.from_capture(capture()), "mean": mean, "max": 1}
    assert B.keeps(fitted) is kept and not B.keeps(None)
    assert B.ani_copy(entry, fitted)["extra"]["fit"]["mean"] == mean


# -- the tools ---------------------------------------------------------------------------------------------------

LISTING = """<table><tr><td>Subject #2 (various expressions)</td></tr><tr><td>Trial #</td><td>Motion Description</td></tr>
<tr><td>5</td><td>punch/strike</td><td>Animated</td></tr><tr><td>7</td><td>swordplay</td></tr>
<tr><td>Subject #13 (misc)</td></tr><tr><td>17</td><td>boxing</td></tr><tr><td>17</td><td>boxing again</td></tr></table>"""


def test_the_cmu_listing_becomes_trials():
    rows = C.parse_listing(LISTING)
    assert [(r["trial"], r["description"]) for r in rows] == [("02_05", "punch/strike"), ("02_07", "swordplay"),
                                                               ("13_17", "boxing")]
    assert [r["trial"] for r in C.search(rows, ["sword", "BOX"])] == ["02_07", "13_17"]


def test_fetch_downloads_once_and_refuses_a_web_page(tmp_path):
    calls = []

    def get(url):
        calls.append(url)
        return b"# skeleton" if url.endswith(".asf") else b":FULLY-SPECIFIED\n1\n"
    files = C.fetch("02_05", tmp_path, get)
    assert files["amc"].name == "02_05.amc" and files["asf"].name == "02.asf"
    C.fetch("02_05", tmp_path, get)
    assert len(calls) == 2
    with pytest.raises(OSError):
        C.fetch("03_01", tmp_path, lambda url: b"<html>not found</html>")
    with pytest.raises(ValueError):
        C.clip_urls("../etc")


def test_contact_sheet_frames_spread_evenly():
    assert S.picks(100, 5) == [1, 26, 50, 75, 100]
    assert S.picks(100, 3, 40, 60) == [40, 50, 60]


FAKE_CORE = """
const sliders = [{name: "a", min: -50, max: 50}, {name: "b", min: 0, max: 90}];
const standing = {a: 0, b: 0};
function findJoints(p) { return {hips: {x: 200 + p.a, y: 300}, head: {x: 200, y: 100 - p.b}}; }
function checkSavedPose(s) { if (!s.name) throw new Error("no name"); }
function poseFromSaved(s) { return {pose: s.pose}; }
module.exports = {sliders, standing, findJoints, checkSavedPose, poseFromSaved, feelings: {}};
"""


@pytest.mark.skipif(shutil.which("node") is None, reason="needs node")
def test_the_dial_fit_finds_her_numbers_and_the_check_catches_a_mismatch(tmp_path):
    core = tmp_path / "core.js"
    core.write_text(FAKE_CORE)
    target = {"hips": {"x": 237, "y": 300, "z": 0}, "head": {"x": 200, "y": 31, "z": 0}}
    [fit] = B.fit_all(core, [{"name": "t", "joints": target}])
    assert fit["pose"] == {"a": 37, "b": 69} and fit["mean"] == 0
    good = {"name": "t", "pose": fit["pose"], "joints": target}
    assert B.verify_all(core, [good])[0]["worst"] == 0
    with pytest.raises(RuntimeError, match="off"):
        B.verify_all(core, [{**good, "joints": {**target, "head": {"x": 200, "y": 40, "z": 0}}}])
    with pytest.raises(RuntimeError, match="no name"):
        B.verify_all(core, [{**good, "name": ""}])


def test_a_scene_reference_puts_each_figure_where_its_character_is(tmp_path):
    from tools.poses import probe_scene_reference as P
    spec = P.SCENES["facing_off"]
    for form, turn, x, scale in spec["figures"]:
        assert (B.LIBRARY / f"{form}.json").exists() and 0 < x < 1 and FM.view_of(turn) == "side"
    picture = np.asarray(R.Image.open(P.scene_picture("facing_off", tmp_path / "s.png"))).astype(int)
    wood = abs(picture - np.array(R.WOOD)).sum(axis=2) < 40
    left, right = wood[:, : P.SIZE // 2], wood[:, P.SIZE // 2:]
    assert left.any() and right.any()
    assert np.flatnonzero(right.any(axis=1)).min() < np.flatnonzero(left.any(axis=1)).min()   # the troll is taller
    words = P.prompt_with_reference("A paladin. A troll.", "facing_off")
    assert words.startswith("A paladin. A troll. ") and "the paladin on the left facing right" in words
    assert "the troll on the right facing left" in words and "no mannequin parts" in words


def test_a_form_set_by_hand_says_so_and_is_hers_exactly():
    spec = json.loads((B.FOR_ANI / "sword-and-shield-ready.json").read_text())
    assert spec["extra"]["source"] == "set by hand on her dials" and spec["extra"]["fit"] == {"mean": 0, "max": 0}
    assert spec["pose"]["shoulderForwardR"] == 100
    assert B.source_of({"dials": {}}) == "set by hand on her dials"
    assert B.source_of({"trial": "02_07", "frame": 9}) == "CMU 02_07 frame 9"
