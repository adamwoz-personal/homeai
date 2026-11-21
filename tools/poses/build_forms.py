#!/usr/bin/env python3
"""Build the pose library from the curated list (homeai/poses/forms.json -> homeai/poses/library/), F10.

Why: the forms are the poses Jarvis can hand klein as a mannequin reference, and a library of forms for Ani's
app. Each form becomes a pose-spec/1 file (rig mannequin-2d/2, anatomical; HOMEAI_CONTRACT.md):
- the frame is read from CMU motion capture and retargeted to her skeleton (homeai/poses/forms.py);
- library/<slug>.json (Jarvis's copy) keeps the captured joints in 3D and no dials: the contract allows it (it
  can be sent, not reopened to edit).
- for_ani/<slug>.json (her copy) is the nearest pose her own dials make, fitted with her own maths
  (tools/poses/fit_dials.js on `git show <ref>:pose-core.js`, read only): her dial numbers plus the joints her
  findJoints makes from them, so it opens and edits in her app exactly as drawn. extra.fit says how close it got.
  Measured 2026-11-20 (mean distance, canvas units): with only her sideways dials (main) even standing was 10
  off and 5 of 19 forms read right; with the forward dials (skeleton2-turn) 16 of 19 do. The default ref is
  that branch while she reviews it; if she says no, rebuild with --ref main.
Writes both folders with an index.json each, and ~/homeai-bench/forms-<time>/sheet.png: per form the capture
from the front and the side, then her version from the front and the side.

Usage:
    .venv/bin/python tools/poses/build_forms.py [--pose-repo ~/ani/pose-app] [--ref main]
"""
from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
import tempfile
from pathlib import Path

from PIL import Image, ImageDraw

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

from homeai.benchio import bench_path  # noqa: E402
from homeai.poses import forms as FM  # noqa: E402
from homeai.poses import mocap as M  # noqa: E402
from homeai.poses import render as R  # noqa: E402
from tools.poses.fetch_cmu import fetch  # noqa: E402

LIST = REPO / "homeai/poses/forms.json"
LIBRARY = REPO / "homeai/poses/library"
FOR_ANI = REPO / "homeai/poses/for_ani"
# Her version is kept only when it still reads as the same pose. By eye on the 2026-11-20 sheets: up to 27 units
# off on average it does (flat rig: waving 17, pointing 26, wide stance 27; forward dials: sword guard 25, long
# jab 24, sitting 22, front and side views); from 28 it doesn't (flat rig: sword thrust 28 folds the arms,
# running stride 29 becomes a lunge; forward dials: kneeling 34 and crouching 46 need a spine that bends forward).
ANI_MEAN = 27.5


def slug(name: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", name.lower()).strip("-")


def captured(entry: dict) -> dict:
    files = fetch(entry["trial"])
    skeleton, frames = M.load(files["asf"], files["amc"])
    return FM.from_capture(FM.capture_frame(skeleton, frames, entry["frame"]))


def her_core(repo: Path, ref: str, folder: Path) -> Path:
    """Her pose-core.js at ``ref``, copied out with git (her working tree is never touched)."""
    text = subprocess.run(["git", "-C", str(repo), "show", f"{ref}:pose-core.js"], capture_output=True, text=True,
                          check=True).stdout
    path = folder / "pose-core.js"
    path.write_text(text)
    return path


def verify_all(core: Path, specs: list[dict]) -> list[dict]:
    """Her own checkSavedPose and reopening (tools/poses/fit_dials.js --verify); raises if any fails."""
    done = subprocess.run(["node", str(REPO / "tools/poses/fit_dials.js"), str(core), "--verify"],
                          input=json.dumps(specs), capture_output=True, text=True, timeout=120)
    if done.returncode:
        why = next((line for line in done.stderr.splitlines() if "Error" in line), done.stderr.strip()[:300])
        raise RuntimeError("her app would not take a form: " + why.strip())
    return json.loads(done.stdout)


def her_joints(core: Path, asked: list[dict]) -> list[dict]:
    """Forms set by hand on her dials ({name, pose}): the whole pose and the joints her findJoints makes."""
    done = subprocess.run(["node", str(REPO / "tools/poses/fit_dials.js"), str(core), "--joints"],
                          input=json.dumps(asked), capture_output=True, text=True, timeout=120)
    if done.returncode:
        why = next((line for line in done.stderr.splitlines() if "Error" in line), done.stderr.strip()[:300])
        raise RuntimeError("her dials would not make a form: " + why.strip())
    return json.loads(done.stdout)


def fit_all(core: Path, targets: list[dict]) -> list[dict]:
    done = subprocess.run(["node", str(REPO / "tools/poses/fit_dials.js"), str(core)], input=json.dumps(targets),
                          capture_output=True, text=True, check=True, timeout=600)
    return json.loads(done.stdout)


def source_of(entry: dict) -> str:
    """Where a form comes from: a CMU frame, or set by hand on her dials."""
    return "set by hand on her dials" if "dials" in entry else f"CMU {entry['trial']} frame {entry['frame']}"


def keeps(fitted: dict | None) -> bool:
    """Her version still reads as the same pose (ANI_MEAN)."""
    return fitted is not None and fitted["mean"] <= ANI_MEAN


def jarvis_copy(entry: dict, joints: dict) -> dict:
    """The captured pose in 3D, no dials."""
    extra = {"source": source_of(entry), "credit": FM.CREDIT, "tags": entry["tags"]}
    if "dials" in entry:
        extra = {"source": source_of(entry), "tags": entry["tags"]}
    return FM.to_spec(joints, name=entry["name"], description=entry["description"], extra=extra)


def ani_copy(entry: dict, fitted: dict) -> dict:
    """Her dials and the joints her own maths makes from them; how far that is from the capture."""
    extra = {"source": source_of(entry) + ("" if "dials" in entry else ", nearest pose her dials make"),
             "tags": entry["tags"], "fit": {"mean": fitted["mean"], "max": fitted["max"]}}
    if "dials" not in entry:
        extra["credit"] = FM.CREDIT
    return FM.to_spec(fitted["joints"], name=entry["name"], description=entry["description"], pose=fitted["pose"],
                      extra=extra)


def contact_sheet(pairs: list[tuple[dict, dict | None]], path: Path) -> None:
    cell, per_row = (120, 156), 3
    out = Image.new("RGB", (cell[0] * 4 * per_row, (cell[1] + 18) * ((len(pairs) + per_row - 1) // per_row)), "white")
    pen = ImageDraw.Draw(out)
    for i, (jarvis, ani) in enumerate(pairs):
        x, y = (i % per_row) * cell[0] * 4, (i // per_row) * (cell[1] + 18)
        views = [jarvis["joints"], FM.turned(jarvis["joints"], 90)]
        views += [ani["joints"], FM.turned(ani["joints"], 90)] if ani else []
        for k, joints in enumerate(views):
            out.paste(R.draw([R.Placed(joints)], *cell), (x + k * cell[0], y + 18))
        fit = f" | Ani's: {ani['extra']['fit']['mean']:.0f} off" if ani else ""
        pen.text((x + 4, y + 3), f"{jarvis['name']}{fit}", fill="black")
    out.save(path)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--pose-repo", type=Path, default=Path.home() / "ani/pose-app")
    ap.add_argument("--ref", default="skeleton2-turn")
    args = ap.parse_args(argv)
    listing = json.loads(LIST.read_text())
    entries = listing["forms"]
    tmp = tempfile.TemporaryDirectory()
    core = her_core(args.pose_repo, args.ref, Path(tmp.name))
    by_hand = {f["name"]: f for f in her_joints(core, [{"name": e["name"], "pose": e["dials"]}
                                                       for e in entries if "dials" in e])}
    targets = [{"name": e["name"], "joints": by_hand[e["name"]]["joints"] if "dials" in e else captured(e)}
               for e in entries]
    try:
        fits = {f["name"]: f for f in fit_all(core, [t for t, e in zip(targets, entries) if "dials" not in e])}
        fits.update(by_hand)
    except (OSError, subprocess.SubprocessError) as exc:
        print("no dial fit (her repo or node missing):", exc)
        fits = {}
    LIBRARY.mkdir(parents=True, exist_ok=True)
    FOR_ANI.mkdir(parents=True, exist_ok=True)
    index, pairs = [], []
    for entry, target in zip(entries, targets):
        name = slug(entry["name"]) + ".json"
        jarvis = jarvis_copy(entry, target["joints"])
        (LIBRARY / name).write_text(json.dumps(jarvis, indent=1) + "\n")
        fitted = fits.get(entry["name"])
        ani = ani_copy(entry, fitted) if keeps(fitted) else None
        if ani:
            (FOR_ANI / name).write_text(json.dumps(ani, indent=1) + "\n")
        pairs.append((jarvis, ani))
        index.append({"name": entry["name"], "file": name, "tags": entry["tags"],
                      "ani_fit": ani["extra"]["fit"] if ani else None,
                      "fit_mean": fitted["mean"] if fitted else None})
    head = {"credit": FM.CREDIT, "list": "homeai/poses/forms.json"}
    (LIBRARY / "index.json").write_text(json.dumps({**head, "forms": index}, indent=1) + "\n")
    (FOR_ANI / "index.json").write_text(json.dumps({**head, "fitted_with": f"pose-core.js @ {args.ref}",
                                                    "forms": [r for r in index if r["ani_fit"]]}, indent=1) + "\n")
    ani_specs = [json.loads((FOR_ANI / r["file"]).read_text()) for r in index if r["ani_fit"]]
    if core is not None and ani_specs:
        checked = verify_all(core, ani_specs)
        print(f"her app takes all {len(checked)} of her forms; reopened joints off by at most "
              f"{max(c['worst'] for c in checked):.4f}")
    tmp.cleanup()
    out = bench_path("forms")
    out.mkdir(parents=True, exist_ok=True)
    contact_sheet(pairs, out / "sheet.png")
    for row in index:
        print(f"{row['name']:24} her dials: {row['fit_mean']} off -> {'for Ani' if row['ani_fit'] else 'Jarvis only'}")
    print(out / "sheet.png")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
