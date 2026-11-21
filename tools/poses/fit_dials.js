#!/usr/bin/env node
// Why this file exists
// --------------------
// Pose forms (homeai/poses, ART_FEATURES_PLAN F10) carry the full joints from motion capture. A form Ani's app
// can reopen and edit also needs her dial numbers ("pose"), and those must be numbers her own maths turns into
// the same joints. This searches her dials with HER findJoints (pose-core.js), so a fitted form is exactly a pose
// her app makes. The waist twist (if her copy has it) stays 0 and the figure faces front, so the numbers mean the
// same on her main branch and on skeleton2-turn.
//
// Read only: her code is loaded from a file given on the command line (build_forms.py writes it from
// `git show main:pose-core.js`), never edited.
//
// Usage: node tools/poses/fit_dials.js /path/to/pose-core.js < targets.json > fits.json
//        node tools/poses/fit_dials.js /path/to/pose-core.js --verify < specs.json    (her app's checks)
//   targets.json: [{"name": "...", "joints": {"hips": {"x":..,"y":..,"z":..}, ...}}, ...]
//   fits.json:    [{"name": "...", "pose": {...}, "joints": {...her findJoints...}, "mean": 3.1, "max": 7.9}, ...]
//                 (mean and max: distance from the target joints, canvas units)

const fs = require("node:fs");
const path = require("node:path");

const STEPS = [32, 16, 8, 4, 2, 1];
const FIXED = { waistTwist: 0 };

function error(core, pose, target) {
  const got = core.findJoints(pose);
  let sum = 0, max = 0, n = 0;
  for (const [name, t] of Object.entries(target)) {
    const g = got[name];
    if (!g) continue;
    const d = Math.hypot(g.x - t.x, g.y - t.y, (g.z || 0) - (t.z || 0));
    sum += d; max = Math.max(max, d); n += 1;
    if (!Number.isFinite(d)) return { mean: Infinity, max: Infinity, score: Infinity };
  }
  const mean = sum / n;
  return { mean, max, score: mean + max / 4 };
}

// Coordinate descent on whole-number dials inside her slider ranges, from a few starts.
function fit(core, target) {
  const dials = core.sliders.filter(s => !(s.name in FIXED));
  const starts = [{ ...core.standing }];
  for (const name of Object.keys(core.feelings || {})) starts.push({ ...core.feelings[name].pose });
  let best = null;
  for (const start of starts) {
    const pose = { ...start };
    for (const [name, value] of Object.entries(FIXED)) if (name in pose) pose[name] = value;
    let here = error(core, pose, target);
    for (const step of STEPS) {
      let moved = true;
      while (moved) {
        moved = false;
        for (const s of dials) {
          for (const sign of [1, -1]) {
            const value = Math.max(s.min, Math.min(s.max, pose[s.name] + sign * step));
            if (value === pose[s.name]) continue;
            const old = pose[s.name];
            pose[s.name] = value;
            const next = error(core, pose, target);
            if (next.score < here.score - 1e-9) { here = next; moved = true; } else pose[s.name] = old;
          }
        }
      }
    }
    if (!best || here.score < best.err.score) best = { pose: { ...pose }, err: here };
  }
  const joints = {};
  for (const [name, q] of Object.entries(core.findJoints(best.pose))) joints[name] = { x: q.x, y: q.y, z: q.z || 0 };
  const worst = Object.entries(target).map(([name, q]) => [name, Math.round(Math.hypot(joints[name].x - q.x,
    joints[name].y - q.y, joints[name].z - (q.z || 0)))]).sort((a, b) => b[1] - a[1]).slice(0, 4);
  return { pose: best.pose, joints, mean: Math.round(best.err.mean * 100) / 100,
           max: Math.round(best.err.max * 100) / 100, worst };
}

// Her app's own checks on a form for her: it is a saved pose she can keep (checkSavedPose), and opening it
// (poseFromSaved, then findJoints) puts every joint where the file says, to a hundredth.
function verify(core, spec) {
  core.checkSavedPose(spec);
  const opened = core.poseFromSaved(spec);
  const got = core.findJoints(opened.pose);
  let worst = 0;
  for (const [name, q] of Object.entries(spec.joints)) {
    worst = Math.max(worst, Math.hypot(got[name].x - q.x, got[name].y - q.y, (got[name].z || 0) - q.z));
  }
  if (worst > 0.01) throw new Error(`${spec.name}: reopened joints are ${worst.toFixed(3)} off`);
  return worst;
}

if (require.main === module) {
  const corePath = process.argv[2];
  if (!corePath) { console.error("usage: fit_dials.js pose-core.js [--verify] < targets.json"); process.exit(2); }
  const core = require(path.resolve(corePath));
  if (process.argv[3] === "--verify") {
    const specs = JSON.parse(fs.readFileSync(0, "utf8"));
    const out = specs.map(spec => ({ name: spec.name, worst: verify(core, spec) }));
    process.stdout.write(JSON.stringify(out) + "\n");
    process.exit(0);
  }
  const targets = JSON.parse(fs.readFileSync(0, "utf8"));
  const out = targets.map(t => ({ name: t.name, ...fit(core, t.joints) }));
  process.stdout.write(JSON.stringify(out, null, 1) + "\n");
}

module.exports = { fit, error, verify };
