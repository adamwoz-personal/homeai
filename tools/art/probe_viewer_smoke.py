"""Smoke-run the real picture viewer headless against a scratch event bus (ART_FEATURES_PLAN F6).

Why: viewer.main() is the one part the unit tests can't reach (it only draws). After a change to its loop
(F6 added typing and the reply line) this proves it starts, follows an ``art`` event, takes a
``typed_reply`` and keeps running, without a desktop, Jarvis or ComfyUI. It saves the two windows'
surfaces (HOMEAI_ART_VIEWER_SHOTS) so the layout can be looked at.

    .venv/bin/python tools/art/probe_viewer_smoke.py [PICTURE.png]

Writes ~/homeai-bench/viewer-smoke-<time>/ (window-1.png, features-1.txt.png, viewer.log) and prints PASS/FAIL.
"""
from __future__ import annotations

import os
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

from homeai.avatar.bus import EventBus  # noqa: E402


def sample_picture() -> Path | None:
    pictures = sorted((Path.home() / "Pictures/homeai").rglob("*.png"))
    return pictures[-1] if pictures else None


def main(argv: list[str]) -> int:
    picture = Path(argv[0]) if argv else sample_picture()
    if picture is None or not picture.exists():
        print("FAIL: no picture to show (pass one)")
        return 2
    out = Path.home() / "homeai-bench" / f"viewer-smoke-{datetime.now():%Y%m%d-%H%M%S}"
    out.mkdir(parents=True)
    sock = out / "ev.sock"
    bus = EventBus(sock)
    bus.start()
    env = {**os.environ, "SDL_VIDEODRIVER": "dummy", "HOMEAI_AVATAR_SOCKET": str(sock),
           "HOMEAI_ART_VIEWER_SHOTS": str(out / "shots")}
    log = (out / "viewer.log").open("w")
    proc = subprocess.Popen([sys.executable, "-m", "homeai.art.viewer"], cwd=REPO, env=env, stdout=log, stderr=log)
    try:
        deadline = time.monotonic() + 10
        while bus.client_count == 0 and time.monotonic() < deadline:
            time.sleep(0.1)
        bus.publish("typed_reply", {"text": "Drawing it now."})
        bus.publish("art", {"active": True, "status": "Here it is.",
                            "current": {"node": "n1", "image": str(picture), "instruction": "a red dragon"},
                            "lineage": [{"node": "n1", "image": str(picture)}],
                            "features": [{"id": "A", "kind": "subject", "text": "a red dragon with one head"}]})
        time.sleep(3)
        alive = proc.poll() is None
    finally:
        proc.terminate()
        try:
            proc.wait(timeout=5)
        except subprocess.TimeoutExpired:
            proc.kill()
        bus.stop()
        log.close()
    shots = sorted(p.name for p in (out / "shots").glob("*")) if (out / "shots").exists() else []
    ok = alive and "window-1.png" in shots
    print(f"{'PASS' if ok else 'FAIL'}: viewer {'ran' if alive else 'exited early'}; shots {shots}; see {out}")
    if not alive:
        print((out / "viewer.log").read_text()[-2000:])
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
