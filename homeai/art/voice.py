"""Art mode by voice: the piece between the voice daemon and the art studio (AVATAR_PLAN 6a step 8).

The daemon offers every sentence here before the agent sees it. ``handle``
returns the sentence to speak, or None to let Jarvis answer as usual:

- Outside art mode only "art mode" / "let's draw", or a drawing request that
  names a picture ("draw a fox", "make a picture of ..."), is taken. It switches
  the card to art mode (the coder stops, ComfyUI starts, ~7 s), opens the picture
  window, and draws if there was something to draw.
- In art mode the drawing session gets the sentence; house talk ("turn off the
  lights") still goes to Jarvis. "I'm done drawing" leaves.
- Only names in ``users`` may draw (default Adam) until AVATAR_PLAN 6e brings
  Ani's kid policy. An unrecognised voice is refused: this fails closed.
- 20 minutes with no art sentence and nothing drawing goes back to voice mode.

Finished pictures pass the image policy, then are copied to ~/Pictures/homeai/.
The picture window (homeai/art/viewer.py) is a separate process that follows
the latest ``art`` event on the event bus; the avatar window is not touched.
"""
from __future__ import annotations

import logging
import os
import re
import shutil
import subprocess
import sys
import threading
import time
from pathlib import Path
from typing import Callable

from homeai.art import grammar

log = logging.getLogger("homeai.art.voice")

PICTURES = Path.home() / "Pictures" / "homeai"
IDLE_EXIT_S = 20 * 60
IDLE_CHECK_S = 30.0
VIEWER_UNIT = "homeai-art-viewer"
REPO = Path(__file__).resolve().parents[2]

ENTER = re.compile(r"^((switch|go|change|move) (in)?to |start |open |enter |turn on )?(the )?art mode( on)?$|"
                   r"^let'?s (draw|paint|make (some )?(art|pictures))$")
PICTURE_WORDS = re.compile(r"\b(draw|paint|sketch|picture|image|illustration|drawing|painting|portrait)\b")

MSG_READY = "Art mode. What shall I draw?"
MSG_ALREADY = "We're in art mode."
MSG_LEFT = "Done drawing."
MSG_NOT_FOR_YOU = "Drawing is only set up for Adam so far."
MSG_SWITCH_FAILED = "I couldn't start art mode: {why}"
MSG_IDLE = "art mode idle for {minutes} minutes: back to voice mode"


def is_enter(text: str) -> bool:
    return bool(ENTER.match(grammar._tidy(text)))


def wants_picture(text: str) -> bool:
    """A drawing request outside art mode: parses as a new picture and names one."""
    intent = grammar.parse(text)
    return intent is not None and intent.kind == "new" and bool(PICTURE_WORDS.search(grammar._tidy(text)))


class Viewer:
    """The picture window as a transient user unit, so it outlives nothing it shouldn't."""

    def __init__(self, run: Callable[[list[str]], int] | None = None, socket: str = "", shots: str = "") -> None:
        self.socket = socket                       # empty: the daemon's event bus
        self.shots = shots                         # live checks: a folder for window screenshots
        self._run = run or (lambda cmd: subprocess.run(cmd, check=False, capture_output=True, timeout=15).returncode)

    def command(self) -> list[str]:
        env = [f"--setenv={k}={os.environ[k]}" for k in ("DISPLAY", "WAYLAND_DISPLAY", "XAUTHORITY")
               if os.environ.get(k)]
        if self.socket:
            env.append(f"--setenv=HOMEAI_AVATAR_SOCKET={self.socket}")
        if self.shots:
            env.append(f"--setenv=HOMEAI_ART_VIEWER_SHOTS={self.shots}")
        return ["systemd-run", "--user", "--collect", "--quiet", f"--unit={VIEWER_UNIT}",
                f"--working-directory={REPO}", "--setenv=SDL_VIDEODRIVER=x11", *env,
                sys.executable, "-m", "homeai.art.viewer"]

    def start(self) -> None:
        self.stop()
        if self._run(self.command()) != 0:
            log.warning("picture window did not start")

    def stop(self) -> None:
        self._run(["systemctl", "--user", "stop", VIEWER_UNIT])


def export_name(project: str, node: str) -> str:
    return f"{project}-{node}.png"


class ArtVoice:
    """``build(owner, notify)`` returns (session, queue) wired to the real studio; tests pass fakes."""

    def __init__(self, switcher, build, publish: Callable[[str, dict], None], viewer=None,
                 users: tuple[str, ...] = ("Adam",), pictures: Path = PICTURES, idle_s: float = IDLE_EXIT_S,
                 clock: Callable[[], float] = time.monotonic, on_state: Callable[[str], None] | None = None,
                 background: Callable[[Callable[[], None]], None] | None = None) -> None:
        self.switcher, self._build, self._publish = switcher, build, publish
        self.viewer = viewer or Viewer()
        self.users = tuple(u.lower() for u in users)
        self.pictures, self.idle_s, self._clock = Path(pictures), idle_s, clock
        self._on_state = on_state or (lambda state: None)
        self._background = background or (lambda fn: threading.Thread(target=fn, name="art-leave",
                                                                      daemon=True).start())
        self.session = None
        self.queue = None
        self.owner: str | None = None
        self.status = ""
        self._last = clock()
        self._lock = threading.RLock()
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None

    @property
    def active(self) -> bool:
        return self.session is not None

    def _may_draw(self, speaker: str | None) -> bool:
        return bool(speaker) and speaker.lower() in self.users

    # -- the one entry point ---------------------------------------------------

    def handle(self, text: str, speaker: str | None) -> str | None:
        with self._lock:
            if not self.active:
                if is_enter(text):
                    return self._enter(speaker, None)
                if wants_picture(text):
                    return self._enter(speaker, text)
                return None
            intent = grammar.parse(text)
            if intent is None and not is_enter(text):
                return None                                  # house talk: Jarvis answers
            self._last = self._clock()
            if intent is not None and intent.kind == "leave":
                return self.leave()
            if is_enter(text):
                return MSG_ALREADY
            if not self._may_draw(speaker):
                return MSG_NOT_FOR_YOU
            reply = self.session.handle(text)
            if reply is None:
                return None
            self.status = reply.spoken
            self._publish_state()
            return reply.spoken

    # -- entering and leaving ----------------------------------------------------

    def _enter(self, speaker: str | None, first: str | None) -> str:
        if not self._may_draw(speaker):
            return MSG_NOT_FOR_YOU
        self._on_state("mode_switching")
        try:
            if self.switcher.mode() != "art":
                self.switcher.art()
        except Exception as exc:  # noqa: BLE001 - ModeError, or a systemctl failure: say it, stay in voice
            log.warning("art mode failed: %s", exc)
            return MSG_SWITCH_FAILED.format(why=str(exc).split(";")[0])
        self.owner = speaker
        self.session, self.queue = self._build(speaker.lower(), self._on_job)
        self.queue.start()
        self.viewer.start()
        self._last = self._clock()
        self._start_idle_watch()
        if first is None:
            self.status = MSG_READY
            self._publish_state()
            return MSG_READY
        reply = self.session.handle(first)
        self.status = reply.spoken if reply else MSG_READY
        self._publish_state()
        return f"Art mode. {reply.spoken}" if reply else MSG_READY

    def leave(self, why: str = "asked") -> str:
        """Back to voice mode. The slow part (coder reload) runs in the background."""
        with self._lock:
            if not self.active:
                return MSG_LEFT
            session, queue = self.session, self.queue
            self.session = self.queue = None
            self.owner, self.status = None, ""
            self._stop.set()
        for job in getattr(session, "jobs", []):
            try:
                queue.cancel(job)
            except Exception:  # noqa: BLE001 - leaving must not fail on one job
                log.warning("could not cancel %s", job, exc_info=True)
        queue.stop()
        self.viewer.stop()
        self._publish("art", {"active": False})
        log.info("leaving art mode (%s)", why)

        def back_to_voice() -> None:
            try:
                self.switcher.voice()
            except Exception:  # noqa: BLE001 - logged; homeai-mode voice fixes it by hand
                log.exception("going back to voice mode failed")
        self._background(back_to_voice)
        return MSG_LEFT

    # -- idle exit -------------------------------------------------------------

    def idle(self) -> bool:
        """Whether art mode has been quiet long enough to leave."""
        if not self.active or self._clock() - self._last < self.idle_s:
            return False
        return not self.queue.waiting() and not self._drawing()

    def _drawing(self) -> bool:
        for job in getattr(self.session, "jobs", []):
            state = self.queue.get(job)
            if state is not None and state.state == "running":
                return True
        return False

    def check_idle(self) -> bool:
        with self._lock:
            if not self.idle():
                return False
        self.leave(MSG_IDLE.format(minutes=round(self.idle_s / 60)))
        return True

    def _start_idle_watch(self) -> None:
        self._stop = threading.Event()
        stop = self._stop

        def watch() -> None:
            while not stop.wait(IDLE_CHECK_S):
                try:
                    if self.check_idle():
                        return
                except Exception:  # noqa: BLE001 - the watch must not die silently
                    log.exception("art idle check failed")
        self._thread = threading.Thread(target=watch, name="art-idle", daemon=True)
        self._thread.start()

    def close(self) -> None:
        """Daemon shutdown: stop the threads and the window; ComfyUI is left to homeai-mode."""
        self._stop.set()
        with self._lock:
            queue, self.session, self.queue = self.queue, None, None
        if queue is not None:
            queue.stop()
            self.viewer.stop()

    # -- pictures arriving -----------------------------------------------------

    def _on_job(self, event: dict) -> None:
        session = self.session
        if session is None:
            return
        refused = session.on_event(event)
        if refused:
            self.status = refused
        elif event.get("state") == "done" and event.get("image"):
            self._export(event)
            self.status = "Done."
        elif event.get("state") == "running":
            self.status = "Drawing..."
        elif event.get("state") in ("failed", "cancelled") and event.get("message"):
            self.status = event["message"]
        self._publish_state()

    def _export(self, event: dict) -> None:
        try:
            self.pictures.mkdir(parents=True, exist_ok=True)
            target = self.pictures / export_name(event["project"], event["node"])
            tmp = target.with_suffix(".png.tmp")
            shutil.copyfile(event["image"], tmp)
            os.replace(tmp, target)
        except OSError as exc:
            log.warning("could not save the picture to %s: %s", self.pictures, exc)

    def state(self) -> dict:
        """The ``art`` event: everything the picture window shows (latest wins on the bus)."""
        session = self.session
        if session is None:
            return {"active": False}
        out = {"active": True, "status": self.status, "owner": self.owner, "waiting": 0, "drawing": False,
               "current": None, "lineage": []}
        if self.queue is not None:
            out["waiting"], out["drawing"] = len(self.queue.waiting()), self._drawing()
        project = session.project
        if project is None or project.current is None:
            return out
        out["project"] = project.id
        for node in project.lineage():
            image = project.image(node.id)
            out["lineage"].append({"node": node.id, "instruction": node.instruction, "state": node.state,
                                   "image": str(image) if image.exists() else None})
        out["current"] = out["lineage"][-1] if out["lineage"] else None
        return out

    def _publish_state(self) -> None:
        try:
            self._publish("art", self.state())
        except Exception:  # noqa: BLE001 - the window is a nicety; drawing carries on
            log.warning("art state publish failed", exc_info=True)


def build_studio(owner: str, notify: Callable[[dict], None]):
    """The real session and queue (ComfyUI must already be up)."""
    from homeai.art import comfy, jobs
    from homeai.art.session import ArtSession
    from homeai.art.store import Store
    import homeai.gpu_mode as G

    store = Store()
    switcher = G.ModeSwitcher(G.System(), log=lambda line: log.info("%s", line))
    queue = jobs.JobQueue(comfy.ComfyClient(), store, notify=notify, restart=switcher.restart_comfy)
    return ArtSession(store, queue, owner), queue
