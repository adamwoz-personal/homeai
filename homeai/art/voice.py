"""Art mode by voice: the piece between the voice daemon and the art studio (AVATAR_PLAN 6a step 8).

The daemon offers every sentence here before the agent sees it. ``handle``
returns the sentence to speak, or None to let Jarvis answer as usual:

- Outside art mode only "art mode" / "let's draw", or a drawing request that
  names a picture ("draw a fox", "make a picture of ..."), is taken. It switches
  the card to art mode (the coder stops, ComfyUI starts, ~7 s), opens the picture
  window, and draws if there was something to draw.
- In art mode the drawing session gets the sentence; house talk ("turn off the
  lights") still goes to Jarvis. "I'm done drawing" leaves.
- Everyone may draw. Registered adults (``adults``, default Adam) draw with no
  limits; registered kids and any voice Jarvis doesn't know draw under the
  kid-safe policy (Adam 2026-11-07). The person who entered art mode is the
  audience: while they draw, a voice that is confidently someone else is
  ignored, and one that can't be told apart counts as them (Adam 2026-11-07:
  "remember who the main audience is and ignore the background people").
  A session started by an unknown voice is a guest session under kid rules.
- 20 minutes with no art sentence and nothing drawing goes back to voice mode.

Finished pictures pass the image policy, then are copied to ~/Pictures/homeai/.
The picture window (homeai/art/viewer.py) is a separate process that follows
the latest ``art`` event on the event bus; the avatar window is not touched.
"""
from __future__ import annotations

import logging
import os
import re
import subprocess
import sys
import threading
import time
from pathlib import Path
from typing import Callable

from homeai.art import grammar, policy
from homeai.fsutil import atomic_copy, atomic_write_json

log = logging.getLogger("homeai.art.voice")

PICTURES = Path.home() / "Pictures" / "homeai"
IDLE_EXIT_S = 20 * 60
IDLE_CHECK_S = 30.0
RESUME_WITHIN_S = 60 * 60      # re-entering art mode within this reopens the latest project
VIEWER_UNIT = "homeai-art-viewer"
REPO = Path(__file__).resolve().parents[2]

# Searched, not matched whole: Whisper adds words ("Hey Jervis, let's draw. Wait, bye, ma'am one.") and people
# add their own ("I'm Adam, switch to art mode"); both were missed in Adam's first session (2026-11-07).
# Whisper's mishearings of "art mode" ("switch to Art Mize" went to the chat model, 2026-11-07).
ART_MODE = r"(art|arts|art's|art-) ?(mode|modes|mize|mod|mood|moat|node)"
ENTER = re.compile(r"(^|\b(switch|go|change|move) (in)?to |\b(start|open|enter|turn on) )(the )?" + ART_MODE + r"\b|"
                   r"\blet'?s (draw|paint|sketch|make (some )?(art|pictures))\b")
LETS_DRAW = re.compile(r"\blet'?s (draw|paint|sketch) (?P<what>(?!(something|some|it|that|this)\b).{3,})$")
MODE_QUESTION = re.compile(r"\b(are we|am i|is this|are you) (still )?in art mode\b|\bwhat mode\b")
# In art mode, a question that doesn't mention the picture is for Jarvis ("what time is it").
QUESTION = re.compile(r"^(what|who|when|where|why|how|is|are|am|was|were|will|would|should|do|does|did|tell me)\b")
ABOUT_PICTURE = re.compile(r"\b(picture|image|drawing|painting|prompt|colou?rs?|style|background|version|"
                           r"option|this one|that one|too \w+|(it|this|that) (look|looks|seem|seems))\b")
PICTURE_WORDS = re.compile(r"\b(draw|paint|sketch|picture|image|illustration|drawing|painting|portrait)\b")

MSG_READY = "Art mode. What shall I draw?"
MSG_ALREADY = "We're in art mode."
MSG_IN_ART_MODE = "Yes, we're in art mode. Say I'm done drawing to leave."
MSG_LEFT = "Done drawing."
MSG_SWITCH_FAILED = "I couldn't start art mode: {why}"
MSG_IDLE = "art mode idle for {minutes} minutes: back to voice mode"


def is_enter(text: str) -> bool:
    return bool(ENTER.search(grammar._tidy(text)))


def lets_draw(text: str) -> str | None:
    """'let's draw a dragon' -> 'draw a dragon' (so it enters art mode and draws)."""
    match = LETS_DRAW.search(grammar._tidy(text))
    return f"draw {match.group('what')}" if match else None


COMMON = {"the", "and", "with", "its", "his", "her", "for", "are", "was", "that", "this", "from", "into", "onto",
          "who", "what", "has", "have", "one", "two", "very", "some"}


def for_jarvis(text: str, subject: str = "") -> bool:
    """In art mode: house talk, or a question that isn't about the picture (nor names what's in it)."""
    said = grammar._tidy(text)
    if grammar.HOUSE.search(said):
        return True
    if not QUESTION.match(said) or ABOUT_PICTURE.search(said):
        return False
    words = set(re.findall(r"[a-z]{3,}", subject.lower())) - COMMON
    return not any(w in said for w in words)


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


def folder_name(person: str | None) -> str:
    """A registered person's own folder under ~/Pictures/homeai; everyone else shares guest/."""
    name = re.sub(r"[^a-z0-9_-]+", "", (person or "").lower())
    return name if name and name != "guest" else "guest"


class ArtVoice:
    """``build(owner, notify)`` returns (session, queue) wired to the real studio; tests pass fakes."""

    def __init__(self, switcher, build, publish: Callable[[str, dict], None], viewer=None,
                 adults: tuple[str, ...] = ("Adam",), pictures: Path = PICTURES, idle_s: float = IDLE_EXIT_S,
                 clock: Callable[[], float] = time.monotonic, on_state: Callable[[str], None] | None = None,
                 background: Callable[[Callable[[], None]], None] | None = None,
                 wall: Callable[[], float] = time.time,
                 checker: Callable[[Path, str], str] | None = None,
                 warm_voice: Callable[[], None] | None = None) -> None:
        self.switcher, self._build, self._publish = switcher, build, publish
        self.viewer = viewer or Viewer()
        self.adults = tuple(u.lower() for u in adults)
        self.pictures, self.idle_s, self._clock = Path(pictures), idle_s, clock
        self._on_state = on_state or (lambda state: None)
        self._background = background or (lambda fn: threading.Thread(target=fn, name="art-leave",
                                                                      daemon=True).start())
        self._wall = wall
        self.checker = checker                  # ask(image, question) -> answer; None: pictures aren't checked
        self._warm_voice = warm_voice or _warm_voice_model
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

    def rules_for(self, speaker: str | None):
        """Registered adults: no limits. Everyone else (registered kids, unknown voices): kid-safe."""
        adult = bool(speaker) and speaker.lower() in self.adults
        return policy.load("allow-all" if adult else "kid-safe")

    # -- the one entry point ---------------------------------------------------

    def handle(self, text: str, speaker: str | None, other: bool = False) -> str | None:
        """``other``: the voice is confidently not the person whose session this is (the daemon's speaker check).

        Returns the sentence to say, None for Jarvis to answer, or "" to stay quiet (someone else in the room
        talking about the picture while its owner draws).
        """
        with self._lock:
            if not self.active:
                if wants_picture(text):
                    return self._enter(speaker, text)
                if is_enter(text):
                    return self._enter(speaker, lets_draw(text))
                return None
            intent = grammar.parse(text)
            if intent is not None and intent.kind == "leave":
                self._last = self._clock()
                return self.leave()
            if MODE_QUESTION.search(grammar._tidy(text)):
                return MSG_IN_ART_MODE
            if is_enter(text) and lets_draw(text) is None:
                return MSG_ALREADY
            node = self.session._current()
            subject = " ".join(str(node.spec.get(k) or "") for k in ("subject", "details")) if node else ""
            if intent is None and for_jarvis(text, subject):
                return None                                  # house talk and other questions: Jarvis answers
            if other and self.owner:
                # The person drawing is the audience: the family talking in the background isn't
                # (a child's "it has two heads" became kid rules and changes, Adam 2026-11-07).
                log.info("art: ignoring another voice in %s's session: %r", self.owner, text)
                return ""
            if speaker is None and self.owner:
                speaker = self.owner        # not confidently someone else: the session's owner, not a stranger
            self._last = self._clock()
            self.session.rules = self.rules_for(speaker)
            self.session.speaker = speaker or "guest"
            # Everything else in art mode is about the picture, even without a drawing verb
            # ("let's start with the dragon", "let's make it cyberpunk"): in Adam's first
            # session those went to the chat model, which looked things up instead.
            reply = self.session.handle(text) if intent is not None else self.session.handle_free(text)
            if reply is None:
                return None
            self.status = reply.spoken
            self._publish_state()
            return reply.spoken

    # -- entering and leaving ----------------------------------------------------

    def _enter(self, speaker: str | None, first: str | None) -> str:
        self._on_state("mode_switching")
        try:
            if self.switcher.mode() != "art":
                self.switcher.art()
        except Exception as exc:  # noqa: BLE001 - ModeError, or a systemctl failure: say it, stay in voice
            log.warning("art mode failed: %s", exc)
            return MSG_SWITCH_FAILED.format(why=str(exc).split(";")[0])
        self.owner = speaker
        self.session, self.queue = self._build((speaker or "guest").lower(), self._on_job)
        self.session.rules = self.rules_for(speaker)
        self.session.speaker = speaker or "guest"
        self._resume_recent()
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

    def _resume_recent(self) -> None:
        """Reopen this person's latest project if it was worked on in the last hour.

        So "make it more skeletal" still means the dragon after Jarvis restarts,
        or after an idle exit and "art mode" again.
        """
        store = getattr(self.session, "store", None)
        if store is None:
            return
        try:
            ids = store.projects(self.session.owner)
            if not ids:
                return
            project = store.open(ids[0])
            touched = max((p.stat().st_mtime for p in project.path.rglob("*") if p.is_file()), default=0.0)
        except Exception as exc:  # noqa: BLE001 - a fresh start is fine
            log.info("not resuming a project: %s", exc)
            return
        if self._wall() - touched <= RESUME_WITHIN_S and project.current is not None:
            self.session.project = project
            log.info("resuming art project %s at %s", project.id, project.current)

    def adopt(self) -> None:
        """At daemon start: art mode left on (Jarvis restarted mid-session) still idles back to voice."""
        try:
            if self.active or self.switcher.mode() != "art":
                return
        except Exception:  # noqa: BLE001 - can't tell: leave it to homeai-mode
            return
        log.info("art mode is on from before this start; back to voice after %d idle minutes",
                 round(self.idle_s / 60))
        self._last = self._clock()
        stop = self._stop = threading.Event()

        def watch() -> None:
            while not stop.wait(IDLE_CHECK_S):
                if self.active:
                    return
                if self._clock() - self._last >= self.idle_s:
                    try:
                        self.switcher.voice()
                    except Exception:  # noqa: BLE001
                        log.exception("going back to voice mode failed")
                    return
        threading.Thread(target=watch, name="art-adopt", daemon=True).start()

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
            issues, redrawn = self._check(session, event)
            if redrawn:
                self._publish_state()
                return
            self._export(event, session)        # the session it was drawn in, even if art mode just ended
            grid = list(getattr(session, "grid", []))
            if issues and len(grid) > 1 and event.get("node") in grid:
                self.status = f"Option {grid.index(event['node']) + 1} came out with {', '.join(issues)}."
            else:
                self.status = f"Best I could do: {', '.join(issues)}." if issues else "Done."
        elif event.get("state") == "running":
            self.status = "Drawing..."
        elif event.get("state") in ("failed", "cancelled") and event.get("message"):
            self.status = event["message"]
        self._publish_state()

    def _check(self, session, event: dict) -> tuple[list[str], bool]:
        """(what's wrong, whether it is being redrawn) for a finished picture (ART_FEATURES_PLAN F4).

        A picture whose heads or riders don't match its features is redrawn with a new seed, up to
        MAX_REDRAWS times, before it is shown. Any failure of the checker shows the picture as it is.
        """
        from homeai.art import checker as CK
        from homeai.art import features as F
        project = session.project
        if self.checker is None or project is None or project.id != event.get("project"):
            return [], False
        node = project.nodes.get(event.get("node"))
        if node is None or "features" not in node.spec:
            return [], False
        try:
            checks = CK.checklist(F.features_of(node.spec))
            if not checks:
                return [], False
            self.status = "Checking..."
            self._publish_state()
            issues = CK.problems(Path(event["image"]), checks, self.checker)
        except Exception as exc:  # noqa: BLE001 - an unchecked picture beats no picture
            log.warning("checking %s/%s failed: %s", project.id, node.id, exc)
            self._background(self._warm_voice)
            return [], False
        log.info("art: checked %s/%s: %s", project.id, node.id, ", ".join(issues) or "ok")
        with self._lock:
            if self.session is not session:
                return issues, False
            again = session.checked(node.id, issues) and node.meta.get("tries", 0) < CK.MAX_REDRAWS
            job = session.redraw(node.id, "checked: " + ", ".join(issues)) if again else None
            if job is not None:
                self.status = f"Redrawing: it came out with {', '.join(issues)}."
                return issues, True
        self._background(self._warm_voice)      # the vision model pushed Jarvis's voice model out; bring it back
        return issues, False

    def _export(self, event: dict, session=None) -> None:
        """The picture into the asker's own folder, with its description as JSON beside it."""
        node = None
        session = session or self.session
        project = session.project if session is not None else None
        if project is not None and project.id == event.get("project"):
            node = project.nodes.get(event["node"])
        who = folder_name((node.meta.get("by") if node else None) or (project.info.get("owner") if project else None))
        folder = self.pictures / who
        try:
            folder.mkdir(parents=True, exist_ok=True)
            target = folder / export_name(event["project"], event["node"])
            atomic_copy(event["image"], target)
            if node is not None:
                sidecar = {"project": event["project"], "node": node.id, "parent": node.parent, "by": who,
                           "instruction": node.instruction, "created": node.created, **{
                               k: node.meta.get(k) for k in ("prompt", "model", "seed", "size")},
                           "description": {k: v for k, v in node.spec.items() if k != "seed"}}
                atomic_write_json(target.with_suffix(".json"), sidecar, indent=1)
        except OSError as exc:
            log.warning("could not save the picture to %s: %s", folder, exc)

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
        node = project.nodes.get(project.current)
        if node is not None:
            from homeai.art import features as F
            try:
                out["features"] = F.dump(F.features_of(node.spec))
            except ValueError:
                out["features"] = []
            out["changed"] = list(getattr(session, "last_changed", []))
        grid = [n for n in getattr(session, "grid", []) if n in project.nodes]
        if len(grid) > 1:
            out["options"] = []
            for number, node_id in enumerate(grid, 1):
                image = project.image(node_id)
                out["options"].append({"n": number, "node": node_id, "state": project.nodes[node_id].state,
                                       "image": str(image) if image.exists() else None,
                                       "problems": list(project.nodes[node_id].meta.get("checked") or [])})
        return out

    def _publish_state(self) -> None:
        try:
            self._publish("art", self.state())
        except Exception:  # noqa: BLE001 - the window is a nicety; drawing carries on
            log.warning("art state publish failed", exc_info=True)


def _warm_voice_model() -> None:
    """Load the voice model again after a check (Ollama holds one model): ~1.9 s, off the speaking path."""
    from homeai import httpjson
    from homeai.art.patcher import MODEL
    try:
        httpjson.ollama_chat(MODEL, [{"role": "user", "content": "ok"}], keep_alive=-1, options={"num_predict": 1},
                             timeout=120)
    except (OSError, KeyError, ValueError) as exc:
        log.warning("could not reload the voice model: %s", exc)


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
