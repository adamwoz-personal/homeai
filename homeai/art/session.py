"""One person's drawing session: words in, pictures queued, a short sentence back (AVATAR_PLAN 6a step 7).

Joins the pieces: grammar (what was asked), patcher (the voice model's change
to the spec), policy (allowed?), store (the picture tree) and the job queue.
Not wired into the voice daemon yet: that is step 8, after Adam's look.

Replies are short because they are spoken. Pictures arrive later through the
queue's events; ``on_event`` checks each finished picture against the policy
before it may be shown.
"""
from __future__ import annotations

import random
import re
from dataclasses import dataclass, field

from homeai.art import grammar, policy, prompt_spec, workflows
from homeai.art import features as F
from homeai.art.patcher import MSG_NOT_CAUGHT, features_patch_for
from homeai.art.store import StoreError

MSG_NOTHING_YET = "There's no picture yet. Say draw, and what you'd like."
MSG_NOT_YET = "I can't do that one yet."
MSG_NOTHING_TO_STOP = "Nothing is drawing."
MSG_NO_GRID = "There's no set of options to pick from."
MSG_FRESH = "Okay, a new picture. What shall I draw?"
MSG_QUESTION = "I can't answer questions about the picture yet, but I can change it."

# A question is never a change: "what is image three?" changed four features (Adam, 2026-11-07).
QUESTION = re.compile(r"^(what|who|when|where|why|how|which|is (it|that|this|there)|are (they|there|those)|"
                      r"does (it|that)|did (it|you))\b")
# Defects a complaint names; a patch that writes them into the features asks for them again
# ("a single flying dragon with two heads, one on each side", 2026-11-07).
DEFECT = re.compile(r"\b(two|three|four|several|multiple|many|extra|second|third|2|3|4) "
                    r"(heads?|riders?|samurais?|legs?|arms?|tails?|wings?|hands?|people|faces?)\b")
TRIES_AFTER_COMPLAINT = 4


LEAD = re.compile(r"^(?:let'?s )?(?:draw|paint|sketch|generate|create|make)(?: me)? "
                  r"(?:(?:a )?(?:picture|image|drawing|painting|sketch) of )?")


def drawing_sentence(said: str) -> str:
    """Say back what was asked in the asker's words: "Drawing a dragon with a samurai on its back."

    The spec's subject alone came out as "Drawing dragon." (Adam's first session, 2026-11-07).
    """
    what = LEAD.sub("", said).strip()
    return f"Drawing {what}." if what and what != said else "Drawing it now."


MODEL = "klein9"


def render_spec(spec: dict, steer: list[str] | tuple = ()):
    """The prompt for a stored spec: lettered features (plus hidden policy words), or the older PromptSpec fields."""
    if "features" in spec:
        return F.render(F.features_of(spec), spec.get("aspect") or "square", steer)
    return prompt_spec.render(spec)


def title_of(spec: dict) -> str:
    if "features" in spec:
        main = next((f for f in F.features_of(spec) if f.kind == "subject"), None)
        return main.text if main else "picture"
    return spec.get("subject") or "picture"


@dataclass
class Reply:
    spoken: str
    kind: str
    jobs: list[str] = field(default_factory=list)


class ArtSession:
    def __init__(self, store, queue, owner: str, rules=None, patch=features_patch_for, rng=None,
                 kind: str = "general"):
        self.store, self.queue, self.owner = store, queue, owner
        self.rules = rules or policy.load("allow-all")
        self._patch, self._rng, self.kind = patch, rng or random.Random(), kind
        self.project = None
        self.grid: list[str] = []
        self.picked = False             # the person chose an option: don't move them off it
        self.jobs: list[str] = []
        self.speaker: str | None = None     # who asked for the next picture (set per sentence by ArtVoice)
        self.last_changed: list[str] = []    # feature letters the last change touched (for the window)

    # -- helpers ---------------------------------------------------------------

    def _seed(self) -> int:
        return self._rng.randrange(2**32)

    def _current(self):
        if self.project is None or self.project.current is None:
            return None
        return self.project.nodes[self.project.current]

    def _draw(self, instruction: str, spec: dict, parent: str | None, preview: bool = False,
              meta: dict | None = None) -> str:
        spec = {**spec, "seed": spec.get("seed") if spec.get("seed") is not None else self._seed()}
        steer = getattr(self.rules, "steer", None)
        rendered = render_spec(spec, steer(F.text_of(spec)) if steer and "features" in spec else ())
        node = self.project.add(instruction, spec, parent=parent,
                                meta={"prompt": rendered.text, "model": MODEL, "seed": spec["seed"],
                                      "size": [rendered.width, rendered.height], "by": self.speaker or self.owner,
                                      **(meta or {})})
        graph = workflows.klein9(rendered.text, spec["seed"], rendered.width, rendered.height,
                                 prefix=f"art/{self.project.id}")
        job = self.queue.submit(self.project.id, node.id, graph,
                                priority=0 if preview else 1)
        self.jobs.append(job.id)
        return job.id

    def _allowed(self, text: str, spec: dict | None) -> tuple[policy.Decision, dict | None]:
        verdict = self.rules.check_request(text, self.owner)
        if not verdict.allowed or spec is None:
            return verdict, spec
        verdict = self.rules.check_spec(spec, self.owner)
        if verdict.verdict == "rewrite" and verdict.spec is not None:
            spec = verdict.spec if "features" in verdict.spec else prompt_spec.validate(verdict.spec)
        return verdict, spec

    # -- the one entry point ---------------------------------------------------

    def handle(self, text: str) -> Reply | None:
        """None if ``text`` isn't about drawing, so the caller passes it to Jarvis as usual."""
        intent = grammar.parse(text)
        if intent is None:
            return None
        handler = getattr(self, f"_on_{intent.kind}", None)
        if handler is None:
            return Reply(MSG_NOT_YET, intent.kind)
        return handler(intent, text)

    def handle_free(self, text: str) -> Reply:
        """A sentence about the picture with no command shape: a new picture, or a change to this one."""
        said = grammar._tidy(text)
        if QUESTION.match(said):
            return Reply(MSG_QUESTION, "question")
        kind = "edit" if self._current() is not None else "new"
        return getattr(self, f"_on_{kind}")(grammar.Intent(kind, text=said), text)

    def _on_new(self, intent, text):
        result = self._patch({}, intent.text)
        if not result.ok:
            return Reply(MSG_NOT_CAUGHT, "new")
        verdict, spec = self._allowed(text, result.spec)
        if not verdict.allowed:
            return Reply(verdict.reason, "denied")
        self.last_changed = list(result.changed)
        self.project = self.store.create(self.kind, (self.speaker or self.owner).lower(), title_of(spec))
        self.grid = []
        return Reply(drawing_sentence(intent.text), "new", [self._draw(intent.text, spec, None)])

    def _on_edit(self, intent, text):
        node = self._current()
        if node is None:
            return Reply(MSG_NOTHING_YET, "edit")
        result = self._patch(node.spec, intent.text)
        if not result.ok:
            return Reply(MSG_NOT_CAUGHT, "edit")
        verdict, spec = self._allowed(text, result.spec)
        if not verdict.allowed:
            return Reply(verdict.reason, "denied")
        self.last_changed = list(result.changed)
        if "features" in spec:
            said = F.describe(result.changed, F.features_of(node.spec), F.features_of(spec))
        else:
            said = f"Changing {prompt_spec.describe(prompt_spec.diff(node.spec, {**spec, 'seed': node.spec.get('seed')}))}."
        if grammar.is_complaint(intent.text):
            if "features" in spec and DEFECT.search(F.text_of(spec)) and not DEFECT.search(F.text_of(node.spec)):
                spec, self.last_changed = {**node.spec}, []         # don't write the defect in; just re-roll
            # The same seed keeps the same broken layout (two heads stayed through four edits, 2026-11-07):
            # draw several new tries and let the person pick.
            jobs, self.grid, self.picked = [], [], False
            for _ in range(TRIES_AFTER_COMPLAINT):
                jobs.append(self._draw(intent.text, {**spec, "seed": None}, node.id))
                self.grid.append(self.project.current)
            self.project.select(self.grid[0])
            return Reply(f"Fixing it: {TRIES_AFTER_COMPLAINT} new tries. Say which one you like.", "edit", jobs)
        spec = {**spec, "seed": node.spec.get("seed")}      # same seed: only the change should change
        return Reply(said, "edit", [self._draw(intent.text, spec, node.id)])

    def checked(self, node_id: str, issues: list[str]) -> bool:
        """Record what the check found; True if this picture should be redrawn (F4 + F5 follow-up).

        Alone, a wrong picture is redrawn. Among numbered options, it isn't while another option is
        still drawing or came out right: the person picks from those instead, and 4 options x 3
        redraws would keep the GPU busy for over a minute. A right option replaces a wrong one on screen.
        """
        if self.project is None or node_id not in self.project.nodes:
            return False
        self.project.note(node_id, checked=list(issues))
        grid = [n for n in self.grid if n in self.project.nodes]
        if node_id not in grid or len(grid) < 2:
            return bool(issues)
        if issues:
            others = [self.project.nodes[n] for n in grid if n != node_id]
            return not any(n.state != "failed" and not n.meta.get("checked") for n in others)
        shown = self.project.nodes.get(self.project.current)
        if not self.picked and shown is not None and shown.id in grid and shown.meta.get("checked"):
            self.project.select(node_id)
        return False

    def redraw(self, node_id: str, reason: str) -> str | None:
        """A checked picture came out wrong: mark it failed and draw it again with a new seed (F4).

        The new try takes the old one's place (current, and its slot among numbered options).
        Returns the new job id, or None if the picture isn't in this project any more.
        """
        if self.project is None or node_id not in self.project.nodes:
            return None
        node = self.project.nodes[node_id]
        before = self.project.current
        self.project.fail(node_id, reason)
        job = self._draw(node.instruction, {**node.spec, "seed": None}, node.parent,
                         meta={"tries": node.meta.get("tries", 0) + 1, "redraw_of": node_id,
                               **({"by": node.meta["by"]} if node.meta.get("by") else {})})
        new = self.project.current
        self.grid = [new if n == node_id else n for n in self.grid]
        if before != node_id and before in self.project.nodes:
            self.project.select(before)         # the person had moved on: don't pull them back
        return job

    def _on_fresh(self, intent, text):
        self.project, self.grid = None, []
        return Reply(MSG_FRESH, "fresh")

    def _on_again(self, intent, text):
        node = self._current()
        if node is None:
            return Reply(MSG_NOTHING_YET, "again")
        spec = {**node.spec, "seed": None}
        return Reply("Another one.", "again", [self._draw("again", spec, node.parent)])

    def _on_variations(self, intent, text):
        node = self._current()
        if node is None:
            return Reply(MSG_NOTHING_YET, "variations")
        jobs, self.grid, self.picked = [], [], False
        for _ in range(intent.n):
            jobs.append(self._draw(f"option of {node.id}", {**node.spec, "seed": None}, node.parent))
            self.grid.append(self.project.current)
        self.project.select(self.grid[0])
        return Reply(f"Drawing {intent.n} options.", "variations", jobs)

    def _on_select(self, intent, text):
        if not self.grid:
            return Reply(MSG_NO_GRID, "select")
        index = intent.n - 1 if intent.n > 0 else len(self.grid) - 1
        if not 0 <= index < len(self.grid):
            return Reply(f"There are only {len(self.grid)}.", "select")
        self.project.select(self.grid[index])
        self.picked = True
        return Reply("That one.", "select")

    def _on_back(self, intent, text):
        if self._current() is None:
            return Reply(MSG_NOTHING_YET, "back")
        try:
            node = self.project.back(intent.n)
        except StoreError as exc:
            return Reply(str(exc).capitalize() + ".", "back")
        return Reply(f"Back to: {node.instruction}.", "back")

    def _on_keep(self, intent, text):
        node = self._current()
        if node is None:
            return Reply(MSG_NOTHING_YET, "keep")
        self.project.pin(node.id)
        if intent.name:
            return Reply("Kept. Saving it by name comes later.", "keep")
        return Reply("Kept.", "keep")

    def _on_what(self, intent, text):
        node = self._current()
        if node is None:
            return Reply(MSG_NOTHING_YET, "what")
        return Reply(node.meta.get("prompt") or render_spec(node.spec).text, "what")

    def _on_cancel(self, intent, text):
        stopped = [job for job in self.jobs if self.queue.cancel(job)]
        return Reply("Stopped." if stopped else MSG_NOTHING_TO_STOP, "cancel", stopped)

    def _on_leave(self, intent, text):
        return Reply("Done drawing.", "leave")

    # -- pictures arriving -----------------------------------------------------

    def on_event(self, event: dict) -> str | None:
        """For a finished picture, the sentence to say if the policy stops it; None if it may be shown."""
        if event.get("state") != "done" or self.project is None or event.get("project") != self.project.id:
            return None
        verdict = self.rules.check_image(event.get("image", ""), self.owner)
        if verdict.allowed:
            return None
        try:
            self.store.open(event["project"]).fail(event["node"], verdict.reason)
        except StoreError:
            pass
        return verdict.reason
