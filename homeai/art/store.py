"""Art projects as a tree of nodes on disk (AVATAR_PLAN §7.3, 6a step 5).

Layout: ``<root>/<project-id>/project.json`` plus ``nodes/NNN.json`` (sidecar)
and ``nodes/NNN.png`` (the picture). A node's parent is the picture it was made
from; "go back" moves ``current`` to an ancestor, and a new instruction from
there starts a branch, so nothing is ever overwritten.

Every write is atomic (temp file + rename). A sidecar that can't be read is
listed in ``Project.broken`` and skipped, never fatal. A node left ``pending`` by
a crash is found by ``pending()`` so the queue can resume or fail it.
Standard library only.
"""
from __future__ import annotations

import json
import os
import re
import shutil
import tempfile
import time
from dataclasses import dataclass, field
from pathlib import Path

SCHEMA = 1
ROOT = Path.home() / ".local/share/homeai/art/projects"
STATES = ("pending", "done", "failed")


class StoreError(Exception):
    pass


def write_json(path: Path, data: dict) -> None:
    """Write ``data`` so a crash leaves either the old file or the new one, never half of either."""
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=path.parent, prefix=f".{path.name}.", suffix=".tmp")
    try:
        with os.fdopen(fd, "w") as handle:
            json.dump(data, handle, indent=1)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(tmp, path)
    except BaseException:
        Path(tmp).unlink(missing_ok=True)
        raise


def slug(text: str, limit: int = 30) -> str:
    words = re.sub(r"[^a-z0-9]+", "-", text.lower()).strip("-")
    return words[:limit].rstrip("-") or "untitled"


@dataclass
class Node:
    id: str
    parent: str | None
    instruction: str
    spec: dict
    state: str = "pending"
    kind: str = "image"
    created: float = 0.0
    deleted: bool = False
    meta: dict = field(default_factory=dict)

    def to_json(self) -> dict:
        return {"id": self.id, "parent": self.parent, "instruction": self.instruction, "spec": self.spec,
                "state": self.state, "kind": self.kind, "created": self.created, "deleted": self.deleted,
                "meta": self.meta}

    @classmethod
    def from_json(cls, data: dict) -> Node:
        node = cls(id=str(data["id"]), parent=data.get("parent"), instruction=str(data.get("instruction", "")),
                   spec=dict(data.get("spec") or {}), state=data.get("state", "pending"),
                   kind=data.get("kind", "image"), created=float(data.get("created", 0.0)),
                   deleted=bool(data.get("deleted", False)), meta=dict(data.get("meta") or {}))
        if node.state not in STATES:
            raise ValueError(f"unknown state {node.state!r}")
        return node


class Project:
    def __init__(self, path: Path, info: dict, nodes: dict[str, Node], broken: list[str]) -> None:
        self.path, self.info, self.nodes, self.broken = path, info, nodes, broken

    # -- reading -------------------------------------------------------------

    @property
    def id(self) -> str:
        return self.info["id"]

    @property
    def current(self) -> str | None:
        return self.info.get("current")

    def image(self, node_id: str) -> Path:
        return self.path / "nodes" / f"{node_id}.png"

    def live(self) -> list[Node]:
        return [n for n in self.nodes.values() if not n.deleted]

    def lineage(self, node_id: str | None = None) -> list[Node]:
        """From the root down to ``node_id`` (default: current)."""
        chain, seen = [], set()
        at = node_id if node_id is not None else self.current
        while at is not None and at in self.nodes and at not in seen:
            seen.add(at)
            chain.append(self.nodes[at])
            at = self.nodes[at].parent
        return list(reversed(chain))

    def children(self, node_id: str) -> list[Node]:
        return sorted((n for n in self.live() if n.parent == node_id), key=lambda n: n.id)

    def pending(self) -> list[Node]:
        return [n for n in self.live() if n.state == "pending"]

    def find(self, words: str) -> Node | None:
        """The newest live node whose instruction contains ``words`` (case ignored)."""
        want = words.lower().strip()
        hits = [n for n in self.live() if want and want in n.instruction.lower()]
        return max(hits, key=lambda n: n.id) if hits else None

    # -- changing ------------------------------------------------------------

    def _save_info(self) -> None:
        write_json(self.path / "project.json", self.info)

    def _save_node(self, node: Node) -> None:
        write_json(self.path / "nodes" / f"{node.id}.json", node.to_json())

    def add(self, instruction: str, spec: dict, parent: str | None = "current", kind: str = "image",
            meta: dict | None = None) -> Node:
        """A new pending node under ``parent`` (default: current), which becomes current."""
        parent_id = self.current if parent == "current" else parent
        if parent_id is not None and parent_id not in self.nodes:
            raise StoreError(f"no node {parent_id} in {self.id}")
        number = max((int(k) for k in self.nodes if k.isdigit()), default=0) + 1
        node = Node(id=f"{number:03d}", parent=parent_id, instruction=instruction, spec=spec, kind=kind,
                    created=time.time(), meta=dict(meta or {}))
        self._save_node(node)
        self.nodes[node.id] = node
        self.info["current"] = node.id
        self._save_info()
        return node

    def finish(self, node_id: str, image: Path, **meta) -> Node:
        """Copy ``image`` in as the node's picture and mark it done."""
        node = self._node(node_id)
        target = self.image(node_id)
        tmp = target.with_suffix(".png.tmp")
        shutil.copyfile(image, tmp)
        os.replace(tmp, target)
        node.state, node.meta = "done", {**node.meta, **meta}
        self._save_node(node)
        return node

    def fail(self, node_id: str, reason: str) -> Node:
        node = self._node(node_id)
        node.state, node.meta = "failed", {**node.meta, "error": reason[:500]}
        self._save_node(node)
        return node

    def back(self, steps: int = 1) -> Node:
        """Move current up ``steps`` live ancestors; stops at the first picture."""
        if steps < 1:
            raise StoreError("go back at least one step")
        chain = [n for n in self.lineage() if not n.deleted]
        if not chain:
            raise StoreError("there is nothing to go back to")
        target = chain[max(0, len(chain) - 1 - steps)]
        self.select(target.id)
        return target

    def select(self, node_id: str) -> Node:
        node = self._node(node_id)
        if node.deleted:
            raise StoreError(f"{node_id} is deleted")
        self.info["current"] = node_id
        self._save_info()
        return node

    def pin(self, node_id: str, on: bool = True) -> None:
        self._node(node_id)
        pins = [p for p in self.info.get("pins", []) if p != node_id]
        self.info["pins"] = pins + [node_id] if on else pins
        self._save_info()

    def delete(self, node_id: str, undo: bool = False) -> None:
        """Soft delete (or undo it). If current is deleted, current moves to its nearest live ancestor."""
        node = self._node(node_id)
        node.deleted = not undo
        self._save_node(node)
        if node.deleted and self.current == node_id:
            alive = [n for n in self.lineage(node_id) if not n.deleted]
            self.info["current"] = alive[-1].id if alive else None
            self._save_info()

    def _node(self, node_id: str) -> Node:
        if node_id not in self.nodes:
            raise StoreError(f"no node {node_id} in {self.id}")
        return self.nodes[node_id]


class Store:
    def __init__(self, root: Path = ROOT, clock=time.time) -> None:
        self.root, self._clock = root, clock

    def create(self, kind: str, owner: str, title: str = "") -> Project:
        stamp = time.strftime("%Y%m%d-%H%M%S", time.localtime(self._clock()))
        base = f"{stamp}-{slug(title or kind)}"
        path, n = self.root / base, 1
        while path.exists():
            n += 1
            path = self.root / f"{base}-{n}"
        info = {"schema": SCHEMA, "id": path.name, "kind": kind, "owner": owner, "title": title,
                "created": self._clock(), "current": None, "pins": [], "notes": ""}
        (path / "nodes").mkdir(parents=True)
        write_json(path / "project.json", info)
        return Project(path, info, {}, [])

    def open(self, project_id: str) -> Project:
        path = self.root / project_id
        try:
            info = json.loads((path / "project.json").read_text())
        except FileNotFoundError:
            raise StoreError(f"no project {project_id}") from None
        except ValueError as exc:
            raise StoreError(f"{project_id}/project.json is damaged: {exc}") from exc
        if not isinstance(info, dict) or info.get("schema") != SCHEMA:
            raise StoreError(f"{project_id} has schema {info.get('schema') if isinstance(info, dict) else '?'}; "
                             f"this version reads {SCHEMA}")
        nodes, broken = {}, []
        for sidecar in sorted((path / "nodes").glob("*.json")):
            try:
                node = Node.from_json(json.loads(sidecar.read_text()))
            except (ValueError, KeyError, TypeError) as exc:
                broken.append(f"{sidecar.name}: {exc}")
                continue
            nodes[node.id] = node
        if info.get("current") not in nodes:
            info["current"] = None
        return Project(path, info, nodes, broken)

    def projects(self, owner: str | None = None) -> list[str]:
        """Project ids, newest first, optionally only ``owner``'s."""
        found = []
        for meta in self.root.glob("*/project.json"):
            try:
                info = json.loads(meta.read_text())
            except ValueError:
                continue
            if owner is None or info.get("owner") == owner:
                found.append((info.get("created", 0), meta.parent.name))
        return [name for _, name in sorted(found, reverse=True)]
