#!/usr/bin/env python3
"""Print the loaders, samplers and settings inside ComfyUI's official workflow templates.

Why: each image model has its own right sampler, steps and CFG, and a wrong setting
makes a good model look bad in the bake-off. The templates are the reference.

Usage:
    .venv/bin/python tools/art/inspect_template.py image_flux3_klein_t2i image_z_image_turbo
"""
from __future__ import annotations

import glob
import json
import sys
from pathlib import Path

ROOT = Path.home() / ".local/share/homeai/comfyui/venv/lib/python3.12/site-packages/comfyui_workflow_templates_json"
SKIP = {"Note", "MarkdownNote", "PrimitiveNode", "Reroute", "SaveImage", "PreviewImage"}


def nodes(graph: dict):
    yield from graph.get("nodes", [])
    for sub in (graph.get("definitions") or {}).get("subgraphs", []):
        yield from sub.get("nodes", [])


def main() -> int:
    for name in sys.argv[1:]:
        found = glob.glob(str(ROOT / "**" / f"{name}.json"), recursive=True)
        if not found:
            print("==", name, "not found")
            continue
        graph = json.load(open(found[0]))
        print("==", name)
        for node in nodes(graph):
            kind = node.get("type", "")
            if kind in SKIP or len(kind) > 30 and "-" in kind:
                continue
            values = [str(v)[:90] for v in (node.get("widgets_values") or []) if not isinstance(v, (dict, list))]
            print("  ", kind, values)
            for model in (node.get("properties") or {}).get("models", []):
                print("      ", model.get("directory"), model.get("url"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
