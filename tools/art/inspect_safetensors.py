#!/usr/bin/env python3
"""What's inside a .safetensors file: tensor count, dtypes, parameter count, and sample key names.

Why: the LoRA spike (ART_FEATURES_PLAN F9) needs to know whether musubi-tuner can load the klein files ComfyUI
already has (ComfyUI repacks: fp8 with scale tensors, maybe renamed keys) or whether BFL's originals must be
downloaded (~34 GB). Reads only the JSON header, so it's instant and touches neither GPU nor much RAM.

Usage:
    .venv/bin/python tools/art/inspect_safetensors.py FILE [FILE ...] [--keys 8] [--grep scale]
"""
from __future__ import annotations

import argparse
import json
import re
import struct
from collections import Counter
from math import prod
from pathlib import Path


def header(path: Path) -> dict:
    with open(path, "rb") as f:
        size = struct.unpack("<Q", f.read(8))[0]
        return json.loads(f.read(size))


def summary(path: Path, keys: int = 8, grep: str | None = None) -> dict:
    head = header(path)
    meta = head.pop("__metadata__", {}) or {}
    dtypes = Counter(t["dtype"] for t in head.values())
    params = sum(prod(t["shape"]) for t in head.values())
    names = sorted(head)
    picked = [n for n in names if re.search(grep, n)] if grep else names
    return {"file": path.name, "gb": round(path.stat().st_size / 1e9, 2), "tensors": len(head),
            "params_b": round(params / 1e9, 2), "dtypes": dict(dtypes), "metadata_keys": sorted(meta)[:10],
            "matching": len(picked), "sample": picked[:keys]}


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("files", type=Path, nargs="+")
    ap.add_argument("--keys", type=int, default=8)
    ap.add_argument("--grep")
    args = ap.parse_args(argv)
    for path in args.files:
        print(json.dumps(summary(path, args.keys, args.grep), indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
