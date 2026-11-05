#!/usr/bin/env python3
"""Can each image model run here, beside the voice model? (AVATAR_PLAN Phase 0)

Why: art mode depends on ComfyUI working on ROCm with Jarvis's voice model still
loaded. For each model this makes one cold and three warm images and records
seconds, peak VRAM and any error. The coder is stopped during the run and started
again after. Images and results.json go to ~/homeai-bench/comfy-smoke-<time>/.

Usage:
    .venv/bin/python tools/art/comfy_smoke.py [--model klein --model sdxl] [--warm 3]
"""
from __future__ import annotations

import argparse
import json
import shutil
import sys
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from comfy_common import MODELS, Comfy, CoderStopped, VramPeak, vram_used_gb, workflow  # noqa: E402

PROMPT = "a watercolour fox reading a book under a lantern, warm evening light"


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--model", action="append", choices=MODELS)
    ap.add_argument("--warm", type=int, default=3)
    args = ap.parse_args()
    out = Path.home() / "homeai-bench" / f"comfy-smoke-{datetime.now():%Y%m%d-%H%M%S}"
    out.mkdir(parents=True)
    results = {"baseline_vram_gb": None, "models": {}}
    with CoderStopped():
        results["baseline_vram_gb"] = vram_used_gb()
        print("VRAM with the coder stopped:", results["baseline_vram_gb"], flush=True)
        for model in args.model or MODELS:
            runs = []
            with Comfy(out / f"comfy-{model}.log") as comfy, VramPeak() as peak:
                for i in range(1 + args.warm):
                    run = comfy.run(workflow(model, PROMPT, seed=7 + i, prefix=f"smoke-{model}"))
                    runs.append(run)
                    print(model, "cold" if i == 0 else "warm", run, flush=True)
                    if "image" in run:
                        shutil.copy(run["image"], out / f"{model}-{i}.png")
                    else:
                        break
            results["models"][model] = {"runs": runs, "peak_vram_gb": peak.peak}
            (out / "results.json").write_text(json.dumps(results, indent=1))
    print(out)
    return 0 if all("error" not in r for m in results["models"].values() for r in m["runs"]) else 1


if __name__ == "__main__":
    raise SystemExit(main())
