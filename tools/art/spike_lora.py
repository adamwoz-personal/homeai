#!/usr/bin/env python3
"""Can this box train a klein LoRA at all? Peak VRAM, peak RAM, seconds a step (ART_FEATURES_PLAN F9 spike, S2-S3).

Why: Adam wants klein to learn what wording can't teach ("self train"). A LoRA is a small add-on file trained on
pictures; the base model stays as it is. BFL's own minimum for 9B (22 GB VRAM, 64 GB RAM) is past this box
(20 GB, 31 GB), so the spike starts with 4B, whose files on disk musubi-tuner can read as they are
(tools/art/inspect_safetensors.py). Captions are the plain words ("... fighting a chimera"), so a LoRA that works
draws the right chimera without the glossary's description.

The GPU is the trainer's: from voice mode, the coder and Jarvis stop and the voice model unloads
(``coder_paused(voice_off=True)``); afterwards voice mode comes back whatever happened. The trainer runs in a
user scope capped at ``--ram-gb`` with no swap, so a runaway kills the trainer, not ComfyUI or Jarvis.

Writes ~/homeai-bench/lora-spike-<time>/{dataset.toml,data.jsonl,*.log,result.json,lora/}.

Usage:
    .venv/bin/python tools/art/spike_lora.py [--model 4b] [--steps 50] [--res 512] [--ram-gb 16] [--dry-run]
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
import threading
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

from homeai.benchio import bench_path  # noqa: E402

MUSUBI = Path.home() / ".local/share/homeai/musubi"
MODELS = Path.home() / ".local/share/homeai/comfyui/app/models"
FILES = {
    "4b": {"version": "klein-base-4b", "dit": MODELS / "diffusion_models/flux-2-klein-base-4b.safetensors",
           "te": MODELS / "text_encoders/qwen_3_4b.safetensors"},
}
VAE = MODELS / "vae/flux2-vae.safetensors"
BENCH = Path.home() / "homeai-bench/wording-20261118-195631"
# Judged right by eye (lion's face, goat's head apart on the back, snake's head on the tail, one blade).
PICTURES = ["v6-1", "v6-2", "v6-4", "v6-5", "v6-7", "v6-8", "v7-1", "v7-2", "v7-4", "v7-8"]
CAPTION = "An anime girl wielding a katana fighting a chimera."
VRAM_USED = Path("/sys/class/drm/card1/device/mem_info_vram_used")      # the 7900 XT (card0 is the iGPU)


def dataset(out: Path, res: int) -> Path:
    """The dataset config musubi reads: one jsonl of picture + plain caption."""
    rows = [{"image_path": str(BENCH / f"{name}.png"), "caption": CAPTION} for name in PICTURES]
    (out / "data.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))
    toml = (f"[general]\nresolution = [{res}, {res}]\nbatch_size = 1\nenable_bucket = true\n"
            f"bucket_no_upscale = false\n\n[[datasets]]\nimage_jsonl_file = \"{out / 'data.jsonl'}\"\n"
            f"cache_directory = \"{out / 'cache'}\"\nnum_repeats = 1\n")
    (out / "dataset.toml").write_text(toml)
    return out / "dataset.toml"


def capped(cmd: list[str], ram_gb: int) -> list[str]:
    """``cmd`` in a user scope: RAM capped, no swap, so an overrun kills only the trainer; only the 7900 XT seen.

    2026-11-18 first run: the iGPU counted as a second card, accelerate started two trainers each loading the
    model, and the box ran out of RAM before the 22 GB cap (it needs to sit under what's free, ~20 GB with
    voice mode up), so the kernel's global OOM killer picked the victim."""
    return ["systemd-run", "--user", "--scope", "-q", "-p", f"MemoryMax={ram_gb}G", "-p", "MemorySwapMax=0",
            "env", "HIP_VISIBLE_DEVICES=0", *cmd]


def commands(model: str, toml: Path, out: Path, steps: int) -> list[tuple[str, list[str]]]:
    f = FILES[model]
    py, src = str(MUSUBI / "venv/bin/python"), MUSUBI / "src/src/musubi_tuner"
    common = ["--dataset_config", str(toml), "--model_version", f["version"]]
    return [
        ("cache_latents", [py, str(src / "flux_2_cache_latents.py"), *common, "--vae", str(VAE),
                           "--vae_dtype", "bfloat16"]),
        ("cache_text", [py, str(src / "flux_2_cache_text_encoder_outputs.py"), *common, "--text_encoder",
                        str(f["te"]), "--batch_size", "1"]),
        ("train", [str(MUSUBI / "venv/bin/accelerate"), "launch", "--num_processes", "1",
                   "--num_cpu_threads_per_process", "1",
                   "--mixed_precision", "bf16", str(src / "flux_2_train_network.py"), *common,
                   "--dit", str(f["dit"]), "--vae", str(VAE), "--text_encoder", str(f["te"]), "--sdpa",
                   "--mixed_precision", "bf16", "--timestep_sampling", "flux2_shift", "--weighting_scheme", "none",
                   "--optimizer_type", "AdamW", "--learning_rate", "1e-4", "--gradient_checkpointing",
                   "--max_data_loader_n_workers", "1", "--network_module", "networks.lora_flux_2",
                   "--network_dim", "16", "--max_train_steps", str(steps), "--seed", "42",
                   "--output_dir", str(out / "lora"), "--output_name", f"chimera-{model}"]),
    ]


class Peaks:
    """Highest VRAM in use and lowest RAM available, sampled every 0.5 s while a step runs."""

    def __init__(self) -> None:
        self.vram_gb, self.ram_avail_gb, self._stop = 0.0, 1e9, threading.Event()

    @staticmethod
    def _avail_gb() -> float:
        for line in Path("/proc/meminfo").read_text().splitlines():
            if line.startswith("MemAvailable:"):
                return int(line.split()[1]) / 1e6
        return 0.0

    def _run(self) -> None:
        while not self._stop.is_set():
            try:
                self.vram_gb = max(self.vram_gb, int(VRAM_USED.read_text()) / 1e9)
            except (OSError, ValueError):
                pass
            self.ram_avail_gb = min(self.ram_avail_gb, self._avail_gb())
            self._stop.wait(0.5)

    def __enter__(self):
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()
        return self

    def __exit__(self, *exc) -> None:
        self._stop.set()
        self._thread.join()


def run_steps(steps: list[tuple[str, list[str]]], out: Path, ram_gb: int) -> list[dict]:
    results = []
    for name, cmd in steps:
        start = time.monotonic()
        with Peaks() as peaks, open(out / f"{name}.log", "w") as log:
            code = subprocess.call(capped(cmd, ram_gb), stdout=log, stderr=subprocess.STDOUT,
                                   cwd=MUSUBI / "src", timeout=3600)
        results.append({"step": name, "exit": code, "seconds": round(time.monotonic() - start, 1),
                        "peak_vram_gb": round(peaks.vram_gb, 2), "min_ram_available_gb": round(peaks.ram_avail_gb, 2)})
        print(json.dumps(results[-1]), flush=True)
        if code:
            break
    return results


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--model", choices=sorted(FILES), default="4b")
    ap.add_argument("--steps", type=int, default=50)
    ap.add_argument("--res", type=int, default=512)
    ap.add_argument("--ram-gb", type=int, default=16)
    ap.add_argument("--dry-run", action="store_true", help="write the config and print the commands only")
    args = ap.parse_args(argv)
    out = bench_path("lora-spike")
    out.mkdir(parents=True)
    steps = commands(args.model, dataset(out, args.res), out, args.steps)
    if args.dry_run:
        for name, cmd in steps:
            print(name, " ".join(capped(cmd, args.ram_gb)))
        return 0
    sys.path.insert(0, str(REPO / "tools/art"))
    from comfy_common import CoderStopped
    from homeai import gpu_mode as G
    switcher = G.ModeSwitcher(G.System(), log=lambda line: print("  ", line, flush=True))
    if switcher.mode() == "art":
        switcher.voice()
    with CoderStopped(voice_off=True):
        results = run_steps(steps, out, args.ram_gb)
    (out / "result.json").write_text(json.dumps({"model": args.model, "steps": args.steps, "res": args.res,
                                                 "pictures": PICTURES, "caption": CAPTION, "runs": results},
                                                indent=1))
    print(out)
    return 0 if results and all(r["exit"] == 0 for r in results) else 1


if __name__ == "__main__":
    raise SystemExit(main())
