"""Shared by the art tools: start ComfyUI, build the three models' workflows, run them, watch VRAM.

Why: the smoke test, contention bench and bake-off all need the same runner. Phase 6
replaces this with homeai/art/comfy.py once the bake-off has picked what to keep.
ComfyUI binds to 127.0.0.1 only. It runs in its own cgroup with a hard memory cap: on
2026-11-05 it held every model in RAM, reached 26.8 GB of 29 GB, and the OOM killer took
Adam's terminal with it. Over the cap, only ComfyUI dies. Use one server per model.
"""
from __future__ import annotations

import json
import os
import subprocess
import threading
import time
import urllib.error
import urllib.request
from pathlib import Path

ROOT = Path.home() / ".local/share/homeai/comfyui"
URL = "http://127.0.0.1:8188"
OLLAMA = "http://127.0.0.1:11434"
MODELS = ("klein", "zimage", "sdxl")
MEMORY_CAP = "18G"
NEGATIVE = "blurry, low quality, watermark, text artefacts, deformed"


def workflow(model: str, prompt: str, seed: int, width: int = 1024, height: int = 1024, prefix: str = "art") -> dict:
    """ComfyUI API-format graph for ``model``. Node ids are fixed, so tests can check the wiring."""
    save = {"class_type": "SaveImage", "inputs": {"images": ["decode", 0], "filename_prefix": prefix}}
    decode = {"class_type": "VAEDecode", "inputs": {"samples": ["sample", 0], "vae": ["vae", 0]}}
    pos = {"class_type": "CLIPTextEncode", "inputs": {"text": prompt, "clip": ["clip", 0]}}
    if model == "sdxl":
        return {
            "ckpt": {"class_type": "CheckpointLoaderSimple", "inputs": {"ckpt_name": "sd_xl_base_1.0.safetensors"}},
            "pos": {"class_type": "CLIPTextEncode", "inputs": {"text": prompt, "clip": ["ckpt", 1]}},
            "neg": {"class_type": "CLIPTextEncode", "inputs": {"text": NEGATIVE, "clip": ["ckpt", 1]}},
            "latent": {"class_type": "EmptyLatentImage", "inputs": {"width": width, "height": height, "batch_size": 1}},
            "sample": {"class_type": "KSampler", "inputs": {
                "model": ["ckpt", 0], "positive": ["pos", 0], "negative": ["neg", 0], "latent_image": ["latent", 0],
                "seed": seed, "steps": 30, "cfg": 7.0, "sampler_name": "dpmpp_2m", "scheduler": "karras", "denoise": 1.0}},
            "decode": {"class_type": "VAEDecode", "inputs": {"samples": ["sample", 0], "vae": ["ckpt", 2]}},
            "save": save}
    if model == "zimage":
        unet, clip_type, vae, latent, steps, sampler = ("z_image_turbo_bf16.safetensors", "lumina2", "ae.safetensors",
                                                        "EmptySD3LatentImage", 8, "res_multistep")
    elif model == "klein":
        unet, clip_type, vae, latent, steps, sampler = ("flux-2-klein-4b.safetensors", "flux2", "flux2-vae.safetensors",
                                                        "EmptyFlux2LatentImage", 4, "euler")
    else:
        raise ValueError(f"unknown model {model!r}")
    graph = {
        "unet": {"class_type": "UNETLoader", "inputs": {"unet_name": unet, "weight_dtype": "default"}},
        "clip": {"class_type": "CLIPLoader", "inputs": {"clip_name": "qwen_3_4b.safetensors", "type": clip_type}},
        "vae": {"class_type": "VAELoader", "inputs": {"vae_name": vae}},
        "pos": pos,
        "neg": {"class_type": "ConditioningZeroOut", "inputs": {"conditioning": ["pos", 0]}},
        "latent": {"class_type": latent, "inputs": {"width": width, "height": height, "batch_size": 1}},
        "sample": {"class_type": "KSampler", "inputs": {
            "model": ["unet", 0], "positive": ["pos", 0], "negative": ["neg", 0], "latent_image": ["latent", 0],
            "seed": seed, "steps": steps, "cfg": 1.0, "sampler_name": sampler, "scheduler": "simple", "denoise": 1.0}},
        "decode": decode, "save": save}
    if model == "zimage":
        graph["shift"] = {"class_type": "ModelSamplingAuraFlow", "inputs": {"model": ["unet", 0], "shift": 3.0}}
        graph["sample"]["inputs"]["model"] = ["shift", 0]
    return graph


def comfy_command(extra: tuple[str, ...] = ()) -> list[str]:
    """ComfyUI, capped: its own scope, no swap, nothing cached between prompts."""
    return ["systemd-run", "--user", "--scope", "--quiet", "-p", f"MemoryMax={MEMORY_CAP}", "-p", "MemorySwapMax=0",
            str(ROOT / "venv/bin/python"), "main.py", "--listen", "127.0.0.1", "--port", "8188",
            "--disable-auto-launch", "--cache-none", "--disable-pinned-memory", *extra]


def http(url: str, payload: dict | None = None, timeout: float = 30.0) -> dict:
    data = json.dumps(payload).encode() if payload is not None else None
    req = urllib.request.Request(url, data=data, headers={"Content-Type": "application/json"} if data else {})
    with urllib.request.urlopen(req, timeout=timeout) as reply:
        body = reply.read()
    return json.loads(body) if body else {}


def vram_used_gb() -> float:
    """VRAM in use on the 7900 XT, the biggest card."""
    best = 0, 0
    for card in Path("/sys/class/drm").glob("card[0-9]/device"):
        try:
            total = int((card / "mem_info_vram_total").read_text())
            used = int((card / "mem_info_vram_used").read_text())
        except OSError:
            continue
        best = max(best, (total, used))
    return round(best[1] / 1e9, 2)


class VramPeak:
    """Peak VRAM while the ``with`` block runs."""

    def __enter__(self):
        self.peak, self._stop = 0.0, threading.Event()
        self._thread = threading.Thread(target=self._watch, daemon=True)
        self._thread.start()
        return self

    def _watch(self) -> None:
        while not self._stop.wait(0.1):
            self.peak = max(self.peak, vram_used_gb())

    def __exit__(self, *exc) -> None:
        self._stop.set()
        self._thread.join()


class Comfy:
    """A ComfyUI server for the length of a ``with`` block."""

    def __init__(self, log: Path, extra: tuple[str, ...] = ()) -> None:
        self.log, self.extra = log, extra

    def __enter__(self):
        env = {**os.environ, "HIP_VISIBLE_DEVICES": "0"}
        self._out = self.log.open("w")
        self.proc = subprocess.Popen(comfy_command(self.extra), cwd=ROOT / "app", env=env,
                                     stdout=self._out, stderr=subprocess.STDOUT)
        end = time.monotonic() + 180
        while time.monotonic() < end:
            if self.proc.poll() is not None:
                raise RuntimeError(f"ComfyUI exited {self.proc.returncode}; see {self.log}")
            try:
                http(f"{URL}/system_stats", timeout=2)
                return self
            except (urllib.error.URLError, OSError):
                time.sleep(1)
        raise RuntimeError(f"ComfyUI did not come up; see {self.log}")

    def __exit__(self, *exc) -> None:
        self.proc.terminate()
        try:
            self.proc.wait(timeout=15)
        except subprocess.TimeoutExpired:
            self.proc.kill()
        self._out.close()

    def run(self, graph: dict, timeout: float = 600.0) -> dict:
        """Run ``graph``: {"seconds", "image" (path)} or {"seconds", "error"}."""
        start = time.monotonic()
        try:
            prompt_id = http(f"{URL}/prompt", {"prompt": graph})["prompt_id"]
        except urllib.error.HTTPError as exc:
            return {"seconds": 0.0, "error": exc.read().decode()[:500]}
        while time.monotonic() - start < timeout:
            if self.proc.poll() is not None:
                return {"seconds": round(time.monotonic() - start, 2), "error": f"ComfyUI died ({self.proc.returncode})"}
            entry = http(f"{URL}/history/{prompt_id}").get(prompt_id)
            if entry and entry.get("status", {}).get("completed"):
                images = [i for node in entry["outputs"].values() for i in node.get("images", [])]
                image = ROOT / "app/output" / images[0]["subfolder"] / images[0]["filename"] if images else None
                return {"seconds": round(time.monotonic() - start, 2), "image": str(image)}
            if entry and entry.get("status", {}).get("status_str") == "error":
                messages = [m[1].get("exception_message", "") for m in entry["status"].get("messages", [])
                            if m[0] == "execution_error"]
                return {"seconds": round(time.monotonic() - start, 2), "error": " ".join(messages)[:500]}
            time.sleep(0.2)
        return {"seconds": timeout, "error": "timed out"}

    def free(self) -> None:
        """Drop loaded models, so the next run is a cold start."""
        http(f"{URL}/free", {"unload_models": True, "free_memory": True})


def coder_idle() -> bool:
    try:
        return not any(s.get("is_processing") for s in http("http://127.0.0.1:8080/slots", timeout=5))
    except (urllib.error.URLError, OSError, ValueError):
        return True


class CoderStopped:
    """llama-server stopped for the ``with`` block and started again after. Refuses if the coder is busy."""

    def __enter__(self):
        if not coder_idle():
            raise RuntimeError("the coder is busy; try again when it's idle")
        subprocess.run(["systemctl", "--user", "stop", "llama-server"], check=True)
        return self

    def __exit__(self, *exc) -> None:
        subprocess.run(["systemctl", "--user", "start", "llama-server"], check=True)
