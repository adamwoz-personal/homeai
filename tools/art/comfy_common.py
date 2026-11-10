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
import sys
import threading
import time
import urllib.error
import urllib.request
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from homeai import gpu_info, voice_model  # noqa: E402
from homeai.httpjson import OLLAMA, post_json  # noqa: E402,F401 - OLLAMA re-exported for contention_bench

ROOT = Path.home() / ".local/share/homeai/comfyui"
URL = "http://127.0.0.1:8188"
MODELS = ("klein", "klein_base", "zimage", "sdxl", "dreamshaper", "qwen", "qwen_turbo", "krea", "ernie", "hidream", "chroma",
          "moss", "boogu", "flux3", "klein9", "klein9_base", "ideogram", "animagine", "noobai")
# Adam 2026-11-05: SDXL out. 2026-11-06, after the bake-off: Ideogram blanked 39 of 60 ordinary prompts with a safety
# card; FLUX 3 garbled; klein base lost to klein; the anime models missed their prompts.
RETIRED = ("sdxl", "ideogram", "flux3", "klein_base", "animagine", "noobai")
CHROMA_NEGATIVE = ("This low quality greyscale unfinished sketch is inaccurate and flawed. The image is very blurred "
                   "and lacks detail with excessive chromatic aberrations and artifacts.")
ANIME = {"animagine": "animagine-xl-4.0.safetensors", "noobai": "NoobAI-XL-v1.1.safetensors"}
ANIME_NEGATIVE = "lowres, bad anatomy, bad hands, text, watermark, signature, worst quality, low quality"
QWEN_TURBO_SIGMAS = "1.0, 0.978453, 0.95418, 0.926626, 0.89508, 0.845148, 0.704534, 0.414568, 0"
VOICE = voice_model.current()   # Jarvis's model, from ZeroClaw's config
MEMORY_CAP = "24G"
NEGATIVE = "blurry, low quality, watermark, text artefacts, deformed"


def _tail(sampled: list, vae: list, prefix: str) -> dict:
    return {"decode": {"class_type": "VAEDecode", "inputs": {"samples": sampled, "vae": vae}},
            "save": {"class_type": "SaveImage", "inputs": {"images": ["decode", 0], "filename_prefix": prefix}}}


def _ksampler(model: list, pos: list, neg: list, latent: list, seed: int, steps: int, cfg: float, sampler: str,
              scheduler: str) -> dict:
    return {"class_type": "KSampler", "inputs": {
        "model": model, "positive": pos, "negative": neg, "latent_image": latent, "seed": seed, "steps": steps,
        "cfg": cfg, "sampler_name": sampler, "scheduler": scheduler, "denoise": 1.0}}


def _loaders(unet: str, clip: str, clip_type: str, vae: str) -> dict:
    return {"unet": {"class_type": "UNETLoader", "inputs": {"unet_name": unet, "weight_dtype": "default"}},
            "clip": {"class_type": "CLIPLoader", "inputs": {"clip_name": clip, "type": clip_type}},
            "vae": {"class_type": "VAELoader", "inputs": {"vae_name": vae}}}


def _custom_sampler(model: list, pos: list, neg: list, latent: list, seed: int, cfg: float, sigmas: dict) -> dict:
    """SamplerCustomAdvanced with explicit ``sigmas``, as the klein and Qwen turbo templates do."""
    return {"noise": {"class_type": "RandomNoise", "inputs": {"noise_seed": seed}},
            "pick": {"class_type": "KSamplerSelect", "inputs": {"sampler_name": "euler"}},
            "sigmas": sigmas,
            "guider": {"class_type": "CFGGuider", "inputs": {"model": model, "positive": pos, "negative": neg, "cfg": cfg}},
            "sample": {"class_type": "SamplerCustomAdvanced", "inputs": {
                "noise": ["noise", 0], "guider": ["guider", 0], "sampler": ["pick", 0], "sigmas": ["sigmas", 0],
                "latent_image": latent}}}


def _sdxl(checkpoint: str, prompt: str, negative: str, seed: int, width: int, height: int, prefix: str, steps: int,
          cfg: float, sampler: str, scheduler: str) -> dict:
    return {"ckpt": {"class_type": "CheckpointLoaderSimple", "inputs": {"ckpt_name": checkpoint}},
            "pos": {"class_type": "CLIPTextEncode", "inputs": {"text": prompt, "clip": ["ckpt", 1]}},
            "neg": {"class_type": "CLIPTextEncode", "inputs": {"text": negative, "clip": ["ckpt", 1]}},
            "latent": {"class_type": "EmptyLatentImage", "inputs": {"width": width, "height": height, "batch_size": 1}},
            "sample": _ksampler(["ckpt", 0], ["pos", 0], ["neg", 0], ["latent", 0], seed, steps, cfg, sampler, scheduler),
            **_tail(["sample", 0], ["ckpt", 2], prefix)}


def workflow(model: str, prompt: str, seed: int, width: int = 1024, height: int = 1024, prefix: str = "art") -> dict:
    """ComfyUI API-format graph for ``model``, on the settings of ComfyUI's own template for it (inspect_template.py)."""
    size = {"width": width, "height": height, "batch_size": 1}
    pos = {"class_type": "CLIPTextEncode", "inputs": {"text": prompt, "clip": ["clip", 0]}}
    zero = {"class_type": "ConditioningZeroOut", "inputs": {"conditioning": ["pos", 0]}}
    empty_neg = {"class_type": "CLIPTextEncode", "inputs": {"text": "", "clip": ["clip", 0]}}
    if model == "sdxl":
        return _sdxl("sd_xl_base_1.0.safetensors", prompt, NEGATIVE, seed, width, height, prefix, 30, 7.0, "dpmpp_2m", "karras")
    if model == "dreamshaper":
        return _sdxl("DreamShaperXL_Turbo_v2_1.safetensors", prompt, NEGATIVE, seed, width, height, prefix, 8, 2.0, "dpmpp_sde", "karras")
    if model in ANIME:
        return _sdxl(ANIME[model], prompt, ANIME_NEGATIVE, seed, width, height, prefix, 28, 5.0, "euler_ancestral", "normal")
    if model == "zimage":
        return {**_loaders("z_image_turbo_bf16.safetensors", "qwen_3_4b.safetensors", "lumina2", "ae.safetensors"),
                "shift": {"class_type": "ModelSamplingAuraFlow", "inputs": {"model": ["unet", 0], "shift": 3.0}},
                "pos": pos, "neg": zero, "latent": {"class_type": "EmptySD3LatentImage", "inputs": size},
                "sample": _ksampler(["shift", 0], ["pos", 0], ["neg", 0], ["latent", 0], seed, 8, 1.0, "res_multistep", "simple"),
                **_tail(["sample", 0], ["vae", 0], prefix)}
    if model in ("klein", "klein_base"):
        unet, steps, cfg, neg = (("flux-2-klein-4b.safetensors", 4, 1.0, zero) if model == "klein"
                                 else ("flux-2-klein-base-4b.safetensors", 20, 5.0, empty_neg))
        schedule = {"class_type": "Flux2Scheduler", "inputs": {"steps": steps, "width": width, "height": height}}
        return {**_loaders(unet, "qwen_3_4b.safetensors", "flux2", "flux2-vae.safetensors"), "pos": pos, "neg": neg,
                "latent": {"class_type": "EmptyFlux2LatentImage", "inputs": size},
                **_custom_sampler(["unet", 0], ["pos", 0], ["neg", 0], ["latent", 0], seed, cfg, schedule),
                **_tail(["sample", 0], ["vae", 0], prefix)}
    if model in ("qwen", "qwen_turbo"):
        unet = "qwen_image_2.1_int8_convrot.safetensors" if model == "qwen" else "qwen_image_2.1_turbo_int8_convrot.safetensors"
        graph = {**_loaders(unet, "qwen3vl_8b_int8_convrot.safetensors", "qwen_image", "qwen_image_2.1_vae_bf16.safetensors"),
                 "text": {"class_type": "TextEncodeQwenImage21", "inputs": {
                     "clip": ["clip", 0], "prompt": prompt, "negative_prompt": "", "resolution": 1024}},
                 "latent": {"class_type": "EmptyLatentImage", "inputs": size}, **_tail(["sample", 0], ["vae", 0], prefix)}
        if model == "qwen":
            graph["sample"] = _ksampler(["unet", 0], ["text", 0], ["text", 1], ["latent", 0], seed, 25, 1.0, "euler", "simple")
            return graph
        sigmas = {"class_type": "ManualSigmas", "inputs": {"sigmas": QWEN_TURBO_SIGMAS}}
        return {**graph, **_custom_sampler(["unet", 0], ["text", 0], ["text", 1], ["latent", 0], seed, 1.0, sigmas)}
    if model == "krea":
        return {**_loaders("krea2_turbo_int8_convrot.safetensors", "qwen3vl_4b_fp8_scaled.safetensors", "krea2",
                           "qwen_image_vae.safetensors"),
                "pos": pos, "neg": zero, "latent": {"class_type": "EmptyLatentImage", "inputs": size},
                "sample": _ksampler(["unet", 0], ["pos", 0], ["neg", 0], ["latent", 0], seed, 8, 1.0, "euler", "simple"),
                **_tail(["sample", 0], ["vae", 0], prefix)}
    if model == "ernie":
        return {**_loaders("ernie-image.safetensors", "ministral-3-3b.safetensors", "flux2", "flux2-vae.safetensors"),
                "pos": pos, "neg": empty_neg, "latent": {"class_type": "EmptyFlux2LatentImage", "inputs": size},
                "sample": _ksampler(["unet", 0], ["pos", 0], ["neg", 0], ["latent", 0], seed, 20, 4.0, "euler", "simple"),
                **_tail(["sample", 0], ["vae", 0], prefix)}
    if model == "hidream":                                  # trained at ~4 megapixels: lower goes off-distribution
        double = {"width": width * 2 // 32 * 32, "height": height * 2 // 32 * 32, "batch_size": 1}
        return {"ckpt": {"class_type": "CheckpointLoaderSimple", "inputs": {"ckpt_name": "hidream_o1_image_fp8_scaled.safetensors"}},
                "scale": {"class_type": "ModelNoiseScale", "inputs": {"model": ["ckpt", 0], "noise_scale": 8.0}},
                "smooth": {"class_type": "HiDreamO1PatchSeamSmoothing", "inputs": {
                    "model": ["scale", 0], "start_percent": 0.8, "end_percent": 1.0, "pattern": "single_shift",
                    "passes": "ramp_2_4", "blend": "median", "strength": 1.0}},
                "pos": {"class_type": "CLIPTextEncode", "inputs": {"text": prompt, "clip": ["ckpt", 1]}},
                "neg": {"class_type": "CLIPTextEncode", "inputs": {"text": "", "clip": ["ckpt", 1]}},
                "latent": {"class_type": "EmptyHiDreamO1LatentImage", "inputs": double},
                "pick": {"class_type": "KSamplerSelect", "inputs": {"sampler_name": "dpmpp_2m_sde_gpu"}},
                "sigmas": {"class_type": "BasicScheduler", "inputs": {
                    "model": ["scale", 0], "scheduler": "normal", "steps": 40, "denoise": 1.0}},
                "sample": {"class_type": "SamplerCustom", "inputs": {
                    "model": ["smooth", 0], "add_noise": True, "noise_seed": seed, "cfg": 5.0, "positive": ["pos", 0],
                    "negative": ["neg", 0], "sampler": ["pick", 0], "sigmas": ["sigmas", 0], "latent_image": ["latent", 0]}},
                **_tail(["sample", 0], ["ckpt", 2], prefix)}
    if model == "flux3":
        return {**_loaders("flux3-image-klein-base-bf16.safetensors", "qwen3vl_4b_int8_convrot.safetensors", "flux3",
                           "flux2-vae.safetensors"),
                "pos": pos, "neg": empty_neg, "latent": {"class_type": "EmptyFlux2LatentImage", "inputs": size},
                "sample": _ksampler(["unet", 0], ["pos", 0], ["neg", 0], ["latent", 0], seed, 15, 1.0, "er_sde", "simple"),
                **_tail(["sample", 0], ["vae", 0], prefix)}
    if model in ("klein9", "klein9_base"):
        unet, steps, cfg, neg = (("flux-2-klein-9b-fp8.safetensors", 4, 1.0, zero) if model == "klein9"
                                 else ("flux-2-klein-base-9b-fp8.safetensors", 20, 5.0, empty_neg))
        schedule = {"class_type": "Flux2Scheduler", "inputs": {"steps": steps, "width": width, "height": height}}
        return {**_loaders(unet, "qwen_3_8b_fp8mixed.safetensors", "flux2", "flux2-vae.safetensors"), "pos": pos, "neg": neg,
                "latent": {"class_type": "EmptyFlux2LatentImage", "inputs": size},
                **_custom_sampler(["unet", 0], ["pos", 0], ["neg", 0], ["latent", 0], seed, cfg, schedule),
                **_tail(["sample", 0], ["vae", 0], prefix)}
    if model == "ideogram":                                 # the template's Quality preset; prompts are JSON
        text = json.dumps({"high_level_description": prompt})
        return {**_loaders("ideogram4_fp8_scaled.safetensors", "qwen3vl_8b_fp8_scaled.safetensors", "ideogram4", "flux2-vae.safetensors"),
                "uncond": {"class_type": "UNETLoader", "inputs": {
                    "unet_name": "ideogram4_unconditional_fp8_scaled.safetensors", "weight_dtype": "default"}},
                "override": {"class_type": "CFGOverride", "inputs": {
                    "model": ["unet", 0], "cfg": 3.0, "start_percent": 0.7, "end_percent": 1.0}},
                "pos": {"class_type": "CLIPTextEncode", "inputs": {"text": text, "clip": ["clip", 0]}}, "neg": zero,
                "latent": {"class_type": "EmptyFlux2LatentImage", "inputs": size},
                "noise": {"class_type": "RandomNoise", "inputs": {"noise_seed": seed}},
                "pick": {"class_type": "KSamplerSelect", "inputs": {"sampler_name": "euler"}},
                "sigmas": {"class_type": "Ideogram4Scheduler", "inputs": {
                    "steps": 48, "width": width, "height": height, "mu": 0.0, "std": 1.5}},
                "guider": {"class_type": "DualModelGuider", "inputs": {
                    "model": ["override", 0], "model_negative": ["uncond", 0], "positive": ["pos", 0],
                    "negative": ["neg", 0], "cfg": 7.0}},
                "sample": {"class_type": "SamplerCustomAdvanced", "inputs": {
                    "noise": ["noise", 0], "guider": ["guider", 0], "sampler": ["pick", 0], "sigmas": ["sigmas", 0],
                    "latent_image": ["latent", 0]}},
                **_tail(["sample", 0], ["vae", 0], prefix)}
    if model == "chroma":
        shift = ["shift", 0]
        schedule = {"class_type": "BasicScheduler", "inputs": {"model": shift, "scheduler": "beta", "steps": 26, "denoise": 1.0}}
        return {**_loaders("Chroma1-HD-fp8mixed.safetensors", "t5xxl_fp8_e4m3fn_scaled.safetensors", "chroma", "ae.safetensors"),
                "shift": {"class_type": "ModelSamplingAuraFlow", "inputs": {"model": ["unet", 0], "shift": 1.0}},
                "t5": {"class_type": "T5TokenizerOptions", "inputs": {"clip": ["clip", 0], "min_padding": 0, "min_length": 0}},
                "pos": {"class_type": "CLIPTextEncode", "inputs": {"text": prompt, "clip": ["t5", 0]}},
                "neg": {"class_type": "CLIPTextEncode", "inputs": {"text": CHROMA_NEGATIVE, "clip": ["t5", 0]}},
                "latent": {"class_type": "EmptySD3LatentImage", "inputs": size},
                **_custom_sampler(shift, ["pos", 0], ["neg", 0], ["latent", 0], seed, 3.5, schedule),
                **_tail(["sample", 0], ["vae", 0], prefix)}
    if model == "moss":
        return {**_loaders("moss_image_sft_int8_convrot.safetensors", "qwen3.5_9b_w4a8_gptq.safetensors", "moss_image",
                           "flux2-vae_small.safetensors"),
                "slg": {"class_type": "SkipLayerGuidanceDiTSimple", "inputs": {
                    "model": ["unet", 0], "double_layers": "", "single_layers": "22", "start_percent": 0.0, "end_percent": 0.5}},
                "pos": pos, "neg": empty_neg, "latent": {"class_type": "EmptyFlux2LatentImage", "inputs": size},
                "sample": _ksampler(["slg", 0], ["pos", 0], ["neg", 0], ["latent", 0], seed, 25, 2.5, "er_sde", "simple"),
                **_tail(["sample", 0], ["vae", 0], prefix)}
    if model == "boogu":
        return {**_loaders("boogu_image_turbo_fp8_scaled.safetensors", "qwen3vl_8b_fp8_scaled.safetensors", "boogu", "ae.safetensors"),
                "pos": pos, "neg": zero, "latent": {"class_type": "EmptyLatentImage", "inputs": size},
                "sample": _ksampler(["unet", 0], ["pos", 0], ["neg", 0], ["latent", 0], seed, 4, 1.0, "lcm", "sgm_uniform"),
                **_tail(["sample", 0], ["vae", 0], prefix)}
    raise ValueError(f"unknown model {model!r}")


def comfy_command(extra: tuple[str, ...] = ()) -> list[str]:
    """ComfyUI, capped: its own scope and no swap. No --cache-none: it reloads the model for every image."""
    return ["systemd-run", "--user", "--scope", "--quiet", "-p", f"MemoryMax={MEMORY_CAP}", "-p", "MemorySwapMax=0",
            str(ROOT / "venv/bin/python"), "main.py", "--listen", "127.0.0.1", "--port", "8188",
            "--disable-auto-launch", "--disable-pinned-memory", *extra]


http = post_json


def vram_used_gb() -> float:
    """VRAM in use on the 7900 XT, the biggest card."""
    card = gpu_info.biggest()
    return round(card.used / 1e9, 2) if card else 0.0


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

    def run(self, graph: dict, timeout: float = 1800.0) -> dict:
        """Run ``graph``: {"seconds", "image" (path)} or {"seconds", "error"}."""
        start = time.monotonic()
        try:
            prompt_id = http(f"{URL}/prompt", {"prompt": graph})["prompt_id"]
        except urllib.error.HTTPError as exc:
            return {"seconds": 0.0, "error": exc.read().decode()[:500]}
        while time.monotonic() - start < timeout:
            if self.proc.poll() is not None:
                return {"seconds": round(time.monotonic() - start, 2), "error": f"ComfyUI died ({self.proc.returncode})"}
            try:
                entry = http(f"{URL}/history/{prompt_id}").get(prompt_id)
            except (urllib.error.URLError, OSError):
                time.sleep(1)                           # dying: the poll above reports it
                continue
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


def CoderStopped(voice_off: bool = False):  # noqa: N802 - kept the name the tools use
    """The coder stopped for the ``with`` block, then voice mode restored: ``gpu_mode.ModeSwitcher.coder_paused``.

    ``voice_off`` also stops Jarvis and unloads the voice model, for models that need the whole card (Adam agreed).
    """
    from homeai import gpu_mode as G
    return G.ModeSwitcher(G.System(), log=lambda line: print("  ", line, flush=True)).coder_paused(voice_off)
