"""ComfyUI graphs for art mode (AVATAR_PLAN §7.1).

Only the default for now: klein 9B distilled text-to-image, the bake-off winner
with Jarvis on (6.6 s warm, 2026-11-06). The graph is the one the bake-off used
(tools/art/comfy_common.py), checked equal by tests/test_art_session.py.
"""
from __future__ import annotations

KLEIN9_UNET = "flux-2-klein-9b-fp8.safetensors"
KLEIN9_CLIP = "qwen_3_8b_fp8mixed.safetensors"
FLUX2_VAE = "flux2-vae.safetensors"
STEPS = 4


def klein9(prompt: str, seed: int, width: int = 1024, height: int = 1024, prefix: str = "art") -> dict:
    size = {"width": width, "height": height, "batch_size": 1}
    return {
        "unet": {"class_type": "UNETLoader", "inputs": {"unet_name": KLEIN9_UNET, "weight_dtype": "default"}},
        "clip": {"class_type": "CLIPLoader", "inputs": {"clip_name": KLEIN9_CLIP, "type": "flux2"}},
        "vae": {"class_type": "VAELoader", "inputs": {"vae_name": FLUX2_VAE}},
        "pos": {"class_type": "CLIPTextEncode", "inputs": {"text": prompt, "clip": ["clip", 0]}},
        "neg": {"class_type": "ConditioningZeroOut", "inputs": {"conditioning": ["pos", 0]}},
        "latent": {"class_type": "EmptyFlux2LatentImage", "inputs": size},
        "noise": {"class_type": "RandomNoise", "inputs": {"noise_seed": seed}},
        "pick": {"class_type": "KSamplerSelect", "inputs": {"sampler_name": "euler"}},
        "sigmas": {"class_type": "Flux2Scheduler", "inputs": {"steps": STEPS, "width": width, "height": height}},
        "guider": {"class_type": "CFGGuider", "inputs": {"model": ["unet", 0], "positive": ["pos", 0],
                                                          "negative": ["neg", 0], "cfg": 1.0}},
        "sample": {"class_type": "SamplerCustomAdvanced", "inputs": {
            "noise": ["noise", 0], "guider": ["guider", 0], "sampler": ["pick", 0], "sigmas": ["sigmas", 0],
            "latent_image": ["latent", 0]}},
        "decode": {"class_type": "VAEDecode", "inputs": {"samples": ["sample", 0], "vae": ["vae", 0]}},
        "save": {"class_type": "SaveImage", "inputs": {"images": ["decode", 0], "filename_prefix": prefix}},
    }
