#!/usr/bin/env bash
# Install ComfyUI and the bake-off models into ~/.local/share/homeai/comfyui (AVATAR_PLAN 3.4, 7.1).
#
# Why: the art modes need an image runner. It gets its own Python 3.12 venv with
# PyTorch's ROCm wheels, because the system has no ROCm and the project runs 3.14.
# Pinned here so a rebuild gets the same thing. Safe to re-run: downloads resume.
#
# Usage: tools/art/install_comfy.sh [--no-models]
set -euo pipefail
COMFY_TAG=v0.44.0
ROCM=rocm7.2
ROOT="$HOME/.local/share/homeai/comfyui"
APP="$ROOT/app"
HF=https://huggingface.co

mkdir -p "$ROOT"
if [ ! -d "$APP/.git" ]; then
  git clone --depth 1 --branch "$COMFY_TAG" https://github.com/comfyanonymous/ComfyUI "$APP"
fi
if [ ! -x "$ROOT/venv/bin/python" ]; then
  uv venv --python 3.12 "$ROOT/venv"
fi
uv pip install --python "$ROOT/venv/bin/python" torch torchvision torchaudio \
  --index-url "https://download.pytorch.org/whl/$ROCM"
grep -v -E '^(torch|torchvision|torchaudio)([<>=~ ]|$)' "$APP/requirements.txt" > "$ROOT/requirements.notorch.txt"
uv pip install --python "$ROOT/venv/bin/python" -r "$ROOT/requirements.notorch.txt"
"$ROOT/venv/bin/python" -c 'import torch; print("torch", torch.__version__, "hip", torch.version.hip)'

[ "${1:-}" = "--no-models" ] && exit 0
get() {  # repo, path in repo, folder under models/
  local name; name=$(basename "$2")
  mkdir -p "$APP/models/$3"
  curl -fL --retry 5 -C - -o "$APP/models/$3/$name" "$HF/$1/resolve/main/$2"
}
gated() {  # the same, with Adam's Hugging Face token: read from its file, never on a command line
  local name; name=$(basename "$2")
  mkdir -p "$APP/models/$3"
  curl -fL --retry 5 -C - -K <(printf 'header = "Authorization: Bearer %s"\n' "$(cat "$HOME/.cache/huggingface/token")") \
    -o "$APP/models/$3/$name" "$HF/$1/resolve/main/$2"
}
get Comfy-Org/Flux-2-Klein-4B split_files/diffusion_models/flux-2-klein-4b.safetensors diffusion_models
get Comfy-Org/Flux-2-Klein-4B split_files/text_encoders/qwen_3_4b.safetensors text_encoders
get Comfy-Org/Flux-2-Klein-4B split_files/vae/flux2-vae.safetensors vae
get Comfy-Org/Z-Image-Turbo split_files/diffusion_models/z_image_turbo_bf16.safetensors diffusion_models
get Comfy-Org/Z-Image-Turbo split_files/vae/ae.safetensors vae
get stabilityai/stable-diffusion-xl-base-1.0 sd_xl_base_1.0.safetensors checkpoints
# Second bake-off round: highly rated open models that fit a 20 GB card (ungated repos only).
get Comfy-Org/Flux-2-Klein-4B split_files/diffusion_models/flux-2-klein-base-4b.safetensors diffusion_models
get Comfy-Org/Qwen-Image-2.1 diffusion_models/qwen_image_2.1_int8_convrot.safetensors diffusion_models
get Comfy-Org/Qwen-Image-2.1 diffusion_models/qwen_image_2.1_turbo_int8_convrot.safetensors diffusion_models
get Comfy-Org/Qwen-Image-2.1 text_encoders/qwen3vl_8b_int8_convrot.safetensors text_encoders
get Comfy-Org/Qwen-Image-2.1 vae/qwen_image_2.1_vae_bf16.safetensors vae
get Comfy-Org/Krea-2 diffusion_models/krea2_turbo_int8_convrot.safetensors diffusion_models
get Comfy-Org/Krea-2 text_encoders/qwen3vl_4b_fp8_scaled.safetensors text_encoders
get Comfy-Org/Krea-2 vae/qwen_image_vae.safetensors vae
get Comfy-Org/ERNIE-Image diffusion_models/ernie-image.safetensors diffusion_models
get Comfy-Org/ERNIE-Image text_encoders/ministral-3-3b.safetensors text_encoders
get cagliostrolab/animagine-xl-4.0 animagine-xl-4.0.safetensors checkpoints
get Laxhar/noobai-XL-1.1 NoobAI-XL-v1.1.safetensors checkpoints
get Comfy-Org/HiDream-O1-Image checkpoints/hidream_o1_image_fp8_scaled.safetensors checkpoints
get Comfy-Org/Chroma1-HD split_files/diffusion_models/Chroma1-HD-fp8mixed.safetensors diffusion_models
get comfyanonymous/flux_text_encoders t5xxl_fp8_e4m3fn_scaled.safetensors text_encoders
get Comfy-Org/MOSS-Image diffusion_models/moss_image_sft_int8_convrot.safetensors diffusion_models
get Comfy-Org/MOSS-Image text_encoders/qwen3.5_9b_w4a8_gptq.safetensors text_encoders
get Comfy-Org/MOSS-Image vae/flux2-vae_small.safetensors vae
get Comfy-Org/Boogu-Image diffusion_models/boogu_image_turbo_fp8_scaled.safetensors diffusion_models
get Comfy-Org/Boogu-Image text_encoders/qwen3vl_8b_fp8_scaled.safetensors text_encoders
get Lykon/dreamshaper-xl-v2-turbo DreamShaperXL_Turbo_v2_1.safetensors checkpoints
# Gated: needs the terms accepted on each model page and a token in ~/.cache/huggingface/token.
gated black-forest-labs/flux-3-image-klein-comfy diffusion_models/flux3-image-klein-base-bf16.safetensors diffusion_models
gated black-forest-labs/flux-3-image-klein-comfy text_encoders/qwen3vl_4b_int8_convrot.safetensors text_encoders
# klein 9B: the fp8 files. The full 18 GB ones go over the 24 GB RAM cap while loading.
gated black-forest-labs/FLUX.2-klein-9b-fp8 flux-2-klein-9b-fp8.safetensors diffusion_models
gated black-forest-labs/FLUX.2-klein-base-9b-fp8 flux-2-klein-base-9b-fp8.safetensors diffusion_models
get Comfy-Org/Flux-2-Klein-9B split_files/text_encoders/qwen_3_8b_fp8mixed.safetensors text_encoders
gated Comfy-Org/Ideogram-4 diffusion_models/ideogram4_fp8_scaled.safetensors diffusion_models
gated Comfy-Org/Ideogram-4 diffusion_models/ideogram4_unconditional_fp8_scaled.safetensors diffusion_models

